from __future__ import annotations

import logging
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

from ..data.dataset import JsonlRecordDataset
from ..data.prepare import tokenizer_fingerprint
from ..models.multi_adapter import (
    extract_adapter_state,
    load_adapter_state,
    load_tokenizer,
    save_adapter_bundle,
)
from ..utils.distributed import (
    DistributedContext,
    all_reduce_grad_lists,
    all_reduce_tensor,
    barrier,
)
from ..utils.local_logging import JsonlLogger
from ..utils.manifest import (
    file_sha256,
    fingerprint,
    read_json,
    require_file_sha256,
    runtime_metadata,
    write_config_snapshot,
    write_json,
)
from ..utils.training import (
    add_gradients_,
    assign_gradients,
    cosine_warmup_lambda,
    global_clip_grad_list_,
    zeros_like_parameters,
)
from .online import compute_record_gradient, create_online_model
from .performance import TrainingPerformanceLogger, performance_log_filename
from .step_logging import (
    create_reasoning_logger,
    log_record_steps,
    reasoning_log_filename,
    trim_jsonl_to_checkpoint,
)

LOGGER = logging.getLogger(__name__)
METHOD = "task_anchored_gradient_geometry_mtkd"
OUTPUT_SPACE_METHOD = "disagreement_adaptive_distribution_aggregation_mtkd"


def _stable_config(config: dict[str, Any]) -> dict[str, Any]:
    value = {key: item for key, item in config.items() if not key.startswith("_")}
    value = {key: dict(item) if isinstance(item, dict) else item for key, item in value.items()}
    if isinstance(value.get("stage2"), dict):
        value["stage2"].pop("resume_from", None)
        if config.get("method") == OUTPUT_SPACE_METHOD:
            value["stage2"].setdefault("incomplete_batch_policy", "drop")
    return value


def _validate_stage2_config(config: dict[str, Any]) -> None:
    method = config.get("method")
    if method not in (METHOD, OUTPUT_SPACE_METHOD):
        raise ValueError(
            "Use an output-space or task-geometry Stage-2 config; "
            "legacy dual-source caches are unsupported"
        )
    if str(config["optimizer"].get("name", "")).lower() != "adamw":
        raise ValueError("Phase 2 implements optimizer.name: adamw")
    if str(config["scheduler"].get("name", "")).lower() != "cosine":
        raise ValueError("Phase 2 implements scheduler.name: cosine")
    if float(config["lora"]["dropout"]) != 0.0:
        raise ValueError("Phase 2 requires lora.dropout: 0.0")
    for key in (
        "epochs",
        "micro_batch_size",
        "global_batch_size",
        "max_length",
        "checkpoint_every_steps",
        "log_every_steps",
    ):
        if int(config["stage2"][key]) <= 0:
            raise ValueError(f"stage2.{key} must be positive")
    for key in ("learning_rate", "max_grad_norm"):
        value = float(config["stage2"][key])
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"stage2.{key} must be finite and positive")
    section = "aggregation" if method == OUTPUT_SPACE_METHOD else "geometry"
    selector = config[section]
    selector_keys = (
        ("temperature",)
        if method == OUTPUT_SPACE_METHOD
        else ("temperature", "epsilon_a", "epsilon_u")
    )
    for key in selector_keys:
        value = float(selector[key])
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{section}.{key} must be finite and positive")
    if method != OUTPUT_SPACE_METHOD and selector.get("teacher_execution") != "online_full_vocab":
        raise ValueError("Phase 2 requires online_full_vocab teacher execution")
    if int(config["runtime"]["lm_head_chunk_tokens"]) <= 0:
        raise ValueError("runtime.lm_head_chunk_tokens must be positive")
    if method == OUTPUT_SPACE_METHOD:
        from .council_cache import validate_council_config

        validate_council_config(config)
        if config["stage2"].get("incomplete_batch_policy", "drop") != "drop":
            raise ValueError(
                "Calibrated Phase 2 uses incomplete_batch_policy: drop for exact global batches"
            )
    if "hard_loss_weight" in config["stage2"] or "kd_loss_weight" in config["stage2"]:
        raise ValueError("Old hard/KD source weights are not part of the new Phase-2 objective")
    if not isinstance(config.get("logging", {}).get("reasoning_steps", True), bool):
        raise TypeError("logging.reasoning_steps must be a boolean")
    if not isinstance(config.get("logging", {}).get("performance", True), bool):
        raise TypeError("logging.performance must be a boolean")


def gradient_accumulation_steps(stage2: dict[str, Any], world_size: int) -> int:
    """Derive the exact global batch; reject fractional/nondivisible settings."""

    def positive_integer(value, name):
        if isinstance(value, bool) or int(value) != value or int(value) <= 0:
            raise ValueError(f"{name} must be a positive integer")
        return int(value)

    micro = positive_integer(stage2["micro_batch_size"], "micro_batch_size")
    global_batch = positive_integer(stage2["global_batch_size"], "global_batch_size")
    world = positive_integer(world_size, "world_size")
    divisor = micro * world
    if global_batch % divisor:
        raise ValueError("global_batch_size must be divisible by micro_batch_size * world_size")
    expected = global_batch // divisor
    explicit = stage2.get("gradient_accumulation_steps")
    if (
        explicit is not None
        and positive_integer(explicit, "gradient_accumulation_steps") != expected
    ):
        raise ValueError("Invalid gradient_accumulation_steps for global_batch_size")
    return expected


def _record_collator(records):
    return records


def full_global_batch_plan(record_count: int, stage2: dict, world_size: int) -> tuple[int, int]:
    """Return per-rank full-window microbatches and the global omitted tail count."""
    accumulation = gradient_accumulation_steps(stage2, world_size)
    if record_count % world_size:
        raise ValueError("Prepared record count must divide world_size without sampler padding")
    windows = record_count // int(stage2["global_batch_size"])
    if not windows:
        raise ValueError(
            "Training corpus is smaller than one full global batch; use a smaller dry-run batch"
        )
    return windows * accumulation, record_count - windows * int(stage2["global_batch_size"])


def _save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    global_step,
    data_step,
    epoch,
    batch_in_epoch,
    run_fingerprint,
    base_seed,
    world_size,
    metrics,
    method: str = METHOD,
    calibration: dict | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "method": method,
            "student_state": extract_adapter_state(model, "student"),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "global_step": global_step,
            "data_step": data_step,
            "epoch": epoch,
            "batch_in_epoch": batch_in_epoch,
            "run_fingerprint": run_fingerprint,
            "metrics": metrics,
            "disagreement_calibration": calibration,
            "sampler_state": {
                "seed": base_seed,
                "epoch": epoch,
                "next_batch_in_epoch": batch_in_epoch,
                "world_size": world_size,
            },
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        temporary,
    )
    temporary.replace(path)


def _load_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    expected_run_fingerprint,
    method=METHOD,
    expected_calibration: dict | None = None,
):
    value = torch.load(path, map_location="cpu", weights_only=False)
    if value.get("method") != method or value["run_fingerprint"] != expected_run_fingerprint:
        raise RuntimeError("Refusing to resume Phase 2 from a different method or configuration")
    if (
        expected_calibration is not None
        and value.get("disagreement_calibration") != expected_calibration
    ):
        raise RuntimeError(
            "Refusing Phase-2 checkpoint with missing/different fitted tau; start a new run"
        )
    load_adapter_state(model, "student", value["student_state"])
    optimizer.load_state_dict(value["optimizer"])
    scheduler.load_state_dict(value["scheduler"])
    random.setstate(value["python_rng"])
    np.random.set_state(value["numpy_rng"])
    torch.set_rng_state(value["torch_rng"])
    if torch.cuda.is_available() and value["cuda_rng"] is not None:
        torch.cuda.set_rng_state_all(value["cuda_rng"])
    return (
        int(value["global_step"]),
        int(value["data_step"]),
        int(value["epoch"]),
        int(value["batch_in_epoch"]),
        value.get("metrics"),
    )


def apply_accumulated_update(
    parameters,
    gradients,
    optimizer,
    scheduler,
    sample_count: int,
    active_steps: int,
    max_grad_norm: float,
) -> tuple[bool, float]:
    """Globally reduced sample sum, including zero-signal examples in its mean.

    A zero-signal window advances data only, leaving AdamW state and LR intact.
    """
    if sample_count <= 0:
        raise ValueError("An effective batch must contain at least one example")
    if active_steps == 0:
        optimizer.zero_grad(set_to_none=True)
        return False, 0.0
    for gradient in gradients:
        gradient.div_(sample_count)
        if not torch.isfinite(gradient).all():
            raise FloatingPointError("Non-finite accumulated Phase-2 gradient")
    norm = global_clip_grad_list_(gradients, max_grad_norm)
    if not math.isfinite(norm):
        raise FloatingPointError("Phase-2 gradient norm overflowed before the update")
    assign_gradients(parameters, gradients)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    return True, norm


def train_stage2(config: dict[str, Any], distributed: DistributedContext) -> dict[str, Any]:
    _validate_stage2_config(config)
    method = config["method"]
    if method == OUTPUT_SPACE_METHOD:
        from .output_space import compute_record_gradient as record_gradient
    else:
        record_gradient = compute_record_gradient
    paths = config["paths"]
    prepared_dir, stage1_dir = Path(paths["prepared"]), Path(paths["stage1"])
    output_dir = Path(paths["output"])
    prepared = read_json(prepared_dir / "manifest.json")
    from .teachers import ensure_stage2_teachers

    stage1 = ensure_stage2_teachers(config)
    from .council_cache import load_council_cache

    cache = load_council_cache(config, prepared, stage1)
    council = cache.manifest
    data_path = require_file_sha256(prepared_dir, prepared, "data_file", "data_file_sha256")
    for root, manifest, pairs in (
        (
            prepared_dir,
            prepared,
            [("config_file", "config_file_sha256")],
        ),
        (
            stage1_dir,
            stage1,
            [("adapter_bundle", "adapter_bundle_sha256"), ("config_file", "config_file_sha256")],
        ),
    ):
        for file_key, hash_key in pairs:
            require_file_sha256(root, manifest, file_key, hash_key)
    tokenizer = load_tokenizer(config["model"])
    if tokenizer_fingerprint(tokenizer) != prepared["tokenizer_fingerprint"]:
        raise RuntimeError("Phase-2 tokenizer does not match prepared tokens")
    dataset = JsonlRecordDataset(data_path)
    if len(dataset) != int(prepared["records"]) or not len(dataset):
        raise RuntimeError("Prepared dataset record count is invalid")
    if len(dataset) % distributed.world_size:
        raise ValueError("Prepared record count must divide world_size without sampler padding")
    micro = int(config["stage2"]["micro_batch_size"])
    accumulation = gradient_accumulation_steps(config["stage2"], distributed.world_size)
    sampler = DistributedSampler(
        dataset,
        num_replicas=distributed.world_size,
        rank=distributed.rank,
        shuffle=True,
        seed=int(config["seed"]),
        drop_last=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=micro,
        sampler=sampler,
        collate_fn=_record_collator,
        num_workers=int(config["runtime"]["dataloader_workers"]),
        pin_memory=False,
        drop_last=False,
    )
    epochs = int(config["stage2"]["epochs"])
    batches_per_epoch = len(loader)
    dropped_tail_examples = 0
    if method == OUTPUT_SPACE_METHOD:
        batches_per_epoch, dropped_tail_examples = full_global_batch_plan(
            len(dataset),
            config["stage2"],
            distributed.world_size,
        )
    total_steps = math.ceil(epochs * batches_per_epoch / accumulation)
    public_config = _stable_config(config)
    config_hash = fingerprint(public_config)
    run_hash = fingerprint(
        {
            "method": method,
            "config_fingerprint": config_hash,
            "prepared_manifest_fingerprint": fingerprint(prepared),
            "stage1_manifest_fingerprint": fingerprint(stage1),
            "council_cache_fingerprint": council["fingerprint"],
            "world_size": distributed.world_size,
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    if distributed.is_main:
        write_config_snapshot(output_dir / "config.yaml", public_config)
    barrier()
    if method == OUTPUT_SPACE_METHOD:
        from .initialization import create_cached_student

        model, adapter_names, parameters = create_cached_student(config, distributed, cache)
    else:
        model, adapter_names, parameters = create_online_model(config, distributed)
    opt = config["optimizer"]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(config["stage2"]["learning_rate"]),
        betas=tuple(opt["betas"]),
        eps=float(opt["eps"]),
        weight_decay=float(opt["weight_decay"]),
    )
    sched = config["scheduler"]
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: cosine_warmup_lambda(
            step, total_steps, float(sched["warmup_ratio"]), float(sched["min_lr_ratio"])
        ),
    )
    global_step = data_step = start_epoch = start_batch = 0
    last_metrics = None
    resume = config["stage2"].get("resume_from")
    if resume:
        global_step, data_step, start_epoch, start_batch, last_metrics = _load_checkpoint(
            resume,
            model,
            optimizer,
            scheduler,
            run_hash,
            method=method,
            expected_calibration=council["calibration"] if method == OUTPUT_SPACE_METHOD else None,
        )
        if start_epoch >= epochs:
            barrier()
            return read_json(output_dir / "manifest.json")
    if resume and distributed.is_main and method == OUTPUT_SPACE_METHOD:
        trim_jsonl_to_checkpoint(
            output_dir / "metrics.jsonl", run_hash, data_step, before_update=False
        )
    logger = JsonlLogger(
        output_dir / "metrics.jsonl", enabled=distributed.is_main, truncate=not bool(resume)
    )
    logger.log(
        "stage2_council_cache",
        run_fingerprint=run_hash,
        cache_status="hit",
        cache_fingerprint=council["fingerprint"],
        preprocessing_wall_seconds=council["preprocessing_wall_seconds"],
        disk_bytes=council["disk_bytes"],
        samples=council["records"],
        tokens=council["tokens"],
        steps=council["steps"],
        expert_sft_scores=council["expert_sft_scores"],
        selected_expert=council["selected_expert"],
        tie_breaking=council["tie_breaking"],
        numerical_anomalies=council["numerical_anomalies"],
        calibration=council["calibration"],
        diagnostics=council["diagnostics"],
        training_examples_per_epoch=len(dataset) - dropped_tail_examples,
        dropped_tail_examples_per_epoch=dropped_tail_examples,
    )
    if method == OUTPUT_SPACE_METHOD:
        LOGGER.info(
            "Phase 2 epochs=%d global_batch_size=%d micro=%d world=%d accumulation=%d "
            "temperature=%g sft_weight=%g disagreement_pooling_power=%g tau_quantile=%g tau=%g",
            epochs,
            int(config["stage2"]["global_batch_size"]),
            micro,
            distributed.world_size,
            accumulation,
            float(config["aggregation"]["temperature"]),
            float(config["aggregation"]["sft_weight"]),
            float(config["aggregation"]["disagreement_pooling_power"]),
            float(config["aggregation"]["tau_quantile"]),
            council["calibration"]["tau"],
        )
        if dropped_tail_examples:
            LOGGER.info(
                "Exact global batches: optimize %d/%d shuffled training examples per epoch; "
                "omit final %d-example incomplete window. Tau still uses the full training corpus.",
                len(dataset) - dropped_tail_examples,
                len(dataset),
                dropped_tail_examples,
            )
    step_logger = None
    if method == OUTPUT_SPACE_METHOD and config.get("logging", {}).get("reasoning_steps", True):
        step_logger = create_reasoning_logger(
            output_dir, distributed.rank, distributed.world_size, run_hash, data_step, bool(resume)
        )
    performance = None
    if method == OUTPUT_SPACE_METHOD and config.get("logging", {}).get("performance", True):
        performance = TrainingPerformanceLogger(
            output_dir,
            distributed.device,
            distributed.rank,
            distributed.world_size,
            run_hash,
            config_hash,
            public_config,
            data_step,
            bool(resume),
        )
    buffer = zeros_like_parameters(parameters)
    # Example/step counts and losses; reduction follows fixed K then example mean.
    # The remaining slots hold method statistics and KD/SFT/tail diagnostics.
    totals = torch.zeros(19, device=distributed.device, dtype=torch.float64)
    accumulated = 0
    for epoch in range(start_epoch, epochs):
        sampler.set_epoch(epoch)
        for batch_index, records in enumerate(loader):
            if batch_index >= batches_per_epoch:
                break
            if epoch == start_epoch and batch_index < start_batch:
                continue
            for sample_index, record in enumerate(records):
                started = performance.begin_sample() if performance is not None else None
                result = record_gradient(
                    model,
                    adapter_names,
                    parameters,
                    record,
                    tokenizer,
                    config,
                    distributed.device,
                    **(
                        {"cached_target": cache.get(record.sample_id)}
                        if method == OUTPUT_SPACE_METHOD
                        else {}
                    ),
                )
                record_seconds = time.perf_counter() - started if started is not None else None
                if step_logger is not None:
                    log_record_steps(
                        step_logger,
                        result,
                        record.sample_id,
                        run_fingerprint=run_hash,
                        rank=distributed.rank,
                        epoch=epoch,
                        batch_in_epoch=batch_index,
                        sample_in_batch=sample_index,
                        data_step_before=data_step,
                        global_step_before=global_step,
                    )
                add_gradients_(buffer, result.gradients)
                totals += totals.new_tensor(
                    [
                        1,
                        result.steps,
                        result.active_steps,
                        result.discarded_steps,
                        result.loss,
                        result.metrics.get(
                            "disagreement_sum"
                            if method == OUTPUT_SPACE_METHOD
                            else "agreement_sum",
                            0.0,
                        ),
                        result.metrics.get(
                            "rho_sum" if method == OUTPUT_SPACE_METHOD else "weight_sum", 0.0
                        ),
                        result.metrics.get("consensus_sum", 0.0),
                        result.metrics.get("anchor_loss_sum", 0.0),
                        result.metrics.get("kd_loss", result.loss),
                        result.metrics.get("sft_loss", 0.0),
                        result.metrics.get("weighted_sft_loss", 0.0),
                        result.metrics.get("student_tail_mass", 0.0),
                        result.metrics.get("target_tail_mass", 0.0),
                        result.metrics.get("target_entropy", 0.0),
                        result.metrics.get("tail_probability_clamps", 0),
                        result.metrics.get("tail_roundoff_corrections", 0),
                        result.metrics.get("tail_complement_fallbacks", 0),
                        result.metrics.get("token_js_sum", 0),
                    ]
                )
                if performance is not None:
                    performance.log_sample(
                        record.sample_id,
                        record_seconds,
                        result,
                        epoch=epoch,
                        batch_in_epoch=batch_index,
                        sample_in_batch=sample_index,
                        data_step_before=data_step,
                        global_step_before=global_step,
                        prepared_tokens=len(record.input_ids),
                    )
                del result
            accumulated += 1
            final_batch = epoch + 1 == epochs and batch_index + 1 == batches_per_epoch
            if accumulated < accumulation and not final_batch:
                continue
            all_reduce_grad_lists([buffer])
            all_reduce_tensor(totals)
            counts = totals.cpu().tolist()
            if method == OUTPUT_SPACE_METHOD and int(counts[0]) != int(
                config["stage2"]["global_batch_size"]
            ):
                raise RuntimeError(
                    "Phase-2 optimizer window does not match configured global batch size"
                )
            updated, norm = apply_accumulated_update(
                parameters,
                buffer,
                optimizer,
                scheduler,
                int(counts[0]),
                int(counts[2]),
                float(config["stage2"]["max_grad_norm"]),
            )
            data_step += 1
            global_step += int(updated)
            performance_summary = {}
            if performance is not None:
                performance_summary = performance.log_update(
                    data_step,
                    global_step,
                    total_steps,
                    epoch=epoch,
                    global_examples=int(counts[0]),
                    skipped_update=not updated,
                )
            last_metrics = {
                "kd_loss": counts[9] / counts[0],
                "total_loss": counts[4] / counts[0],
                "examples": int(counts[0]),
                "reasoning_steps": int(counts[1]),
                "active_steps": int(counts[2]),
                "discarded_steps": int(counts[3]),
                "skipped_update": not updated,
            }
            if method == OUTPUT_SPACE_METHOD:
                last_metrics.update(
                    configured_global_batch_size=int(config["stage2"]["global_batch_size"]),
                    effective_global_batch_size=int(counts[0]),
                    partial_batch=int(counts[0]) != int(config["stage2"]["global_batch_size"]),
                    sft_loss=counts[10] / counts[0],
                    weighted_sft_loss=counts[11] / counts[0],
                    student_tail_mass=counts[12] / counts[0],
                    student_support_mass=1 - counts[12] / counts[0],
                    target_tail_mass=counts[13] / counts[0],
                    target_support_mass=1 - counts[13] / counts[0],
                    target_entropy=counts[14] / counts[0],
                    tail_probability_clamps=int(counts[15]),
                    tail_roundoff_corrections=int(counts[16]),
                    tail_complement_fallbacks=int(counts[17]),
                    temperature=float(config["aggregation"]["temperature"]),
                    js_temperature=float(config["aggregation"]["temperature"]),
                    kd_temperature=float(config["aggregation"]["temperature"]),
                    tau=council["calibration"]["tau"],
                    tau_quantile=council["calibration"]["tau_quantile"],
                    disagreement_pooling_power=council["calibration"]["disagreement_pooling_power"],
                    step_disagreement_mean=counts[5] / max(counts[1], 1),
                    rho_mean=counts[6] / max(counts[1], 1),
                    token_js_mean=counts[18] / max(counts[1], 1),
                )
            else:
                last_metrics.update(
                    agreement_mean=counts[5] / max(counts[1], 1),
                    utility_weight_mean=counts[6] / max(counts[1], 1),
                    consensus_weight_mean=counts[7] / max(counts[1], 1),
                    anchor_ce_mean=counts[8] / max(counts[1], 1),
                )
            if data_step % int(config["stage2"]["log_every_steps"]) == 0:
                logger.log(
                    "stage2_step",
                    run_fingerprint=run_hash,
                    step=global_step,
                    data_step=data_step,
                    epoch=epoch,
                    learning_rate=optimizer.param_groups[0]["lr"],
                    preclip_gradient_norm=norm,
                    **performance_summary,
                    **last_metrics,
                )
                LOGGER.info(
                    "Phase 2 batch=%d update=%d loss=%.6f active=%d/%d lr=%.3e%s%s%s",
                    data_step,
                    global_step,
                    last_metrics["total_loss"],
                    int(counts[2]),
                    int(counts[1]),
                    optimizer.param_groups[0]["lr"],
                    (
                        f" Delta={last_metrics['step_disagreement_mean']:.6g} nats rho={last_metrics['rho_mean']:.6g}"
                        if method == OUTPUT_SPACE_METHOD
                        else ""
                    ),
                    " (skip)" if not updated else "",
                    (
                        f" wall={performance_summary['update_window_wall_seconds']:.1f}s"
                        + (
                            f" peak={performance_summary['peak_allocated_gib']:.1f} GiB"
                            if performance_summary["peak_allocated_gib"] is not None
                            else ""
                        )
                        if performance_summary
                        else ""
                    ),
                )
            if (
                distributed.is_main
                and data_step % int(config["stage2"]["checkpoint_every_steps"]) == 0
            ):
                _save_checkpoint(
                    output_dir / "checkpoint.pt",
                    model,
                    optimizer,
                    scheduler,
                    global_step,
                    data_step,
                    epoch,
                    batch_index + 1,
                    run_hash,
                    int(config["seed"]),
                    distributed.world_size,
                    last_metrics,
                    method=method,
                    calibration=council["calibration"] if method == OUTPUT_SPACE_METHOD else None,
                )
            buffer = zeros_like_parameters(parameters)
            totals.zero_()
            accumulated = 0
        start_batch = 0
        barrier()
    if distributed.is_main:
        bundle = save_adapter_bundle(model, ["student"], output_dir / "final")
        checkpoint = output_dir / "checkpoint.pt"
        _save_checkpoint(
            checkpoint,
            model,
            optimizer,
            scheduler,
            global_step,
            data_step,
            epochs,
            0,
            run_hash,
            int(config["seed"]),
            distributed.world_size,
            last_metrics,
            method=method,
            calibration=council["calibration"] if method == OUTPUT_SPACE_METHOD else None,
        )
        manifest = {
            "schema_version": 2,
            "artifact": "stage2_checkpoint",
            "method": method,
            "global_step": global_step,
            "data_step": data_step,
            "training_checkpoint": checkpoint.name,
            "training_checkpoint_sha256": file_sha256(checkpoint),
            "student_adapter": "student",
            "initial_expert_adapter": council["selected_expert"],
            "expert_sft_scores": council["expert_sft_scores"],
            "adapter_bundle": str(bundle.relative_to(output_dir)),
            "adapter_bundle_sha256": file_sha256(bundle),
            "prepared_manifest_fingerprint": fingerprint(prepared),
            "stage1_manifest_fingerprint": fingerprint(stage1),
            "council_cache_fingerprint": council["fingerprint"],
            "config": public_config,
            "config_fingerprint": config_hash,
            "config_file": "config.yaml",
            "config_file_sha256": file_sha256(output_dir / "config.yaml"),
            "run_fingerprint": run_hash,
            "loss_scalars": last_metrics,
            "metrics_file": "metrics.jsonl",
            "metrics_file_sha256": file_sha256(output_dir / "metrics.jsonl"),
            "runtime": runtime_metadata(config["_project_root"]),
        }
        if step_logger is not None:
            manifest["reasoning_step_logs"] = [
                {
                    "rank": rank,
                    "file": reasoning_log_filename(rank, distributed.world_size),
                    "sha256": file_sha256(
                        output_dir / reasoning_log_filename(rank, distributed.world_size)
                    ),
                }
                for rank in range(distributed.world_size)
            ]
        if method == OUTPUT_SPACE_METHOD:
            from .diagnostics import summarize_reasoning_logs

            diagnostics = {
                "artifact": "stage2_calibrated_diagnostics",
                "run_fingerprint": run_hash,
                "calibration": council["calibration"],
                "training_corpus_cache": council["diagnostics"],
                "observed_training": summarize_reasoning_logs(
                    [
                        output_dir / reasoning_log_filename(rank, distributed.world_size)
                        for rank in range(distributed.world_size)
                    ],
                    run_hash,
                )
                if step_logger is not None
                else {"available": False, "reason": "reasoning_steps logging disabled"},
            }
            write_json(output_dir / "diagnostics.json", diagnostics)
            manifest.update(
                disagreement_calibration=council["calibration"],
                council_cache_version=council["cache_version"],
                diagnostics_file="diagnostics.json",
                diagnostics_file_sha256=file_sha256(output_dir / "diagnostics.json"),
                training_examples_per_epoch=len(dataset) - dropped_tail_examples,
                dropped_tail_examples_per_epoch=dropped_tail_examples,
                incomplete_batch_policy="drop",
            )
        if performance is not None:
            manifest["performance_logs"] = [
                {
                    "rank": rank,
                    "file": performance_log_filename(rank, distributed.world_size),
                    "sha256": file_sha256(
                        output_dir / performance_log_filename(rank, distributed.world_size)
                    ),
                }
                for rank in range(distributed.world_size)
            ]
        write_json(output_dir / "manifest.json", manifest)
    barrier()
    return read_json(output_dir / "manifest.json")
