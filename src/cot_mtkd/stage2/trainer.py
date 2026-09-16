from __future__ import annotations

import logging
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

from ..data.collator import LongCoTCollator, shifted_token_views
from ..data.dataset import JsonlRecordDataset, load_jsonl_files
from ..data.prepare import tokenizer_fingerprint
from ..data.schema import TokenRegion
from ..models.chunked_head import (
    decoder_and_lm_head,
    forward_hidden,
    gather_hidden_positions,
)
from ..models.multi_adapter import (
    adapter_parameter_map,
    create_student_model,
    extract_adapter_state,
    load_adapter_bundle,
    load_adapter_state,
    load_tokenizer,
    require_same_model_source,
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
    files_fingerprint,
    fingerprint,
    read_json,
    require_file_sha256,
    runtime_metadata,
    write_config_snapshot,
    write_json,
)
from ..utils.seed import derived_seed, deterministic_rng
from ..utils.training import (
    add_gradients_,
    assign_gradients,
    cosine_warmup_lambda,
    global_clip_grad_list_,
    zeros_like_parameters,
)
from .cache import SparseTeacherCache
from .losses import dual_source_hidden_gradients

LOGGER = logging.getLogger(__name__)


def _batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: (
            value.to(device, non_blocking=True)
            if isinstance(value, torch.Tensor)
            else value
        )
        for key, value in batch.items()
    }


def _stable_config(config: dict[str, Any]) -> dict[str, Any]:
    value = {key: item for key, item in config.items() if not key.startswith("_")}
    value = {
        key: (dict(item) if isinstance(item, dict) else item)
        for key, item in value.items()
    }
    if isinstance(value.get("stage2"), dict):
        value["stage2"].pop("resume_from", None)
    return value


def _validate_stage2_config(config: dict[str, Any]) -> None:
    if str(config["optimizer"].get("name", "")).lower() != "adamw":
        raise ValueError("Stage 2 currently implements optimizer.name: adamw")
    if str(config["scheduler"].get("name", "")).lower() != "cosine":
        raise ValueError("Stage 2 currently implements scheduler.name: cosine")
    if int(config["stage2"]["epochs"]) <= 0:
        raise ValueError("stage2.epochs must be positive")
    if float(config["stage2"]["learning_rate"]) <= 0.0:
        raise ValueError("stage2.learning_rate must be positive")
    if float(config["stage2"]["max_grad_norm"]) <= 0.0:
        raise ValueError("stage2.max_grad_norm must be positive")
    hard = float(config["stage2"]["hard_loss_weight"])
    kd = float(config["stage2"]["kd_loss_weight"])
    if hard < 0.0 or kd < 0.0 or hard + kd == 0.0:
        raise ValueError(
            "Stage-2 source weights must be non-negative and not both zero"
        )
    if float(config["cache"]["temperature"]) <= 0.0:
        raise ValueError("cache.temperature must be positive")
    epsilon = float(config["cache"]["clamp_epsilon"])
    if not (0.0 < epsilon < 1.0):
        raise ValueError("cache.clamp_epsilon must be in (0, 1)")
    if int(config["runtime"]["lm_head_chunk_tokens"]) <= 0:
        raise ValueError("runtime.lm_head_chunk_tokens must be positive")


def _hard_weights_for_batch(
    batch: dict[str, Any], signals: dict[str, dict[str, Any]]
) -> torch.Tensor:
    rows: list[torch.Tensor] = []
    labels = batch["labels"][:, 1:]
    regions = batch["region_ids"][:, 1:]
    steps = batch["step_ids"][:, 1:]
    for row, sample_id in enumerate(batch["sample_ids"]):
        valid = labels[row].ne(-100)
        importance = torch.tensor(
            signals[sample_id]["importance"], device=labels.device, dtype=torch.float32
        )
        if not torch.isfinite(importance).all() or (importance <= 0).any():
            raise ValueError(
                f"Hard importance weights must be finite and positive for {sample_id}"
            )
        current_regions = regions[row, valid]
        current_steps = steps[row, valid]
        current = torch.ones(
            current_regions.shape, device=labels.device, dtype=torch.float32
        )
        weighted = current_regions.eq(int(TokenRegion.REASONING)) | current_regions.eq(
            int(TokenRegion.DELIMITER)
        )
        if weighted.any():
            selected_steps = current_steps[weighted]
            if (
                selected_steps.min().item() < 0
                or selected_steps.max().item() >= importance.numel()
            ):
                raise RuntimeError(f"Invalid step id in prepared record {sample_id}")
            current[weighted] = importance[selected_steps]
        rows.append(current)
    return torch.cat(rows)


def _cache_for_batch(
    batch: dict[str, Any], cache: SparseTeacherCache, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    top_ids, probabilities, tail = [], [], []
    labels = batch["labels"][:, 1:]
    for row, sample_id in enumerate(batch["sample_ids"]):
        expected_positions = labels[row].ne(-100).nonzero(as_tuple=True)[0].cpu() + 1
        value = cache.get(sample_id)
        if not torch.equal(value["token_positions"].long(), expected_positions.long()):
            raise RuntimeError(
                f"Sparse cache token positions do not match prepared record {sample_id}"
            )
        top_ids.append(value["top_ids"])
        probabilities.append(value["top_probabilities"])
        tail.append(value["tail_mass"])
    return (
        torch.cat(top_ids).to(device, non_blocking=True),
        torch.cat(probabilities).to(device, non_blocking=True),
        torch.cat(tail).to(device, non_blocking=True),
    )


def _save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    global_step: int,
    epoch: int,
    batch_in_epoch: int,
    run_fingerprint: str,
    base_seed: int,
    world_size: int,
    source_loss_scalars: dict[str, float] | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "student_state": extract_adapter_state(model, "student"),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "global_step": global_step,
            "epoch": epoch,
            "batch_in_epoch": batch_in_epoch,
            "run_fingerprint": run_fingerprint,
            "source_loss_scalars": source_loss_scalars,
            "sampler_state": {
                "seed": base_seed,
                "epoch": epoch,
                "next_batch_in_epoch": batch_in_epoch,
                "world_size": world_size,
            },
            "dropout_rng": {
                "scheme": "sha256(base_seed,stage2,global_step,microbatch,rank)",
                "base_seed": base_seed,
            },
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
        },
        temporary,
    )
    temporary.replace(path)


def _load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    expected_run_fingerprint: str,
) -> tuple[int, int, int, dict[str, float] | None]:
    value = torch.load(path, map_location="cpu", weights_only=False)
    if value["run_fingerprint"] != expected_run_fingerprint:
        raise RuntimeError("Refusing to resume Stage 2 from a different configuration")
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
        int(value["epoch"]),
        int(value["batch_in_epoch"]),
        value.get("source_loss_scalars"),
    )


def train_stage2(
    config: dict[str, Any], distributed: DistributedContext
) -> dict[str, Any]:
    _validate_stage2_config(config)
    prepared_dir = Path(config["paths"]["prepared"])
    stage1_dir = Path(config["paths"]["stage1"])
    supervision_dir = Path(config["paths"]["supervision"])
    cache_dir = Path(config["paths"]["cache"])
    output_dir = Path(config["paths"]["output"])
    output_dir.mkdir(parents=True, exist_ok=True)
    config_snapshot = output_dir / "config.yaml"
    if distributed.is_main:
        write_config_snapshot(config_snapshot, _stable_config(config))
    barrier()
    prepared_manifest = read_json(prepared_dir / "manifest.json")
    stage1_manifest = read_json(stage1_dir / "manifest.json")
    supervision_manifest = read_json(supervision_dir / "manifest.json")
    cache_manifest = read_json(cache_dir / "manifest.json")
    require_file_sha256(
        prepared_dir, prepared_manifest, "data_file", "data_file_sha256"
    )
    require_file_sha256(
        prepared_dir, prepared_manifest, "config_file", "config_file_sha256"
    )
    require_file_sha256(
        stage1_dir, stage1_manifest, "adapter_bundle", "adapter_bundle_sha256"
    )
    require_file_sha256(
        stage1_dir, stage1_manifest, "config_file", "config_file_sha256"
    )
    require_file_sha256(
        supervision_dir,
        supervision_manifest,
        "config_file",
        "config_file_sha256",
    )
    require_file_sha256(cache_dir, cache_manifest, "index_file", "index_file_sha256")
    require_file_sha256(cache_dir, cache_manifest, "config_file", "config_file_sha256")
    signal_paths = [supervision_dir / name for name in supervision_manifest["shards"]]
    if files_fingerprint(signal_paths) != supervision_manifest["signals_fingerprint"]:
        raise RuntimeError("Supervision shard content does not match its manifest")
    cache_paths = [cache_dir / name for name in cache_manifest["shards"]]
    if files_fingerprint(cache_paths) != cache_manifest["cache_files_fingerprint"]:
        raise RuntimeError("Teacher-cache shard content does not match its manifest")
    if cache_manifest["prepared_manifest_fingerprint"] != fingerprint(
        prepared_manifest
    ):
        raise RuntimeError("Teacher cache/prepared dataset mismatch")
    if cache_manifest["stage1_manifest_fingerprint"] != fingerprint(stage1_manifest):
        raise RuntimeError("Teacher cache/Stage-1 mismatch")
    if cache_manifest["supervision_manifest_fingerprint"] != fingerprint(
        supervision_manifest
    ):
        raise RuntimeError("Teacher cache/supervision mismatch")
    if (
        cache_manifest["teacher_weighting_fingerprint"]
        != supervision_manifest["signals_fingerprint"]
    ):
        raise RuntimeError("Teacher cache was compiled with different teacher weights")
    if float(cache_manifest["temperature"]) != float(config["cache"]["temperature"]):
        raise RuntimeError(
            "Teacher cache temperature differs from Stage-2 configuration"
        )
    if int(cache_manifest["top_k"]) != int(config["cache"]["top_k"]):
        raise RuntimeError("Teacher cache K differs from Stage-2 configuration")

    tokenizer = load_tokenizer(config["model"])
    if tokenizer_fingerprint(tokenizer) != prepared_manifest["tokenizer_fingerprint"]:
        raise RuntimeError("Stage-2 tokenizer does not match the prepared dataset")
    require_same_model_source(
        config["model"], stage1_manifest["config"]["model"], "Stage 2/Stage 1"
    )
    dataset = JsonlRecordDataset(prepared_dir)
    if len(dataset) % distributed.world_size != 0:
        raise ValueError(
            "The prepared dataset size must be divisible by world_size so the "
            "distributed sampler never pads with duplicate samples"
        )
    signals = load_jsonl_files(
        supervision_dir / name for name in supervision_manifest["shards"]
    )
    if len(signals) != len(dataset):
        raise RuntimeError("Supervision records do not cover the Stage-2 dataset")
    cache = SparseTeacherCache(cache_dir)
    if len(cache.index) != len(dataset):
        raise RuntimeError("Sparse teacher cache does not cover the Stage-2 dataset")
    sampler = DistributedSampler(
        dataset,
        num_replicas=distributed.world_size,
        rank=distributed.rank,
        shuffle=True,
        seed=int(config["seed"]),
        drop_last=False,
    )
    micro_batch = int(config["stage2"]["micro_batch_size"])
    if micro_batch <= 0:
        raise ValueError("micro_batch_size must be positive")
    samples_per_rank = len(dataset) // distributed.world_size
    if samples_per_rank % micro_batch != 0:
        raise ValueError(
            "Samples per rank must be divisible by micro_batch_size so every "
            "optimizer update has an exact, deterministic sample count"
        )
    dataloader = DataLoader(
        dataset,
        batch_size=micro_batch,
        sampler=sampler,
        shuffle=False,
        collate_fn=LongCoTCollator(tokenizer.pad_token_id),
        num_workers=int(config["runtime"]["dataloader_workers"]),
        pin_memory=bool(config["runtime"]["pin_memory"]),
        drop_last=False,
    )
    global_batch = int(config["stage2"]["global_batch_size"])
    if global_batch <= 0:
        raise ValueError("global_batch_size must be positive")
    configured_accumulation = config["stage2"].get("gradient_accumulation_steps")
    if configured_accumulation is None:
        divisor = micro_batch * distributed.world_size
        if global_batch % divisor != 0:
            raise ValueError(
                "global_batch_size must be divisible by micro_batch_size * world_size"
            )
        accumulation_steps = global_batch // divisor
    else:
        accumulation_steps = int(configured_accumulation)
        if accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        if accumulation_steps * micro_batch * distributed.world_size != global_batch:
            raise ValueError(
                "gradient_accumulation_steps does not produce requested global batch"
            )
    epochs = int(config["stage2"]["epochs"])
    total_steps = math.ceil(epochs * len(dataloader) / accumulation_steps)

    model = create_student_model(
        config["model"], config["lora"], distributed.device, int(config["seed"])
    )
    medoid_name = str(supervision_manifest["functional_medoid_adapter"])
    stage1_bundle = load_adapter_bundle(stage1_dir / stage1_manifest["adapter_bundle"])
    load_adapter_state(model, "student", stage1_bundle[medoid_name])
    parameters = list(adapter_parameter_map(model, "student").values())
    optimizer_config = config["optimizer"]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(config["stage2"]["learning_rate"]),
        betas=tuple(float(value) for value in optimizer_config["betas"]),
        eps=float(optimizer_config["eps"]),
        weight_decay=float(optimizer_config["weight_decay"]),
    )
    scheduler_config = config["scheduler"]
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: cosine_warmup_lambda(
            step,
            total_steps,
            float(scheduler_config["warmup_ratio"]),
            float(scheduler_config.get("min_lr_ratio", 0.0)),
        ),
    )
    config_hash = fingerprint(_stable_config(config))
    run_hash = fingerprint(
        {
            "config_fingerprint": config_hash,
            "prepared_manifest_fingerprint": fingerprint(prepared_manifest),
            "stage1_manifest_fingerprint": fingerprint(stage1_manifest),
            "supervision_manifest_fingerprint": fingerprint(supervision_manifest),
            "cache_manifest_fingerprint": fingerprint(cache_manifest),
            "world_size": distributed.world_size,
        }
    )
    global_step = start_epoch = start_batch = 0
    last_source_metrics: dict[str, float] | None = None
    resume = config["stage2"].get("resume_from")
    if resume:
        global_step, start_epoch, start_batch, last_source_metrics = _load_checkpoint(
            Path(resume), model, optimizer, scheduler, run_hash
        )
        LOGGER.info(
            "Resumed Stage 2 at step=%d epoch=%d batch=%d",
            global_step,
            start_epoch,
            start_batch,
        )
        if start_epoch >= epochs:
            manifest_path = output_dir / "manifest.json"
            if not manifest_path.is_file():
                raise RuntimeError("Completed Stage-2 checkpoint has no final manifest")
            barrier()
            return read_json(manifest_path)
    metrics = JsonlLogger(
        output_dir / "metrics.jsonl",
        enabled=distributed.is_main,
        truncate=not bool(resume),
    )
    chunk_tokens = int(config["runtime"]["lm_head_chunk_tokens"])
    model.train()

    # Keep partial accumulation across epoch boundaries. This implements the
    # optimizer-step schedule on the complete 3-epoch sample stream, yielding
    # ceil(3,000 / 8) = 375 canonical updates.
    hard_buffer = zeros_like_parameters(parameters)
    kd_buffer = zeros_like_parameters(parameters)
    hard_weight_count = kd_token_count = 0.0
    hard_loss_total = kd_loss_total = 0.0
    accumulated = 0

    for epoch in range(start_epoch, epochs):
        sampler.set_epoch(epoch)
        for batch_index, raw_batch in enumerate(dataloader):
            if epoch == start_epoch and batch_index < start_batch:
                continue
            batch = _batch_to_device(raw_batch, distributed.device)
            views = shifted_token_views(batch)
            top_ids, teacher_probabilities, teacher_tail = _cache_for_batch(
                batch, cache, distributed.device
            )
            hard_weights = _hard_weights_for_batch(batch, signals)
            rng_seed = derived_seed(
                config["seed"], "stage2", global_step, accumulated, distributed.rank
            )
            with deterministic_rng(rng_seed, distributed.device):
                outputs = forward_hidden(
                    model, batch["input_ids"], batch["attention_mask"], use_cache=False
                )
            response_hidden = gather_hidden_positions(
                outputs.last_hidden_state,
                views["response_batch_indices"],
                views["response_hidden_indices"],
            )
            _, head = decoder_and_lm_head(model)
            hard_dhidden, kd_dhidden, hard_sum, kd_sum, hard_count, kd_count = (
                dual_source_hidden_gradients(
                    response_hidden,
                    head,
                    views["response_targets"],
                    hard_weights,
                    top_ids,
                    teacher_probabilities,
                    teacher_tail,
                    float(config["cache"]["temperature"]),
                    float(config["cache"]["clamp_epsilon"]),
                    chunk_tokens,
                )
            )
            hard_gradients = torch.autograd.grad(
                response_hidden,
                parameters,
                grad_outputs=hard_dhidden,
                retain_graph=True,
                allow_unused=False,
            )
            kd_gradients = torch.autograd.grad(
                response_hidden,
                parameters,
                grad_outputs=kd_dhidden,
                retain_graph=False,
                allow_unused=False,
            )
            add_gradients_(
                hard_buffer, [value.detach().float() for value in hard_gradients]
            )
            add_gradients_(
                kd_buffer, [value.detach().float() for value in kd_gradients]
            )
            hard_loss_total += hard_sum
            kd_loss_total += kd_sum
            hard_weight_count += hard_count
            kd_token_count += kd_count
            accumulated += 1
            final_microbatch = epoch + 1 == epochs and batch_index + 1 == len(
                dataloader
            )
            boundary = accumulated == accumulation_steps or final_microbatch
            if not boundary:
                continue

            all_reduce_grad_lists([hard_buffer, kd_buffer])
            counts = torch.tensor(
                [hard_weight_count, kd_token_count],
                device=distributed.device,
                dtype=torch.float64,
            )
            losses = torch.tensor(
                [hard_loss_total, kd_loss_total],
                device=distributed.device,
                dtype=torch.float64,
            )
            all_reduce_tensor(counts)
            all_reduce_tensor(losses)
            hard_mean = float(losses[0].item() / max(counts[0].item(), 1.0))
            kd_mean = float(losses[1].item() / max(counts[1].item(), 1.0))
            last_source_metrics = {
                "hard_loss": hard_mean,
                "kd_loss": kd_mean,
                "total_loss": (
                    float(config["stage2"]["hard_loss_weight"]) * hard_mean
                    + float(config["stage2"]["kd_loss_weight"]) * kd_mean
                ),
            }
            hard_scale = float(config["stage2"]["hard_loss_weight"]) / max(
                float(counts[0].item()), 1.0
            )
            kd_scale = float(config["stage2"]["kd_loss_weight"]) / max(
                float(counts[1].item()), 1.0
            )
            final_gradients = [
                hard.float() * hard_scale + kd.float() * kd_scale
                for hard, kd in zip(hard_buffer, kd_buffer, strict=True)
            ]
            preclip_norm = global_clip_grad_list_(
                final_gradients, float(config["stage2"]["max_grad_norm"])
            )
            assign_gradients(parameters, final_gradients)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            if (
                distributed.is_main
                and global_step % int(config["stage2"]["log_every_steps"]) == 0
            ):
                metrics.log(
                    "stage2_step",
                    step=global_step,
                    epoch=epoch,
                    **last_source_metrics,
                    learning_rate=optimizer.param_groups[0]["lr"],
                    preclip_gradient_norm=preclip_norm,
                )
            if (
                distributed.is_main
                and global_step % int(config["stage2"]["checkpoint_every_steps"]) == 0
            ):
                _save_checkpoint(
                    output_dir / "checkpoint.pt",
                    model,
                    optimizer,
                    scheduler,
                    global_step,
                    epoch,
                    batch_index + 1,
                    run_hash,
                    int(config["seed"]),
                    distributed.world_size,
                    last_source_metrics,
                )
            hard_buffer = zeros_like_parameters(parameters)
            kd_buffer = zeros_like_parameters(parameters)
            hard_weight_count = kd_token_count = 0.0
            hard_loss_total = kd_loss_total = 0.0
            accumulated = 0
        start_batch = 0
        barrier()

    if distributed.is_main:
        final_dir = output_dir / "final"
        bundle_path = save_adapter_bundle(model, ["student"], final_dir)
        _save_checkpoint(
            output_dir / "checkpoint.pt",
            model,
            optimizer,
            scheduler,
            global_step,
            epochs,
            0,
            run_hash,
            int(config["seed"]),
            distributed.world_size,
            last_source_metrics,
        )
        public_config = _stable_config(config)
        manifest = {
            "schema_version": 1,
            "artifact": "stage2_checkpoint",
            "global_step": global_step,
            "training_checkpoint": "checkpoint.pt",
            "training_checkpoint_sha256": file_sha256(output_dir / "checkpoint.pt"),
            "student_adapter": "student",
            "parent_medoid_adapter": medoid_name,
            "adapter_bundle": str(bundle_path.relative_to(output_dir)),
            "adapter_bundle_sha256": file_sha256(bundle_path),
            "prepared_manifest_fingerprint": fingerprint(prepared_manifest),
            "stage1_manifest_fingerprint": fingerprint(stage1_manifest),
            "supervision_manifest_fingerprint": fingerprint(supervision_manifest),
            "cache_manifest_fingerprint": fingerprint(cache_manifest),
            "config": public_config,
            "config_fingerprint": config_hash,
            "config_file": config_snapshot.name,
            "config_file_sha256": file_sha256(config_snapshot),
            "run_fingerprint": run_hash,
            "source_loss_scalars": last_source_metrics,
            "metrics_file": "metrics.jsonl",
            "metrics_file_sha256": file_sha256(output_dir / "metrics.jsonl"),
            "runtime": runtime_metadata(config["_project_root"]),
        }
        write_json(output_dir / "manifest.json", manifest)
    barrier()
    return read_json(output_dir / "manifest.json")
