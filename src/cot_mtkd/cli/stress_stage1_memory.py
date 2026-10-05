from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from ..config import load_config
from ..data.collator import LongCoTCollator, shifted_token_views
from ..data.dataset import JsonlRecordDataset
from ..data.prepare import tokenizer_fingerprint
from ..data.schema import PreparedRecord, TokenRegion
from ..models.multi_adapter import (
    adapter_parameter_groups,
    create_multi_adapter_model,
    load_tokenizer,
    require_same_model_source,
    set_all_adapters_trainable,
)
from ..stage1.gac_gradient import stable_gac_gradients
from ..stage1.rbf import (
    BandwidthEMA,
    effective_update_distances,
    interaction_bandwidths,
    rbf_kernel,
    repulsion_updates,
)
from ..stage1.trainer import (
    _batch_to_device,
    _optimizer_and_scheduler,
    _pairwise_statistics,
    _run_compatible_config,
    _validate_stage1_config,
    one_pass_expert_gradients,
    sft_only_expert_gradients,
)
from ..utils.manifest import fingerprint, read_json, require_file_sha256
from ..utils.seed import seed_everything
from ..utils.training import (
    add_gradients_,
    assign_gradients,
    global_clip_grad_list_,
    zeros_like_parameters,
)

TARGET_LENGTH = 32_768
GIB = 1024**3
PHASES = ("sft_only", "full_interaction")


def extend_reasoning(record: PreparedRecord, target_length: int) -> PreparedRecord:
    """Insert real reasoning tokens before its delimiter to reach the context limit."""
    extra = target_length - len(record.input_ids)
    if extra < 0:
        raise ValueError("The longest real sample exceeds the synthetic target length")
    if extra == 0:
        return replace(record, sample_id=f"{record.sample_id}-length-{target_length}")
    reasoning_positions = [
        index
        for index, (region, label, step) in enumerate(
            zip(record.region_ids, record.labels, record.step_ids, strict=True)
        )
        if region == int(TokenRegion.REASONING) and label != -100 and step >= 0
    ]
    if not reasoning_positions:
        raise ValueError("Cannot extend a sample with no labeled reasoning tokens")
    insertion = reasoning_positions[-1] + 1
    source_steps = [record.step_ids[index] for index in reasoning_positions]
    step_span = max(source_steps) - min(source_steps) + 1
    next_step = max(record.step_ids) + 1
    inserted_positions = [
        reasoning_positions[index % len(reasoning_positions)] for index in range(extra)
    ]
    inserted_ids = [record.input_ids[index] for index in inserted_positions]
    inserted_steps = [
        next_step
        + (index // len(reasoning_positions)) * step_span
        + source_steps[index % len(reasoning_positions)]
        - min(source_steps)
        for index in range(extra)
    ]

    def insert(values: list[Any], additions: list[Any]) -> list[Any]:
        return values[:insertion] + additions + values[insertion:]

    synthetic = replace(
        record,
        sample_id=f"{record.sample_id}-synthetic-{target_length}",
        input_ids=insert(record.input_ids, inserted_ids),
        labels=insert(record.labels, inserted_ids),
        attention_mask=insert(record.attention_mask, [1] * extra),
        offset_mapping=insert(record.offset_mapping, [(0, 0)] * extra),
        region_ids=insert(record.region_ids, [int(TokenRegion.REASONING)] * extra),
        step_ids=insert(record.step_ids, inserted_steps),
        original_length=target_length,
        kept_length=target_length,
        truncated=False,
        answer_start=record.answer_start + extra,
    )
    assert len(synthetic.input_ids) == target_length
    return synthetic


def _run_microbatch(
    phase: str,
    record: PreparedRecord,
    collator: LongCoTCollator,
    model: torch.nn.Module,
    names: list[str],
    groups: list[dict[str, torch.nn.Parameter]],
    parameters: list[list[torch.nn.Parameter]],
    optimizers: list[torch.optim.Optimizer],
    schedulers: list[torch.optim.lr_scheduler.LRScheduler],
    bandwidth: BandwidthEMA,
    config: dict[str, Any],
    device: torch.device,
    iteration: int,
) -> dict[str, Any]:
    """Exercise one optimizer update using the trainer's phase-specific path.

    This is a single-sample optimizer window. Its token/sample normalizers
    match training, but its time is not a global-batch optimizer-step time.
    """
    if phase not in PHASES:
        raise ValueError(f"Unknown Stage-1 stress phase: {phase}")
    task_buffers = [zeros_like_parameters(current) for current in parameters]
    batch = _batch_to_device(collator([record]), device)
    views = shifted_token_views(batch)
    sft_tokens = int(views["response_targets"].numel())
    dpp_samples = int(torch.unique(views["reasoning_batch_indices"]).numel())
    if sft_tokens <= 0:
        raise ValueError("Stress sample has no labeled response tokens")
    combined_dpp_scale = None
    if phase == "sft_only":
        results = (
            sft_only_expert_gradients(
                model,
                name,
                expert,
                current,
                batch,
                iteration,
                iteration,
                int(config["seed"]),
                int(config["runtime"]["lm_head_chunk_tokens"]),
                device,
            )
            for expert, (name, current) in enumerate(zip(names, parameters, strict=True))
        )
    else:
        combined_dpp_scale = (
            float(config["stage1"]["dpp_weight"]) * sft_tokens / dpp_samples
            if dpp_samples else 0.0
        )
        probe, results = one_pass_expert_gradients(
            model,
            names,
            parameters,
            batch,
            iteration,
            iteration,
            int(config["seed"]),
            config,
            device,
            combined_dpp_scale=combined_dpp_scale,
        )
        if probe.dpp_sample_count != dpp_samples:
            raise RuntimeError("Stress DPP sample count differs from its planned denominator")
    for expert, (task, _dpp, _, _) in enumerate(results):
        add_gradients_(task_buffers[expert], task)
    for values in task_buffers:
        for value in values:
            value.div_(sft_tokens)
    details: dict[str, Any] = {}
    if phase == "sft_only":
        final = task_buffers
    else:
        set_all_adapters_trainable(model, names)
        scaling = float(config["lora"]["alpha"]) / float(config["lora"]["rank"])
        distances_for_step = effective_update_distances(groups, scaling)
        h_base = bandwidth.update(distances_for_step)
        h_gac, h_rbf = interaction_bandwidths(
            h_base,
            float(config["stage1"].get("gac_bandwidth_scale", 0.5)),
            float(config["stage1"].get("rbf_bandwidth_scale", 1.0)),
        )
        gac_kernel = rbf_kernel(distances_for_step.detach(), h_gac)
        repulsion, repulsion_kernel, distances = repulsion_updates(
            groups, scaling, h_rbf, distances=distances_for_step
        )
        final, diagnostics = stable_gac_gradients(
            task_buffers, repulsion, gac_kernel,
            beta=float(config["stage1"].get("gac_beta", 0.5)),
            rbf_weight=float(config["stage1"]["rbf_weight"]),
        )
        details = {
            "h_base": h_base, "h_gac": h_gac, "h_rbf": h_rbf,
            "pairwise_distances": _pairwise_statistics(distances, names),
            "gac_pairwise_kernels": _pairwise_statistics(gac_kernel, names),
            "rbf_pairwise_kernels": _pairwise_statistics(repulsion_kernel, names),
            "gac_cross_coefficients": diagnostics.cross_coefficients,
            "gac_self_coefficients": diagnostics.self_coefficients,
            "repulsion_norms_before_cap": diagnostics.repulsion_norms_before_cap,
            "repulsion_cap_factors": diagnostics.repulsion_cap_factors,
            "capped_repulsion_norms": diagnostics.capped_repulsion_norms,
            "weighted_repulsion_norms": diagnostics.weighted_repulsion_norms,
        }
    for values, current, optimizer, scheduler in zip(
        final, parameters, optimizers, schedulers, strict=True
    ):
        norm = global_clip_grad_list_(values, float(config["stage1"]["max_grad_norm"]))
        if not math.isfinite(norm):
            raise FloatingPointError(f"Stress {phase} path produced a non-finite gradient norm")
        assign_gradients(current, values)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
    return {
        "response_tokens": sft_tokens,
        "dpp_samples": dpp_samples if phase != "sft_only" else 0,
        "combined_dpp_scale": combined_dpp_scale,
        **details,
    }


def run_case(
    name: str,
    phase: str,
    record: PreparedRecord,
    collator: LongCoTCollator,
    model: torch.nn.Module,
    names: list[str],
    groups: list[dict[str, torch.nn.Parameter]],
    parameters: list[list[torch.nn.Parameter]],
    optimizers: list[torch.optim.Optimizer],
    schedulers: list[torch.optim.lr_scheduler.LRScheduler],
    bandwidth: BandwidthEMA,
    config: dict[str, Any],
    device: torch.device,
    *,
    warmup: int = 1,
    repetitions: int = 1,
    first_iteration: int = 0,
) -> dict[str, Any]:
    if phase not in PHASES:
        raise ValueError(f"Unknown Stage-1 stress phase: {phase}")
    if warmup < 0 or repetitions <= 0:
        raise ValueError("Stress warmup must be nonnegative and repetitions must be positive")
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    measured_times: list[float] = []
    completed_warmup = 0
    details: dict[str, Any] = {}
    status = "ok"
    failed_iteration_kind = None
    for index in range(warmup + repetitions):
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        try:
            details = _run_microbatch(
                phase,
                record,
                collator,
                model,
                names,
                groups,
                parameters,
                optimizers,
                schedulers,
                bandwidth,
                config,
                device,
                first_iteration + index,
            )
            torch.cuda.synchronize(device)
        except torch.cuda.OutOfMemoryError:
            status = "cuda_oom"
            failed_iteration_kind = "warmup" if index < warmup else "measured"
            break
        elapsed = time.perf_counter() - start
        if index < warmup:
            completed_warmup += 1
        else:
            measured_times.append(elapsed)
    # Reset only once per case so warm-up allocation spikes also count toward
    # the preflight decision, even when subsequent measured iterations fit.
    allocated = torch.cuda.max_memory_allocated(device)
    reserved = torch.cuda.max_memory_reserved(device)
    return {
        "case": name,
        "case_id": f"{name}_{phase}",
        "status": status,
        "update_mode": phase,
        "sample_id": record.sample_id,
        "sequence_length": len(record.input_ids),
        **details,
        "micro_batch_size": 1,
        "training_global_batch_size": int(config["stage1"]["global_batch_size"]),
        "timing_unit": "one_microbatch_with_optimizer_step",
        "warmup_iterations": warmup,
        "measured_iterations": repetitions,
        "completed_warmup_iterations": completed_warmup,
        "completed_measured_iterations": len(measured_times),
        "failed_iteration_kind": failed_iteration_kind,
        "memory_includes_warmup": True,
        "max_memory_allocated_bytes": allocated,
        "max_memory_reserved_bytes": reserved,
        "max_memory_allocated_gib": round(allocated / GIB, 3),
        "max_memory_reserved_gib": round(reserved / GIB, 3),
        "measured_elapsed_seconds": [round(value, 3) for value in measured_times],
        "elapsed_seconds": round(statistics.mean(measured_times), 3) if measured_times else None,
        "median_elapsed_seconds": (
            round(statistics.median(measured_times), 3) if measured_times else None
        ),
    }


def memory_recommendation(cases: list[dict[str, Any]]) -> tuple[str, float]:
    worst_reserved = max(case["max_memory_reserved_bytes"] for case in cases) / GIB
    if any(case["status"] != "ok" for case in cases) or worst_reserved >= 120:
        return "two_pass_fallback", worst_reserved
    if worst_reserved < 110:
        return "one_pass_ready", worst_reserved
    return "review_headroom", worst_reserved


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stress SFT-only/full-interaction GAC Phase-1 paths on two long samples with warm-up"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="artifacts/stage1/stress_memory.json")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=1)
    arguments = parser.parse_args()
    if arguments.warmup < 0 or arguments.repetitions <= 0:
        parser.error("--warmup must be nonnegative and --repetitions must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("An H200 CUDA GPU is required for the VRAM stress tests")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Run this stress command with exactly one visible H200")
    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(device)
    if "H200" not in gpu_name.upper():
        raise RuntimeError(f"Expected H200; visible GPU is {gpu_name}")
    config = load_config(arguments.config)
    _validate_stage1_config(config)
    if config["stage1"].get("forward_mode", "one_pass") != "one_pass":
        raise ValueError("Stress test requires stage1.forward_mode: one_pass")
    if int(config["stage1"]["micro_batch_size"]) != 1:
        raise ValueError("Stress test requires micro_batch_size: 1")
    if int(config["stage1"]["num_experts"]) != 3:
        raise ValueError("Stress test requires exactly three experts")
    seed_everything(int(config["seed"]), False)
    prepared = Path(config["paths"]["prepared"])
    manifest = read_json(prepared / "manifest.json")
    require_file_sha256(prepared, manifest, "data_file", "data_file_sha256")
    require_file_sha256(prepared, manifest, "config_file", "config_file_sha256")
    tokenizer = load_tokenizer(config["model"])
    if tokenizer_fingerprint(tokenizer) != manifest["tokenizer_fingerprint"]:
        raise RuntimeError("Stress test tokenizer does not match the prepared dataset")
    require_same_model_source(
        config["model"],
        manifest["config"]["model"],
        "Stage-1 memory stress/preprocessing",
    )
    dataset = JsonlRecordDataset(prepared)
    if not len(dataset):
        raise ValueError("Prepared dataset is empty")
    longest = max(
        (dataset[index] for index in range(len(dataset))),
        key=lambda item: len(item.input_ids),
    )
    synthetic = extend_reasoning(longest, TARGET_LENGTH)
    collator = LongCoTCollator(tokenizer.pad_token_id)
    model, names = create_multi_adapter_model(
        config["model"],
        config["lora"],
        int(config["stage1"]["num_experts"]),
        device,
        int(config["seed"]),
    )
    if TARGET_LENGTH > int(model.config.max_position_embeddings):
        raise ValueError("Synthetic sequence exceeds the model context limit")
    groups = adapter_parameter_groups(model, names)
    parameters = [list(group.values()) for group in groups]
    case_count = len(PHASES) * 2
    iterations_per_case = arguments.warmup + arguments.repetitions
    pairs = [
        _optimizer_and_scheduler(current, config, case_count * iterations_per_case)
        for current in parameters
    ]
    optimizers = [pair[0] for pair in pairs]
    schedulers = [pair[1] for pair in pairs]
    bandwidth = BandwidthEMA(
        decay=float(config["rbf"]["bandwidth_ema"]),
        floor=float(config["rbf"]["bandwidth_floor"]),
    )
    cases = []
    for phase in PHASES:
        for name, record in (("longest_real", longest), ("synthetic_32768", synthetic)):
            result = run_case(
                name,
                phase,
                record,
                collator,
                model,
                names,
                groups,
                parameters,
                optimizers,
                schedulers,
                bandwidth,
                config,
                device,
                warmup=arguments.warmup,
                repetitions=arguments.repetitions,
                first_iteration=len(cases) * iterations_per_case,
            )
            cases.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
            if result["status"] == "cuda_oom":
                break
        if cases[-1]["status"] == "cuda_oom":
            break
    recommendation, worst_reserved = memory_recommendation(cases)
    report = {
        "gpu_name": gpu_name,
        "gpu_total_memory_gib": round(
            torch.cuda.get_device_properties(device).total_memory / GIB, 3
        ),
        "forward_mode": "one_pass",
        "update_modes": list(PHASES),
        "gradient_checkpointing": bool(config["model"].get("gradient_checkpointing", False)),
        "lm_head_chunk_tokens": int(config["runtime"]["lm_head_chunk_tokens"]),
        "probe_hidden_device": config["runtime"].get("probe_hidden_device", "cpu"),
        "timing_unit": "one_microbatch_with_optimizer_step",
        "warmup_iterations_per_case": arguments.warmup,
        "measured_iterations_per_case": arguments.repetitions,
        "config_fingerprint": fingerprint(_run_compatible_config(config)),
        "prepared_manifest_fingerprint": fingerprint(manifest),
        "cases": cases,
        "cases_planned": case_count,
        "cases_completed": sum(case["status"] == "ok" for case in cases),
        "worst_reserved_gib": round(worst_reserved, 3),
        "recommendation": recommendation,
    }
    destination = Path(arguments.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps({"report": str(destination), "recommendation": recommendation}),
        flush=True,
    )
    if recommendation == "two_pass_fallback":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
