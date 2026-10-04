from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from ..config import load_config
from ..data.dataset import JsonlRecordDataset
from ..data.prepare import tokenizer_fingerprint
from ..data.schema import PreparedRecord, TokenRegion
from ..models.multi_adapter import load_tokenizer, require_same_model_source
from ..stage2.online import (
    Phase2RecordPlan,
    RecordGradientResult,
    compute_record_gradient,
    create_online_model,
    plan_record,
)
from ..stage2.trainer import (
    METHOD,
    OUTPUT_SPACE_METHOD,
    _stable_config,
    _validate_stage2_config,
    apply_accumulated_update,
)
from ..utils.distributed import DistributedContext
from ..utils.manifest import (
    fingerprint,
    read_json,
    require_file_sha256,
    runtime_metadata,
    write_json,
)
from ..utils.seed import seed_everything
from ..utils.training import add_gradients_, cosine_warmup_lambda, zeros_like_parameters

GIB = 1024**3


def make_synthetic_record(
    record: PreparedRecord, tokenizer: Any, max_length: int, method: str = METHOD
) -> tuple[PreparedRecord, Phase2RecordPlan]:
    """Extend the last retained step to the method's full context limit.

    Output-space uses the reasoning sequence limit; task-geometry reserves its
    complete gold-anchor suffix. The source answer is preserved in the record.
    The extended step is a memory stress only; its timing does not estimate
    typical training cost. Remove discarded trailing steps before extending.
    """
    planner = _record_planner(method)
    plan = planner(record, tokenizer, max_length)
    if plan.num_steps == 0:
        raise ValueError("Cannot extend a sample with no eligible reasoning steps")
    retained_end = len(plan.input_ids)
    if retained_end > record.answer_start:
        raise RuntimeError("Phase-2 reasoning input unexpectedly includes the answer block")
    source = replace(
        record,
        input_ids=record.input_ids[:retained_end] + record.input_ids[record.answer_start :],
        labels=record.labels[:retained_end] + record.labels[record.answer_start :],
        attention_mask=(
            record.attention_mask[:retained_end] + record.attention_mask[record.answer_start :]
        ),
        offset_mapping=(
            record.offset_mapping[:retained_end] + record.offset_mapping[record.answer_start :]
        ),
        region_ids=record.region_ids[:retained_end] + record.region_ids[record.answer_start :],
        step_ids=record.step_ids[:retained_end] + record.step_ids[record.answer_start :],
        answer_start=retained_end,
        kept_steps=plan.num_steps,
        truncated=record.truncated or plan.discarded_steps > 0,
    )
    extra = max_length - max(len(plan.input_ids), plan.max_anchor_length)
    if extra < 0:
        raise ValueError("Planned real sample exceeds the configured Phase-2 context limit")
    positions = plan.step_positions[-1]
    if not positions:
        raise RuntimeError("The final retained step has no KD content tokens")
    insertion = positions[-1] + 1
    step_id = source.step_ids[positions[-1]]
    repeated = [source.input_ids[positions[index % len(positions)]] for index in range(extra)]

    def insert(values: list[Any], additions: list[Any]) -> list[Any]:
        return values[:insertion] + additions + values[insertion:]

    synthetic = replace(
        source,
        sample_id=f"{record.sample_id}-synthetic-{'reasoning' if method == OUTPUT_SPACE_METHOD else 'anchor'}-{max_length}",
        input_ids=insert(source.input_ids, repeated),
        labels=insert(source.labels, repeated),
        attention_mask=insert(source.attention_mask, [1] * extra),
        offset_mapping=insert(source.offset_mapping, [(0, 0)] * extra),
        region_ids=insert(source.region_ids, [int(TokenRegion.REASONING)] * extra),
        step_ids=insert(source.step_ids, [step_id] * extra),
        answer_start=source.answer_start + extra,
        original_length=len(source.input_ids) + extra,
        kept_length=len(source.input_ids) + extra,
    )
    synthetic_plan = planner(synthetic, tokenizer, max_length)
    if synthetic_plan.num_steps != plan.num_steps or synthetic_plan.discarded_steps != 0:
        raise RuntimeError("Synthetic extension unexpectedly changed the retained reasoning steps")
    if max(len(synthetic_plan.input_ids), synthetic_plan.max_anchor_length) != max_length:
        raise RuntimeError("Synthetic extension did not reach the configured context limit")
    return synthetic, synthetic_plan


def longest_eligible_record(
    dataset: JsonlRecordDataset, tokenizer: Any, max_length: int, method: str = METHOD
) -> tuple[PreparedRecord, Phase2RecordPlan, dict[str, int]]:
    planner = _record_planner(method)
    selected: tuple[PreparedRecord, Phase2RecordPlan] | None = None
    selected_size = (-1, -1)
    counts = {
        "examples_scanned": len(dataset),
        "examples_without_eligible_steps": 0,
        "examples_with_discarded_steps": 0,
        "eligible_examples": 0,
    }
    for index in range(len(dataset)):
        record = dataset[index]
        plan = planner(record, tokenizer, max_length)
        counts["examples_with_discarded_steps"] += int(plan.discarded_steps > 0)
        if plan.num_steps == 0:
            counts["examples_without_eligible_steps"] += 1
            continue
        counts["eligible_examples"] += 1
        size = (max(len(plan.input_ids), plan.max_anchor_length), plan.num_steps)
        if size > selected_size:
            selected, selected_size = (record, plan), size
    if selected is None:
        raise ValueError("Prepared dataset has no Phase-2-ready reasoning samples")
    return selected[0], selected[1], counts


def _record_planner(method: str):
    if method == OUTPUT_SPACE_METHOD:
        from ..stage2.output_space import plan_record as output_plan

        return output_plan
    return plan_record


def _optimizer_and_scheduler(
    parameters: list[torch.nn.Parameter], config: dict[str, Any], total_steps: int
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(config["stage2"]["learning_rate"]),
        betas=tuple(float(value) for value in config["optimizer"]["betas"]),
        eps=float(config["optimizer"]["eps"]),
        weight_decay=float(config["optimizer"]["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: cosine_warmup_lambda(
            step,
            total_steps,
            float(config["scheduler"]["warmup_ratio"]),
            float(config["scheduler"].get("min_lr_ratio", 0.0)),
        ),
    )
    return optimizer, scheduler


def _run_microbatch(
    record: PreparedRecord,
    tokenizer: Any,
    model: torch.nn.Module,
    adapter_names: list[str],
    parameters: list[torch.nn.Parameter],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    """Run the full trainer gradient path and a temporary single-record update."""
    optimizer.zero_grad(set_to_none=True)
    # Training holds its FP32 accumulation buffer during the complete forward.
    # Allocate it before the forward so the measured peak includes that state.
    buffer = zeros_like_parameters(parameters)
    if config.get("method") == OUTPUT_SPACE_METHOD:
        from ..stage2.output_space import compute_record_gradient as record_gradient
    else:
        record_gradient = compute_record_gradient
    result = record_gradient(
        model,
        adapter_names,
        parameters,
        record,
        tokenizer,
        config,
        device,
        profile=True,
        **(
            {"cached_target": config["_stress_targets"][record.sample_id]}
            if config.get("method") == OUTPUT_SPACE_METHOD
            else {}
        ),
    )
    if not isinstance(result, RecordGradientResult):
        raise TypeError("The Phase-2 gradient path returned an unexpected result type")
    if len(result.gradients) != len(parameters):
        raise RuntimeError("Stress result does not cover every student LoRA parameter")
    for parameter, gradient in zip(parameters, result.gradients, strict=True):
        if parameter.shape != gradient.shape:
            raise RuntimeError("Stress gradient shape does not match its student LoRA parameter")
        if not torch.isfinite(gradient).all():
            raise FloatingPointError("Phase-2 stress produced a non-finite gradient")
    if not math.isfinite(result.loss):
        raise FloatingPointError("Phase-2 stress produced a non-finite loss")
    if not 0 <= result.active_steps <= result.steps:
        raise RuntimeError("Stress active-step count is invalid")
    timings = dict(result.timings)
    applied = result.active_steps > 0
    details = {
        "loss": result.loss,
        "steps": result.steps,
        "active_steps": result.active_steps,
        "discarded_steps": result.discarded_steps,
        "metrics": dict(result.metrics),
    }
    add_gradients_(buffer, result.gradients)
    del result
    gradient_norm = None
    if applied:
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        applied, gradient_norm = apply_accumulated_update(
            parameters,
            buffer,
            optimizer,
            scheduler,
            1,
            details["active_steps"],
            float(config["stage2"]["max_grad_norm"]),
        )
        if not math.isfinite(gradient_norm):
            raise FloatingPointError("Phase-2 stress gradient norm is non-finite")
        torch.cuda.synchronize(device)
        timings["optimizer_update_seconds"] = time.perf_counter() - start
    else:
        timings["optimizer_update_seconds"] = 0.0
    return {
        **details,
        "optimizer_update_applied": applied,
        "gradient_norm": gradient_norm,
        "component_seconds": timings,
    }


def run_case(
    name: str,
    record: PreparedRecord,
    plan: Phase2RecordPlan,
    tokenizer: Any,
    model: torch.nn.Module,
    adapter_names: list[str],
    parameters: list[torch.nn.Parameter],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    config: dict[str, Any],
    device: torch.device,
    *,
    warmup: int = 1,
    repetitions: int = 1,
) -> dict[str, Any]:
    if warmup < 0 or repetitions <= 0:
        raise ValueError("Stress warmup must be nonnegative and repetitions must be positive")
    if plan.num_steps == 0:
        raise ValueError("Stress case must contain eligible reasoning steps")
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    warmup_times: list[float] = []
    measured_times: list[float] = []
    warmup_components: list[dict[str, float]] = []
    measured_components: list[dict[str, float]] = []
    iteration_details: list[dict[str, Any]] = []
    status, error, failed_iteration_kind = "ok", None, None
    for index in range(warmup + repetitions):
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        try:
            details = _run_microbatch(
                record,
                tokenizer,
                model,
                adapter_names,
                parameters,
                optimizer,
                scheduler,
                config,
                device,
            )
            torch.cuda.synchronize(device)
        except (torch.cuda.OutOfMemoryError, FloatingPointError) as exception:
            status = (
                "cuda_oom" if isinstance(exception, torch.cuda.OutOfMemoryError) else "nonfinite"
            )
            error = str(exception)
            failed_iteration_kind = "warmup" if index < warmup else "measured"
            optimizer.zero_grad(set_to_none=True)
            break
        elapsed = time.perf_counter() - start
        iteration_details.append(details)
        if index < warmup:
            warmup_times.append(elapsed)
            warmup_components.append(details["component_seconds"])
        else:
            measured_times.append(elapsed)
            measured_components.append(details["component_seconds"])
    allocated = torch.cuda.max_memory_allocated(device)
    reserved = torch.cuda.max_memory_reserved(device)
    component_keys = sorted({key for item in measured_components for key in item})
    return {
        "case": name,
        "status": status,
        "error": error,
        "sample_id": record.sample_id,
        "reasoning_sequence_length": len(plan.input_ids),
        **(
            {"max_teacher_forced_sequence_length": len(plan.input_ids)}
            if config.get("method") == OUTPUT_SPACE_METHOD
            else {"max_gold_anchor_sequence_length": plan.max_anchor_length}
        ),
        "planned_steps": plan.num_steps,
        "all_retained_steps_evaluated": all(
            item["steps"] == plan.num_steps for item in iteration_details
        )
        and len(iteration_details) == warmup + repetitions,
        "micro_batch_size": 1,
        "training_global_batch_size": int(config["stage2"]["global_batch_size"]),
        "timing_unit": "one_microbatch_with_temporary_optimizer_step",
        "real_sample": name == "longest_real",
        "timing_representative_of_training": False,
        "timing_interpretation": (
            "observed longest-real microbatch only; not a training-run or global-batch estimate"
            if name == "longest_real"
            else "synthetic single-step extension is a memory stress only"
        ),
        "warmup_iterations": warmup,
        "measured_iterations": repetitions,
        "completed_warmup_iterations": len(warmup_times),
        "completed_measured_iterations": len(measured_times),
        "optimizer_updates": sum(item["optimizer_update_applied"] for item in iteration_details),
        "measured_optimizer_updates": sum(
            item["optimizer_update_applied"] for item in iteration_details[warmup:]
        ),
        "skipped_zero_signal_iterations": sum(
            not item["optimizer_update_applied"] for item in iteration_details
        ),
        "failed_iteration_kind": failed_iteration_kind,
        "memory_includes_warmup": True,
        "max_memory_allocated_bytes": allocated,
        "max_memory_reserved_bytes": reserved,
        "max_memory_allocated_gib": round(allocated / GIB, 3),
        "max_memory_reserved_gib": round(reserved / GIB, 3),
        "warmup_elapsed_seconds": warmup_times,
        "measured_elapsed_seconds": measured_times,
        "mean_measured_elapsed_seconds": statistics.mean(measured_times)
        if measured_times
        else None,
        "median_measured_elapsed_seconds": (
            statistics.median(measured_times) if measured_times else None
        ),
        "warmup_component_seconds": warmup_components,
        "measured_component_seconds": measured_components,
        "mean_measured_component_seconds": {
            key: statistics.mean(item.get(key, 0.0) for item in measured_components)
            for key in component_keys
        },
        "iterations": iteration_details,
    }


def memory_recommendation(
    cases: list[dict[str, Any]], total_memory_bytes: int, min_headroom_gib: float
) -> tuple[str, float | None]:
    if not cases:
        return "not_measured", None
    worst_reserved = max(case["max_memory_reserved_bytes"] for case in cases)
    headroom = (total_memory_bytes - worst_reserved) / GIB
    if any(case["status"] != "ok" for case in cases):
        return "direct_teacher_not_ready", headroom
    if len(cases) != 2 or any(not case["all_retained_steps_evaluated"] for case in cases):
        return "incomplete_measurement", headroom
    if any(case["measured_optimizer_updates"] == 0 for case in cases):
        return "inconclusive_zero_signal", headroom
    if headroom < min_headroom_gib:
        return "review_headroom", headroom
    return "direct_teacher_ready", headroom


def _h200_requirement() -> tuple[torch.device | None, str | None, str | None]:
    if not torch.cuda.is_available():
        return None, None, "An H200 CUDA GPU is required; no CUDA GPU is available"
    if torch.cuda.device_count() != 1:
        return None, None, "Run this command with exactly one visible H200 GPU"
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        return None, None, "Run the stress command in one process without a distributed launcher"
    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(device)
    if "H200" not in gpu_name.upper():
        return None, gpu_name, f"Expected H200; visible GPU is {gpu_name}"
    return device, gpu_name, None


def _source_manifests(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    from ..stage2.teachers import ensure_stage2_teachers

    manifests = {}
    for key in ("prepared", "stage1"):
        source = Path(config["paths"][key])
        manifests[key] = (
            ensure_stage2_teachers(config)
            if key == "stage1" and config.get("teacher_source")
            else read_json(source if source.is_file() else source / "manifest.json")
        )
    from ..stage2.council_cache import load_council_cache

    manifests["council_cache"] = load_council_cache(
        config, manifests["prepared"], manifests["stage1"]
    ).manifest
    return manifests


def _record_provenance(
    config: dict[str, Any], report: dict[str, Any], manifests: dict[str, dict[str, Any]]
) -> None:
    file_checksums = {}
    for key, pairs in (
        ("prepared", [("data_file", "data_file_sha256"), ("config_file", "config_file_sha256")]),
        (
            "stage1",
            [("adapter_bundle", "adapter_bundle_sha256"), ("config_file", "config_file_sha256")],
        ),
    ):
        root, manifest = Path(config["paths"][key]), manifests[key]
        for file_key, hash_key in pairs:
            path = require_file_sha256(root, manifest, file_key, hash_key)
            file_checksums[f"{key}.{file_key}"] = {
                "path": str(path),
                "sha256": manifest[hash_key],
                "verified": True,
            }
    council = manifests["council_cache"]
    if council.get("artifact") != "stage2_council_cache":
        raise RuntimeError("Phase-2 stress requires the council cache")
    if council["prepared_manifest_fingerprint"] != fingerprint(manifests["prepared"]):
        raise RuntimeError("Council cache/prepared mismatch")
    if council["stage1_manifest_fingerprint"] != fingerprint(manifests["stage1"]):
        raise RuntimeError("Council cache/teachers mismatch")
    root = (
        Path(config["paths"]["teacher_cache_dir"]) / council["fingerprint"]
        if "teacher_cache_dir" in config["paths"]
        else Path(council["cache_directory"])
    )
    for file_key, hash_key in [
        ("index_file", "index_file_sha256"),
        ("best_expert_file", "best_expert_file_sha256"),
    ]:
        path = require_file_sha256(root, council, file_key, hash_key)
        file_checksums[f"council_cache.{file_key}"] = {
            "path": str(path),
            "sha256": council[hash_key],
            "verified": True,
        }
    source_hashes = {key: fingerprint(value) for key, value in manifests.items()}
    report["source_manifest_fingerprints"] = source_hashes
    report["verified_source_files"] = file_checksums
    report["source_provenance_status"] = "verified"
    # Exactly the same run identity as the one-device training command.
    report["run_fingerprint"] = fingerprint(
        {
            "method": config.get("method", METHOD),
            "config_fingerprint": report["config_fingerprint"],
            "prepared_manifest_fingerprint": source_hashes["prepared"],
            "stage1_manifest_fingerprint": source_hashes["stage1"],
            "council_cache_fingerprint": council["fingerprint"],
            "world_size": 1,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure configured Phase-2 updates on one H200 with all LoRA parameters and steps"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="artifacts/stage2/output_space_stress_memory.json")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--min-headroom-gib", type=float, default=12.0)
    arguments = parser.parse_args()
    if arguments.warmup < 0 or arguments.repetitions <= 0:
        parser.error("--warmup must be nonnegative and --repetitions must be positive")
    if not math.isfinite(arguments.min_headroom_gib) or arguments.min_headroom_gib < 0:
        parser.error("--min-headroom-gib must be finite and nonnegative")
    config = load_config(arguments.config)
    _validate_stage2_config(config)
    if int(config["stage2"]["micro_batch_size"]) != 1:
        raise ValueError("Phase-2 stress requires stage2.micro_batch_size: 1")
    stable_config = _stable_config(config)
    method = config.get("method", METHOD)
    output_space = method == OUTPUT_SPACE_METHOD
    section = "aggregation" if output_space else "geometry"
    report: dict[str, Any] = {
        "status": "not_measured",
        "recommendation": "not_measured",
        "config_fingerprint": fingerprint(stable_config),
        "config_fingerprint_definition": "sha256 of resolved Stage-2 config excluding private and resume fields",
        "config": stable_config,
        "runtime": runtime_metadata(config.get("_project_root", Path.cwd())),
        "cases_planned": 2,
        "cases_completed": 0,
        "cases": [],
        "warmup_iterations_per_case": arguments.warmup,
        "measured_iterations_per_case": arguments.repetitions,
        "required_headroom_gib": arguments.min_headroom_gib,
        "method": method,
        "execution": "cached_support_tail_student_only"
        if output_space
        else "legacy_online_task_geometry",
        "gold_answer_source": None if output_space else "entire_solution_field",
        "trained_artifact_written": False,
        "temporary_optimizer_updates_only": True,
        "timing_unit": "one_microbatch_with_temporary_optimizer_step",
        "global_batch_step_time_measured": False,
        "synthetic_case_timing_representative_of_training": False,
        "configured_limits": {
            "max_length": int(config["stage2"]["max_length"]),
            "lm_head_chunk_tokens": int(config["runtime"]["lm_head_chunk_tokens"]),
            "micro_batch_size": 1,
            "global_batch_size": int(config["stage2"]["global_batch_size"]),
            "gradient_checkpointing": bool(config["model"].get("gradient_checkpointing", False)),
            "lora_dropout": float(config["lora"]["dropout"]),
            **(
                {
                    "js_temperature": float(config["aggregation"]["js_temperature"]),
                    "kd_temperature": float(config["aggregation"]["kd_temperature"]),
                    "sft_weight": float(config["aggregation"]["sft_weight"]),
                }
                if output_space
                else {"temperature": float(config[section]["temperature"])}
            ),
            **(
                {}
                if output_space
                else {
                    "epsilon_a": float(config["geometry"]["epsilon_a"]),
                    "epsilon_u": float(config["geometry"]["epsilon_u"]),
                }
            ),
            "reasoning_step_cap": None,
            "lora_parameter_sampling": False,
        },
    }
    destination = Path(arguments.output)
    device, gpu_name, requirement = _h200_requirement()
    report["gpu_name"] = gpu_name
    if requirement:
        # Provenance is still useful on a laptop when the input artifacts are
        # present. Absent artifacts must not turn a missing GPU into invented
        # hardware or an apparently successful preflight.
        try:
            manifests = _source_manifests(config)
        except (FileNotFoundError, KeyError) as exception:
            report["source_provenance_status"] = "unavailable"
            report["source_provenance_error"] = str(exception)
        else:
            try:
                _record_provenance(config, report, manifests)
            except (OSError, KeyError, RuntimeError, ValueError) as exception:
                report["source_provenance_status"] = "invalid"
                report["source_provenance_error"] = str(exception)
        report.update(status="requirement_not_met", requirement=requirement)
        write_json(destination, report)
        print(json.dumps({"report": str(destination), "requirement": requirement}), flush=True)
        raise SystemExit(2)
    assert device is not None
    torch.cuda.set_device(device)
    total_memory = torch.cuda.get_device_properties(device).total_memory
    report["gpu_total_memory_bytes"] = total_memory
    report["gpu_total_memory_gib"] = total_memory / GIB
    seed_everything(int(config["seed"]), False)
    try:
        manifests = _source_manifests(config)
        _record_provenance(config, report, manifests)
        prepared = Path(config["paths"]["prepared"])
        tokenizer = load_tokenizer(config["model"])
        if tokenizer_fingerprint(tokenizer) != manifests["prepared"]["tokenizer_fingerprint"]:
            raise RuntimeError("Phase-2 stress tokenizer differs from the prepared dataset")
        require_same_model_source(
            config["model"],
            manifests["prepared"]["config"]["model"],
            "Phase-2 stress/preprocessing",
        )
        dataset = JsonlRecordDataset(prepared / manifests["prepared"]["data_file"])
        if len(dataset) != int(manifests["prepared"]["records"]) or not len(dataset):
            raise RuntimeError("Prepared dataset record count is invalid")
        max_length = int(config["stage2"]["max_length"])
        longest, longest_plan, counts = longest_eligible_record(
            dataset, tokenizer, max_length, method=method
        )
        synthetic, synthetic_plan = make_synthetic_record(
            longest, tokenizer, max_length, method=method
        )
        report["dataset_scan"] = counts
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        load_start = time.perf_counter()
        distributed = DistributedContext(0, 0, 1, device)
        if output_space:
            from ..stage2.council_cache import compile_record, load_council_cache
            from ..stage2.initialization import create_cached_student
            from ..models.multi_adapter import (
                create_multi_adapter_model,
                load_adapter_bundle,
                load_adapter_state,
            )

            cache = load_council_cache(config, manifests["prepared"], manifests["stage1"])
            # Synthetic trajectories cannot reuse the real record's target.
            # Precompute them separately, before any measured student iteration.
            prep_started = time.perf_counter()
            teachers, names = create_multi_adapter_model(
                config["model"], config["lora"], 3, device, int(config["seed"])
            )
            bundle_path = Path(config["paths"]["stage1"]) / manifests["stage1"]["adapter_bundle"]
            bundle = load_adapter_bundle(bundle_path)
            for name in names:
                load_adapter_state(teachers, name, bundle[name])
            synthetic_target, _, _ = compile_record(teachers, names, synthetic, config, device)
            del teachers, bundle
            torch.cuda.empty_cache()
            report["synthetic_preprocessing_seconds_excluded_from_training"] = (
                time.perf_counter() - prep_started
            )
            config["_stress_targets"] = {
                longest.sample_id: cache.get(longest.sample_id),
                synthetic.sample_id: synthetic_target,
            }
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            load_start = time.perf_counter()
            model, adapter_names, parameters = create_cached_student(config, distributed, cache)
        else:
            model, adapter_names, parameters = create_online_model(config, distributed)
        torch.cuda.synchronize(device)
        report["model_load_seconds"] = time.perf_counter() - load_start
        report["model_load_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        report["model_load_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
        report["council_experts"] = adapter_names
        report["teachers_loaded_during_student_measurement"] = not output_space
        report["student_lora_parameter_count"] = sum(parameter.numel() for parameter in parameters)
        if max_length > int(model.config.max_position_embeddings):
            raise ValueError("Configured Phase-2 context exceeds the model context limit")
        total_training_steps = math.ceil(
            int(config["stage2"]["epochs"])
            * len(dataset)
            / int(config["stage2"]["global_batch_size"])
        )
        optimizer, scheduler = _optimizer_and_scheduler(parameters, config, total_training_steps)
        report["scheduler_training_steps"] = total_training_steps
        for name, record, plan in (
            ("longest_real", longest, longest_plan),
            (
                f"synthetic_{'reasoning' if output_space else 'anchor'}_{max_length}",
                synthetic,
                synthetic_plan,
            ),
        ):
            result = run_case(
                name,
                record,
                plan,
                tokenizer,
                model,
                adapter_names,
                parameters,
                optimizer,
                scheduler,
                config,
                device,
                warmup=arguments.warmup,
                repetitions=arguments.repetitions,
            )
            report["cases"].append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
            if result["status"] != "ok":
                break
        recommendation, headroom = memory_recommendation(
            report["cases"], total_memory, arguments.min_headroom_gib
        )
        report.update(
            status="measured",
            recommendation=recommendation,
            worst_case_reserved_gib=max(
                case["max_memory_reserved_bytes"] for case in report["cases"]
            )
            / GIB,
            remaining_headroom_gib=headroom,
            cases_completed=sum(case["status"] == "ok" for case in report["cases"]),
        )
    except torch.cuda.OutOfMemoryError as exception:
        report.update(
            status="cuda_oom", recommendation="direct_teacher_not_ready", error=str(exception)
        )
        report["failure_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        report["failure_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
    except Exception as exception:
        report.update(status="error", recommendation="not_measured", error=str(exception))
        write_json(destination, report)
        raise
    write_json(destination, report)
    print(
        json.dumps({"report": str(destination), "recommendation": report["recommendation"]}),
        flush=True,
    )
    if report["recommendation"] != "direct_teacher_ready":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
