from __future__ import annotations

import argparse
import json
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
from ..stage1.rbf import BandwidthEMA, effective_update_distances, repulsion_updates
from ..stage1.trainer import (
    _batch_to_device,
    _optimizer_and_scheduler,
    _run_compatible_config,
    one_pass_expert_gradients,
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


def run_case(
    name: str,
    record: PreparedRecord,
    collator: LongCoTCollator,
    model: torch.nn.Module,
    names: list[str],
    groups: list[dict[str, torch.nn.Parameter]],
    parameters: list[list[torch.nn.Parameter]],
    optimizers: list[torch.optim.Optimizer],
    schedulers: list[torch.optim.lr_scheduler.LRScheduler],
    sft_buffers: list[list[torch.Tensor]],
    dpp_buffers: list[list[torch.Tensor]],
    bandwidth: BandwidthEMA,
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    for group in (*sft_buffers, *dpp_buffers):
        for value in group:
            value.zero_()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    batch = _batch_to_device(collator([record]), device)
    views = shifted_token_views(batch)
    sft_tokens = int(views["response_targets"].numel())
    dpp_samples = int(torch.unique(views["reasoning_batch_indices"]).numel())
    probe, results = one_pass_expert_gradients(
        model,
        names,
        parameters,
        batch,
        0,
        0,
        int(config["seed"]),
        config,
        device,
    )
    if probe.dpp_sample_count != dpp_samples:
        raise RuntimeError(
            "Stress DPP sample count differs from its planned denominator"
        )
    for expert, (sft, dpp, _, _) in enumerate(results):
        add_gradients_(sft_buffers[expert], sft)
        add_gradients_(dpp_buffers[expert], dpp)
    for values in sft_buffers:
        for value in values:
            value.div_(max(sft_tokens, 1))
    for values in dpp_buffers:
        for value in values:
            value.div_(max(dpp_samples, 1))
    set_all_adapters_trainable(model, names)
    scaling = float(config["lora"]["alpha"]) / float(config["lora"]["rank"])
    distances_for_step = effective_update_distances(groups, scaling)
    current_bandwidth = bandwidth.update(distances_for_step)
    repulsion, kernel, _ = repulsion_updates(
        groups, scaling, current_bandwidth, distances=distances_for_step
    )
    final, _ = stable_gac_gradients(
        sft_buffers,
        dpp_buffers,
        repulsion,
        kernel,
        0.5,  # The ramp keeps both transformer VJPs and their gradient buffers.
        float(config["stage1"]["dpp_weight"]),
        float(config["stage1"]["rbf_weight"]),
    )
    for values, current, optimizer, scheduler in zip(
        final, parameters, optimizers, schedulers, strict=True
    ):
        global_clip_grad_list_(values, float(config["stage1"]["max_grad_norm"]))
        assign_gradients(current, values)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize(device)
    allocated = torch.cuda.max_memory_allocated(device)
    reserved = torch.cuda.max_memory_reserved(device)
    return {
        "case": name,
        "status": "ok",
        "interaction_phase": "ramp",
        "sample_id": record.sample_id,
        "sequence_length": len(record.input_ids),
        "dpp_samples": probe.dpp_sample_count,
        "max_memory_allocated_bytes": allocated,
        "max_memory_reserved_bytes": reserved,
        "max_memory_allocated_gib": round(allocated / GIB, 3),
        "max_memory_reserved_gib": round(reserved / GIB, 3),
        "elapsed_seconds": round(time.perf_counter() - start, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run exactly two one-pass Phase-1 H200 memory stress cases"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="artifacts/stage1/stress_memory.json")
    arguments = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("An H200 CUDA GPU is required for the VRAM stress tests")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Run this stress command with exactly one visible H200")
    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(device)
    if "H200" not in gpu_name.upper():
        raise RuntimeError(f"Expected H200; visible GPU is {gpu_name}")
    config = load_config(arguments.config)
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
    pairs = [_optimizer_and_scheduler(current, config, 2) for current in parameters]
    optimizers = [pair[0] for pair in pairs]
    schedulers = [pair[1] for pair in pairs]
    sft_buffers = [zeros_like_parameters(current) for current in parameters]
    dpp_buffers = [zeros_like_parameters(current) for current in parameters]
    bandwidth = BandwidthEMA(
        decay=float(config["rbf"]["bandwidth_ema"]),
        floor=float(config["rbf"]["bandwidth_floor"]),
    )
    cases = []
    for name, record in (("longest_real", longest), ("synthetic_32768", synthetic)):
        try:
            result = run_case(
                name,
                record,
                collator,
                model,
                names,
                groups,
                parameters,
                optimizers,
                schedulers,
                sft_buffers,
                dpp_buffers,
                bandwidth,
                config,
                device,
            )
        except torch.cuda.OutOfMemoryError:
            allocated = torch.cuda.max_memory_allocated(device)
            reserved = torch.cuda.max_memory_reserved(device)
            result = {
                "case": name,
                "status": "cuda_oom",
                "sample_id": record.sample_id,
                "sequence_length": len(record.input_ids),
                "max_memory_allocated_bytes": allocated,
                "max_memory_reserved_bytes": reserved,
                "max_memory_allocated_gib": round(allocated / GIB, 3),
                "max_memory_reserved_gib": round(reserved / GIB, 3),
            }
        cases.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        if result["status"] == "cuda_oom":
            break
    worst_reserved = max(case["max_memory_reserved_bytes"] for case in cases) / GIB
    if any(case["status"] == "cuda_oom" for case in cases) or worst_reserved >= 120:
        recommendation = "two_pass_fallback"
    elif worst_reserved < 110:
        recommendation = "one_pass_ready"
    else:
        recommendation = "review_headroom"
    report = {
        "gpu_name": gpu_name,
        "gpu_total_memory_gib": round(
            torch.cuda.get_device_properties(device).total_memory / GIB, 3
        ),
        "forward_mode": "one_pass",
        "interaction_phase": "ramp",
        "config_fingerprint": fingerprint(_run_compatible_config(config)),
        "prepared_manifest_fingerprint": fingerprint(manifest),
        "cases": cases,
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
