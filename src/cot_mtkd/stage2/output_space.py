from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import torch

from ..data.schema import PreparedRecord, TokenRegion
from ..data.token_spans import validate_token_contract
from ..models.chunked_head import decoder_and_lm_head, forward_hidden
from ..utils.training import zeros_like_parameters
from .disagreement import saturation_rho
from .online import Phase2RecordPlan, RecordGradientResult, _select_adapter
from .output_space_losses import cached_kd_sft_hidden_gradient


@dataclass
class OutputSpaceGradientResult(RecordGradientResult):
    step_metrics: list[dict[str, Any]] = field(default_factory=list)


def plan_record(record: PreparedRecord, tokenizer: Any, max_length: int) -> Phase2RecordPlan:
    """Retain complete reasoning steps without reserving an answer-anchor suffix.

    Prepared token/step boundaries are reused unchanged. Only REASONING content
    is supervised; delimiters and control tokens remain in the shared prefix.
    The tokenizer argument preserves the Phase-2 runner interface and is unused.
    """
    if max_length <= 0:
        raise ValueError("stage2.max_length must be positive")
    validate_token_contract(record.input_ids, record.labels, record.region_ids, record.step_ids)
    if record.reasoning_start > max_length:
        raise ValueError(f"{record.sample_id}: prompt/control exceeds stage2.max_length")
    step_ids = sorted({step for step in record.step_ids if step >= 0})
    positions, prefix_ends = [], []
    for step in step_ids:
        all_positions = [i for i, value in enumerate(record.step_ids) if value == step]
        end = max(all_positions) + 1
        if end > max_length:
            break
        content = [i for i in all_positions if record.region_ids[i] == int(TokenRegion.REASONING)]
        if not content or min(content) < 1:
            raise ValueError(f"{record.sample_id}: invalid reasoning content for step {step}")
        positions.append(content)
        prefix_ends.append(end)
    end = prefix_ends[-1] if prefix_ends else record.reasoning_start
    # Empty suffixes let the existing stress-plan interface describe exactly
    # the reasoning sequence length, with no dependence on the gold solution.
    return Phase2RecordPlan(
        input_ids=record.input_ids[:end],
        step_positions=positions,
        prefix_ends=prefix_ends,
        answer_prefix=[],
        solution=[],
        discarded_steps=len(step_ids) - len(positions),
    )


def unpack_cached_target(
    value: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad a single cached sample in RAM; persisted arrays contain no padding."""
    offsets = value["support_offsets"].long()
    sizes = offsets.diff()
    count = len(sizes)
    width = int(sizes.max()) if count else 0
    ids = torch.zeros((count, width), dtype=torch.long)
    mask = torch.arange(width)[None, :] < sizes[:, None]
    logq = torch.full((count, width + 1), -1e30)
    for index, (start, end) in enumerate(zip(offsets[:-1], offsets[1:], strict=True)):
        support = value["support_ids"][start:end].long()
        ids[index].fill_(int(support[0]))
        ids[index, : len(support)] = support
        logq[index, : len(support)] = value["support_log_target"][start:end]
    logq[:, -1] = value["tail_log_target"]
    return ids, mask, logq


def compute_record_gradient(
    model: torch.nn.Module,
    adapter_names: list[str],
    parameters: list[torch.nn.Parameter],
    record: PreparedRecord,
    tokenizer: Any,
    config: dict[str, Any],
    device: torch.device,
    profile: bool = False,
    *,
    cached_target: dict[str, torch.Tensor] | None = None,
) -> OutputSpaceGradientResult:
    """Student-only training from validated static support+tail supervision."""
    if cached_target is None:
        raise RuntimeError("Output-space training requires a council cache; run stage2-cache")
    required = (
        "tau",
        "tau_quantile",
        "disagreement_pooling_power",
        "temperature",
        "step_disagreement",
        "token_js_mean",
        "step_rho",
    )
    if any(key not in cached_target for key in required):
        raise RuntimeError("Missing fitted tau/calibrated target; rebuild stage2-cache")
    aggregation = config["aggregation"]
    for key in ("tau_quantile", "disagreement_pooling_power", "temperature"):
        if float(cached_target[key]) != float(aggregation[key]):
            raise RuntimeError(f"Cached {key} disagrees with Phase-2 config; rebuild stage2-cache")
    tau = float(cached_target["tau"])
    if not torch.equal(
        cached_target["step_rho"], saturation_rho(cached_target["step_disagreement"], tau)
    ):
        raise RuntimeError("Cached calibrated rho is inconsistent; rebuild stage2-cache")
    plan = plan_record(record, tokenizer, int(config["stage2"]["max_length"]))
    timers: dict[str, float] = {}
    if not plan.num_steps:
        return OutputSpaceGradientResult(
            zeros_like_parameters(parameters),
            0.0,
            0,
            0,
            plan.discarded_steps,
            {
                "prefix_tokens": len(plan.input_ids),
                "reasoning_tokens": 0,
                "head_chunks": 0,
                "cached_steps": 0,
                "recomputed_steps": 0,
                "teacher_head_chunk_sweeps": 0,
            },
            timers,
        )
    positions = [position for step in plan.step_positions for position in step]
    expected_offsets = [0]
    for step in plan.step_positions:
        expected_offsets.append(expected_offsets[-1] + len(step))
    if (
        cached_target["token_positions"].tolist() != positions
        or cached_target["step_offsets"].tolist() != expected_offsets
    ):
        raise RuntimeError(f"{record.sample_id}: cache token/step mapping mismatch")
    support_ids, mask, logq = unpack_cached_target(cached_target)
    targets = torch.tensor([record.input_ids[p] for p in positions])
    if not bool(((support_ids == targets[:, None]) & mask).any(-1).all()):
        raise RuntimeError("Cached support is missing an observed target")
    weights = torch.cat(
        [torch.full((len(step),), 1 / (len(step) * plan.num_steps)) for step in plan.step_positions]
    )

    def tick() -> float:
        if profile and device.type == "cuda":
            torch.cuda.synchronize(device)
        return time.perf_counter()

    started = tick()
    _select_adapter(model, "student", training=True)
    ids = torch.tensor([plan.input_ids], device=device, dtype=torch.long)
    output = forward_hidden(model, ids, torch.ones_like(ids), use_cache=False)
    hidden = output.last_hidden_state[0].index_select(0, torch.tensor(positions, device=device) - 1)
    del output
    timers["student_forward"] = tick() - started
    _, head = decoder_and_lm_head(model)
    aggregation = config["aggregation"]
    chunk = int(config["runtime"]["lm_head_chunk_tokens"])
    anomalies: dict[str, int] = {}
    started = tick()
    cotangent, tokens = cached_kd_sft_hidden_gradient(
        hidden,
        head,
        targets,
        support_ids,
        mask,
        logq,
        weights,
        float(aggregation["temperature"]),
        float(aggregation["sft_weight"]),
        chunk,
        anomalies,
    )
    timers["cached_target_and_hidden_gradient"] = tick() - started
    started = tick()
    gradients = torch.autograd.grad(hidden, parameters, grad_outputs=cotangent, allow_unused=False)
    gradients = [g.detach().float() for g in gradients]
    timers["final_kd_sft_gradient"] = tick() - started
    if not all(bool(torch.isfinite(g).all()) for g in gradients):
        raise FloatingPointError(f"Nonfinite cached KD+SFT gradient: {record.sample_id}")
    kd = float((tokens["kd"] * weights).sum())
    sft = float((tokens["sft"] * weights).sum())
    weighted = float(aggregation["sft_weight"]) * sft
    step_metrics = []
    for index, step in enumerate(plan.step_positions):
        start, end = expected_offsets[index : index + 2]
        step_kd = float(tokens["kd"][start:end].mean())
        step_sft = float(tokens["sft"][start:end].mean())
        step_metrics.append(
            {
                "step_index": index,
                "step_id": record.step_ids[step[0]],
                "n_tokens": len(step),
                "token_start": step[0],
                "token_end": step[-1] + 1,
                "token_js_mean": float(cached_target["token_js_mean"][index]),
                "js_units": "nats",
                "step_disagreement": float(cached_target["step_disagreement"][index]),
                "disagreement_pooling_power": float(aggregation["disagreement_pooling_power"]),
                "tau": tau,
                "tau_quantile": float(aggregation["tau_quantile"]),
                "rho": float(cached_target["step_rho"][index]),
                "temperature": float(aggregation["temperature"]),
                "js_temperature": float(aggregation["temperature"]),
                "kd_temperature": float(aggregation["temperature"]),
                "teacher_count": len(adapter_names),
                "step_kd_loss": step_kd,
                "step_sft_loss": step_sft,
                "weighted_sft_loss": float(aggregation["sft_weight"]) * step_sft,
                "step_total_loss": step_kd + float(aggregation["sft_weight"]) * step_sft,
                "target_tail_mass": float(logq[start:end, -1].exp().mean()),
                "target_support_mass": float((1 - logq[start:end, -1].exp()).mean()),
                "student_tail_mass": float(tokens["student_tail"][start:end].mean()),
                "student_support_mass": float((1 - tokens["student_tail"][start:end]).mean()),
                "target_entropy": float(tokens["entropy"][start:end].mean()),
                "support_size_mean": float(mask[start:end].sum(-1).float().mean()),
            }
        )
    return OutputSpaceGradientResult(
        gradients,
        kd + weighted,
        plan.num_steps,
        plan.num_steps,
        plan.discarded_steps,
        {
            "disagreement_sum": float(cached_target["step_disagreement"].sum()),
            "token_js_sum": float(cached_target["token_js_mean"].sum()),
            "rho_sum": float(cached_target["step_rho"].sum()),
            "kd_loss": kd,
            "sft_loss": sft,
            "weighted_sft_loss": weighted,
            "student_tail_mass": float((tokens["student_tail"] * weights).sum()),
            "student_support_mass": float(((1 - tokens["student_tail"]) * weights).sum()),
            "target_tail_mass": float((logq[:, -1].exp() * weights).sum()),
            "target_support_mass": float(((1 - logq[:, -1].exp()) * weights).sum()),
            "target_entropy": float((tokens["entropy"] * weights).sum()),
            "prefix_tokens": len(plan.input_ids),
            "reasoning_tokens": len(positions),
            "head_chunks": math.ceil(len(positions) / chunk),
            "cached_steps": plan.num_steps,
            "recomputed_steps": 0,
            "teacher_head_chunk_sweeps": 0,
            **anomalies,
        },
        timers,
        step_metrics,
    )
