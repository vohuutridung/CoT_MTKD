from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass, field
from typing import Any

import torch

from ..data.schema import PreparedRecord, TokenRegion
from ..data.token_spans import validate_token_contract
from ..models.chunked_head import decoder_and_lm_head, forward_hidden
from ..utils.training import zeros_like_parameters
from .online import Phase2RecordPlan, RecordGradientResult, _select_adapter
from .output_space_losses import cached_kd_sft_hidden_gradient, power_mean_log_target


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
    """Pad a single cached sample in RAM; persisted arrays contain no padding.

    Returns support ids [T, W], mask [T, W] and every expert's reduced log
    distribution [M, T, W + 1] (last category = tail), padding at -1e30.
    """
    offsets = value["support_offsets"].long()
    sizes = offsets.diff()
    count = len(sizes)
    width = int(sizes.max()) if count else 0
    experts = value["expert_support_log_probs"]
    support = value["support_ids"].long()
    mask = torch.arange(width)[None, :] < sizes[:, None]
    # Row-major mask order equals the ragged order. Padding repeats a valid id.
    ids = support[offsets[:-1]][:, None].repeat(1, width) if count else support.new_zeros((0, 0))
    ids[mask] = support
    logp = torch.full((len(experts), count, width + 1), -1e30, dtype=torch.float64)
    logp[:, :, :-1][:, mask] = experts.double()
    logp[:, :, -1] = value["expert_tail_log_probs"]
    return ids, mask, logp


def map_step_rho(
    step_js: torch.Tensor,
    aggregation: dict[str, Any],
    experts: int,
    js_reference: torch.Tensor | None,
) -> torch.Tensor:
    """Per-step power-mean parameter from the step's teacher JS (nats).

    ecdf: the step's rank among all corpus steps, uniform on (0, 1] whatever
    the scale of raw JS. linear: JS / log M (normalized JS). constant: fixed
    rho for the geometric (0) and arithmetic (1) controls.
    """
    mapping = aggregation.get("rho_mapping", "ecdf")
    js = step_js.double()
    if mapping == "constant":
        return torch.full_like(js, float(aggregation["rho_constant"]))
    if mapping == "linear":
        return (js / math.log(experts)).clamp(0.0, 1.0)
    if js_reference is None or not len(js_reference):
        raise RuntimeError("rho_mapping: ecdf requires the cache JS reference")
    ranks = torch.searchsorted(js_reference.double().contiguous(), js.contiguous(), right=True)
    return (ranks.double() / len(js_reference)).clamp(0.0, 1.0)


def council_log_target(
    expert_log_probs: torch.Tensor,
    step_offsets: list[int],
    step_rho: torch.Tensor,
    teacher_indices: list[int],
) -> torch.Tensor:
    """Step-wise power-mean target over the selected teachers, [T, W + 1].

    One teacher reduces to its own distribution (single-expert control).
    """
    selected = expert_log_probs[teacher_indices]
    if len(teacher_indices) == 1:
        return selected[0] - torch.logsumexp(selected[0], -1, keepdim=True)
    target = torch.empty_like(selected[0])
    for index, (start, end) in enumerate(itertools.pairwise(step_offsets)):
        target[start:end] = power_mean_log_target(selected[:, start:end], float(step_rho[index]))
    return target


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
    aggregation = config["aggregation"]
    council = config.get("_council", {})
    support_ids, mask, expert_logp = unpack_cached_target(cached_target)
    experts = len(expert_logp)
    if experts != len(adapter_names):
        raise RuntimeError(f"{record.sample_id}: cache/council expert count mismatch")
    teacher_indices = council.get("teacher_indices", list(range(experts)))
    step_js = cached_target["step_js"].double()
    step_rho = map_step_rho(step_js, aggregation, experts, council.get("js_reference"))
    logq = council_log_target(expert_logp, expected_offsets, step_rho, teacher_indices).float()
    del expert_logp
    targets = torch.tensor([record.input_ids[p] for p in positions])
    if not bool(((support_ids == targets[:, None]) & mask).any(-1).all()):
        raise RuntimeError("Cached support is missing an observed target")
    if aggregation.get("loss_normalization", "token") == "token":
        # Every supervised token counts once: short steps ("Wait,") are not upweighted.
        weights = torch.full((len(positions),), 1 / len(positions))
    else:
        weights = torch.cat(
            [
                torch.full((len(step),), 1 / (len(step) * plan.num_steps))
                for step in plan.step_positions
            ]
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
    chunk = int(config["runtime"]["lm_head_chunk_tokens"])
    kd_weight = float(aggregation.get("kd_weight", 1.0))
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
        float(aggregation["kd_temperature"]),
        float(aggregation["sft_weight"]),
        chunk,
        anomalies,
        kd_weight=kd_weight,
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
    js_normalized = step_js / math.log(experts)
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
                "js_mean": float(step_js[index]),
                "js_units": "nats",
                "rho": float(step_rho[index]),
                "rho_mapping": aggregation.get("rho_mapping", "ecdf"),
                "js_normalized": float(js_normalized[index]),
                "js_temperature": float(aggregation["js_temperature"]),
                "kd_temperature": float(aggregation["kd_temperature"]),
                "teacher_count": len(teacher_indices),
                "step_kd_loss": step_kd,
                "step_sft_loss": step_sft,
                "weighted_sft_loss": float(aggregation["sft_weight"]) * step_sft,
                "step_total_loss": kd_weight * step_kd
                + float(aggregation["sft_weight"]) * step_sft,
                "target_tail_mass": float(logq[start:end, -1].exp().mean()),
                "student_tail_mass": float(tokens["student_tail"][start:end].mean()),
                "target_entropy": float(tokens["entropy"][start:end].mean()),
                "support_size_mean": float(mask[start:end].sum(-1).float().mean()),
            }
        )
    return OutputSpaceGradientResult(
        gradients,
        kd_weight * kd + weighted,
        plan.num_steps,
        plan.num_steps,
        plan.discarded_steps,
        {
            "disagreement_sum": float(js_normalized.sum()),
            "rho_sum": float(step_rho.sum()),
            "kd_loss": kd,
            "sft_loss": sft,
            "weighted_sft_loss": weighted,
            "student_tail_mass": float((tokens["student_tail"] * weights).sum()),
            "target_tail_mass": float((logq[:, -1].exp() * weights).sum()),
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
