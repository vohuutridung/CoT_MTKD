"""Phase-2 student step: full-sequence SFT plus council-guided self-distillation.

For one prepared record the student (medoid-initialized LoRA) is run once over
the whole sequence. Reasoning tokens receive length-normalized SFT per step,
the token-wise masked self-tempered KL on ``V_k`` (``k >= 2``) and the binary
mass loss (``y* in V_k``); the answer block and the fixed format block receive
SFT only. ``V_k``, ``M(v)``, ``q`` and ``JS`` come from the council cache; the
dynamic temperature is derived at training time so that temperature
hyperparameters never invalidate the cache.
"""

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
from .council import (
    PlainSupervision,
    ReasoningSupervision,
    dynamic_temperature,
    effective_temperature,
    student_hidden_gradient,
)
from .online import RecordGradientResult, _select_adapter

FORMAT_REGIONS = (
    int(TokenRegion.ASSISTANT_CONTROL),
    int(TokenRegion.DELIMITER),
    int(TokenRegion.ANSWER_MARKER),
    int(TokenRegion.EOS),
)


@dataclass
class StudentPlan:
    """Supervised positions of one record: ``R`` steps and the two always-on blocks."""

    input_ids: list[int]
    step_positions: list[list[int]]
    answer_positions: list[int]
    format_positions: list[int]

    @property
    def num_steps(self) -> int:
        return len(self.step_positions)

    @property
    def reasoning_positions(self) -> list[int]:
        return [position for step in self.step_positions for position in step]

    @property
    def plain_positions(self) -> list[int]:
        return self.answer_positions + self.format_positions

    @property
    def sft_units(self) -> int:
        """``|R| + 2`` when both blocks exist; an empty block contributes no unit."""
        return self.num_steps + int(bool(self.answer_positions)) + int(bool(self.format_positions))


@dataclass
class CouncilGradientResult(RecordGradientResult):
    step_metrics: list[dict[str, Any]] = field(default_factory=list)


def plan_record(record: PreparedRecord, tokenizer: Any, max_length: int) -> StudentPlan:
    """Use the prepared token contract unchanged; no truncation happens here."""
    if max_length <= 0:
        raise ValueError("stage2.max_length must be positive")
    validate_token_contract(record.input_ids, record.labels, record.region_ids, record.step_ids)
    if len(record.input_ids) > max_length:
        raise ValueError(f"{record.sample_id}: prepared sequence exceeds stage2.max_length")
    step_ids = sorted({step for step in record.step_ids if step >= 0})
    steps = []
    for step in step_ids:
        content = [
            i
            for i, (region, value) in enumerate(zip(record.region_ids, record.step_ids))
            if value == step and region == int(TokenRegion.REASONING)
        ]
        if not content:
            raise ValueError(f"{record.sample_id}: reasoning step {step} has no content")
        steps.append(content)
    answer = [i for i, r in enumerate(record.region_ids) if r == int(TokenRegion.ANSWER)]
    fixed = [i for i, r in enumerate(record.region_ids) if r in FORMAT_REGIONS]
    supervised = [p for s in steps for p in s] + answer + fixed
    if any(record.labels[p] != record.input_ids[p] for p in supervised):
        raise ValueError(f"{record.sample_id}: supervised token label mismatch")
    if supervised and min(supervised) < 1:
        raise ValueError(f"{record.sample_id}: supervised token without a preceding context")
    return StudentPlan(list(record.input_ids), steps, answer, fixed)


def unpack_cached_support(
    value: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad ragged cached ``V_k``/``M(v)`` to ``[tokens, width]`` in RAM."""
    offsets = value["support_offsets"].long()
    sizes = offsets.diff()
    count = len(sizes)
    width = int(sizes.max()) if count else 0
    ids = torch.zeros((count, width), dtype=torch.long)
    mask = torch.arange(width)[None, :] < sizes[:, None]
    variance_mask = torch.zeros((count, width), dtype=torch.float64)
    for index, (start, end) in enumerate(zip(offsets[:-1].tolist(), offsets[1:].tolist())):
        if end <= start:
            raise RuntimeError("Cached support must contain at least one category per token")
        ids[index, : end - start] = value["support_ids"][start:end].long()
        variance_mask[index, : end - start] = value["support_mask"][start:end].double()
    return ids, mask, variance_mask


def _empty_result(parameters, plan: StudentPlan) -> CouncilGradientResult:
    return CouncilGradientResult(
        zeros_like_parameters(parameters),
        0.0,
        0,
        0,
        0,
        {
            "sequence_tokens": len(plan.input_ids),
            "reasoning_tokens": 0,
            "supervised_tokens": 0,
            "head_chunks": 0,
        },
        {},
    )


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
    js_median: float | None = None,
    js_p95: float | None = None,
) -> CouncilGradientResult:
    """Student-only gradient of ``L_SFT + alpha D_KL + beta L_mass`` for one record."""
    if cached_target is None:
        raise RuntimeError("Phase 2 requires a council cache; run stage2-cache first")
    council = config["council"]
    alpha, beta = float(council["alpha"]), float(council["beta"])
    epsilon_m = float(council["epsilon_m"])
    chunk = int(config["runtime"]["lm_head_chunk_tokens"])
    plan = plan_record(record, tokenizer, int(config["stage2"]["max_length"]))
    reasoning_positions = plan.reasoning_positions
    plain_positions = plan.plain_positions
    if not reasoning_positions and not plain_positions:
        return _empty_result(parameters, plan)
    expected_offsets = [0]
    for step in plan.step_positions:
        expected_offsets.append(expected_offsets[-1] + len(step))
    if (
        cached_target["token_positions"].tolist() != reasoning_positions
        or cached_target["step_offsets"].tolist() != expected_offsets
    ):
        raise RuntimeError(f"{record.sample_id}: council cache token/step mapping mismatch")
    units = plan.sft_units
    count = len(reasoning_positions)
    targets = torch.tensor([record.input_ids[p] for p in reasoning_positions], dtype=torch.long)
    reasoning: ReasoningSupervision | None = None
    tau = k = js = q = gold_in = kl_tokens = mass_tokens = None
    if count:
        support_ids, support_mask, variance_mask = unpack_cached_support(cached_target)
        k = cached_target["k"].long()
        js = cached_target["js"].double()
        q = cached_target["q"].double()
        gold_in = cached_target["gold_in_support"].bool()
        if not torch.equal(support_mask.sum(-1), k):
            raise RuntimeError(f"{record.sample_id}: cached k disagrees with support sizes")
        if not torch.equal(((support_ids == targets[:, None]) & support_mask).any(-1), gold_in):
            raise RuntimeError(f"{record.sample_id}: cached gold membership mismatch")
        tau = dynamic_temperature(js, council, len(adapter_names), js_median, js_p95)
        tau_eff = effective_temperature(variance_mask, tau)
        kl_tokens = k >= 2
        mass_tokens = gold_in
        sft_weight = torch.cat(
            [torch.full((len(step),), 1.0 / (len(step) * units)) for step in plan.step_positions]
        )
        kl_weight = torch.where(
            kl_tokens, torch.full_like(sft_weight, alpha / max(int(kl_tokens.sum()), 1)), 0.0
        )
        mass_weight = torch.where(
            mass_tokens, torch.full_like(sft_weight, beta / max(int(mass_tokens.sum()), 1)), 0.0
        )
        reasoning = ReasoningSupervision(
            targets, support_ids, support_mask, tau_eff, q, sft_weight, kl_weight, mass_weight
        )
    plain: PlainSupervision | None = None
    if plain_positions:
        blocks = [plan.answer_positions, plan.format_positions]
        plain = PlainSupervision(
            torch.tensor([record.input_ids[p] for p in plain_positions], dtype=torch.long),
            torch.cat([torch.full((len(b),), 1.0 / (len(b) * units)) for b in blocks if b]),
        )

    def tick() -> float:
        if profile and device.type == "cuda":
            torch.cuda.synchronize(device)
        return time.perf_counter()

    timers: dict[str, float] = {}
    started = tick()
    _select_adapter(model, "student", training=True)
    ids = torch.tensor([plan.input_ids], device=device, dtype=torch.long)
    output = forward_hidden(model, ids, torch.ones_like(ids), use_cache=False)
    rows = torch.tensor(reasoning_positions + plain_positions, device=device) - 1
    hidden = output.last_hidden_state[0].index_select(0, rows)
    del output
    timers["student_forward"] = tick() - started
    _, head = decoder_and_lm_head(model)
    started = tick()
    cotangent, terms = student_hidden_gradient(hidden, head, reasoning, plain, chunk, epsilon_m)
    timers["head_chunks_and_hidden_gradient"] = tick() - started
    started = tick()
    gradients = torch.autograd.grad(hidden, parameters, grad_outputs=cotangent, allow_unused=False)
    gradients = [g.detach().float() for g in gradients]
    timers["adapter_gradient"] = tick() - started
    if not all(bool(torch.isfinite(g).all()) for g in gradients):
        raise FloatingPointError(f"Nonfinite Phase-2 gradient: {record.sample_id}")

    sft_total = 0.0
    if reasoning is not None:
        sft_total += float((terms["sft"] * reasoning.sft_weight).sum())
    if plain is not None:
        sft_total += float((terms["plain_sft"] * plain.sft_weight).sum())
    metrics: dict[str, float] = {
        "sequence_tokens": len(plan.input_ids),
        "reasoning_tokens": count,
        "supervised_tokens": count + len(plain_positions),
        "head_chunks": math.ceil(count / chunk) + math.ceil(len(plain_positions) / chunk),
        "sft_loss": sft_total,
        "kl_loss": 0.0,
        "mass_loss": 0.0,
        "kl_tokens": 0,
        "mass_tokens": 0,
        "k1_tokens": 0,
        "gold_missing_tokens": 0,
        "js_sum": 0.0,
        "tau_sum": 0.0,
        "k_sum": 0.0,
        "tail_mass_sum": 0.0,
        "entropy_sum": 0.0,
        "abs_mass_gap_sum": 0.0,
        "kl_sharpen_sum": 0.0,
        "kl_sharpen_tokens": 0,
        "kl_flatten_sum": 0.0,
        "kl_flatten_tokens": 0,
    }
    step_metrics: list[dict[str, Any]] = []
    if reasoning is not None:
        kl_values = terms["kl"].double()
        mass_values = terms["mass"].double()
        kl_mean = float(kl_values[kl_tokens].mean()) if bool(kl_tokens.any()) else 0.0
        mass_mean = float(mass_values[mass_tokens].mean()) if bool(mass_tokens.any()) else 0.0
        gap = (terms["student_mass"].double() - q).abs()
        sharpen = kl_tokens & (tau < 1.0)
        flatten = kl_tokens & (tau >= 1.0)
        metrics.update(
            kl_loss=kl_mean,
            mass_loss=mass_mean,
            kl_tokens=int(kl_tokens.sum()),
            mass_tokens=int(mass_tokens.sum()),
            k1_tokens=int((k == 1).sum()),
            gold_missing_tokens=int((~gold_in).sum()),
            js_sum=float(js.sum()),
            tau_sum=float(tau.sum()),
            k_sum=float(k.sum()),
            tail_mass_sum=float((1.0 - q).sum()),
            entropy_sum=float(terms["entropy"].double()[kl_tokens].sum()),
            abs_mass_gap_sum=float(gap[mass_tokens].sum()),
            kl_sharpen_sum=float(kl_values[sharpen].sum()),
            kl_sharpen_tokens=int(sharpen.sum()),
            kl_flatten_sum=float(kl_values[flatten].sum()),
            kl_flatten_tokens=int(flatten.sum()),
        )
        for index, step in enumerate(plan.step_positions):
            start, end = expected_offsets[index : index + 2]
            local_kl, local_mass = kl_tokens[start:end], mass_tokens[start:end]
            step_metrics.append(
                {
                    "step_index": index,
                    "step_id": record.step_ids[step[0]],
                    "n_tokens": len(step),
                    "token_start": step[0],
                    "token_end": step[-1] + 1,
                    "js_mean": float(js[start:end].mean()),
                    "js_units": "nats",
                    "tau_mean": float(tau[start:end].mean()),
                    "k_mean": float(k[start:end].double().mean()),
                    "k1_fraction": float((k[start:end] == 1).double().mean()),
                    "gold_missing_fraction": float((~gold_in[start:end]).double().mean()),
                    "step_sft_loss": float(terms["sft"][start:end].double().mean()),
                    "step_kl_loss": (
                        float(kl_values[start:end][local_kl].mean())
                        if bool(local_kl.any())
                        else 0.0
                    ),
                    "step_mass_loss": (
                        float(mass_values[start:end][local_mass].mean())
                        if bool(local_mass.any())
                        else 0.0
                    ),
                    "student_entropy_on_support": float(
                        terms["entropy"][start:end].double().mean()
                    ),
                    "council_tail_mass": float((1.0 - q[start:end]).mean()),
                    "student_tail_mass": float(
                        (1.0 - terms["student_mass"][start:end].double()).mean()
                    ),
                    "abs_mass_gap": float(gap[start:end].mean()),
                }
            )
    total = sft_total + alpha * metrics["kl_loss"] + beta * metrics["mass_loss"]
    return CouncilGradientResult(
        gradients,
        total,
        plan.num_steps,
        plan.num_steps or int(bool(plain_positions)),
        0,
        metrics,
        timers,
        step_metrics,
    )
