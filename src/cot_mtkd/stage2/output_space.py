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
from .online import Phase2RecordPlan, RecordGradientResult, _select_adapter
from .output_space_losses import adaptive_kd_hidden_gradient


@dataclass
class OutputSpaceGradientResult(RecordGradientResult):
    step_metrics: list[dict[str, Any]] = field(default_factory=list)


def runtime_options(runtime: dict[str, Any]) -> tuple[str, int]:
    storage = runtime.get("teacher_hidden_storage", "cpu")
    if storage not in ("cpu", "device"):
        raise ValueError("runtime.teacher_hidden_storage must be cpu or device")
    cache_gib = float(runtime.get("teacher_probability_cache_gib", 0.0))
    if not math.isfinite(cache_gib) or cache_gib < 0:
        raise ValueError("runtime.teacher_probability_cache_gib must be finite and nonnegative")
    return storage, int(cache_gib * 2**30)


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


def compute_record_gradient(
    model: torch.nn.Module,
    adapter_names: list[str],
    parameters: list[torch.nn.Parameter],
    record: PreparedRecord,
    tokenizer: Any,
    config: dict[str, Any],
    device: torch.device,
    profile: bool = False,
) -> OutputSpaceGradientResult:
    """PDF equations (1)--(19), with one final student-LoRA VJP per example.

    Teachers use no-grad forwards with configurable hidden-state storage and
    bounded probability caching. Head projections are chunked in tokens while
    retaining full vocabulary. Every retained step contributes equally; the
    target and its step rho depend only on teachers.
    """
    if len(adapter_names) < 2:
        raise ValueError("Disagreement-adaptive MTKD requires at least two teachers")
    hidden_storage, cache_bytes = runtime_options(config["runtime"])
    plan = plan_record(record, tokenizer, int(config["stage2"]["max_length"]))
    timers: dict[str, float] = {}

    def tick() -> float:
        if profile and device.type == "cuda":
            torch.cuda.synchronize(device)
        return time.perf_counter()

    def elapsed(name: str, started: float) -> None:
        timers[name] = timers.get(name, 0.0) + tick() - started

    if not plan.num_steps:
        return OutputSpaceGradientResult(
            zeros_like_parameters(parameters),
            0.0,
            0,
            0,
            plan.discarded_steps,
            {"disagreement_sum": 0.0, "rho_sum": 0.0},
            timers,
        )
    chunk = int(config["runtime"]["lm_head_chunk_tokens"])
    temperature = float(config["aggregation"]["temperature"])
    ids = torch.tensor([plan.input_ids], device=device, dtype=torch.long)
    positions = [position for step in plan.step_positions for position in step]
    hidden_indices = torch.tensor(positions, device=device, dtype=torch.long) - 1
    teacher_hidden = []
    started = tick()
    for name in adapter_names:
        _select_adapter(model, name, training=False)
        with torch.no_grad():
            output = forward_hidden(model, ids, torch.ones_like(ids), use_cache=False)
            selected = output.last_hidden_state[0].index_select(0, hidden_indices)
            teacher_hidden.append(
                selected.detach().cpu() if hidden_storage == "cpu" else selected.detach()
            )
        del selected, output
    elapsed("teacher_forward", started)

    _select_adapter(model, "student", training=True)
    started = tick()
    output = forward_hidden(model, ids, torch.ones_like(ids), use_cache=False)
    student_hidden = output.last_hidden_state[0].index_select(0, hidden_indices)
    del output
    elapsed("student_forward", started)
    _, head = decoder_and_lm_head(model)
    final_cotangent = torch.zeros_like(student_hidden)
    cursor, loss, disagreement_sum = 0, 0.0, 0.0
    step_metrics = []
    for step_index, step_positions in enumerate(plan.step_positions):
        count = len(step_positions)
        started = tick()
        cotangent, current_loss, disagreement = adaptive_kd_hidden_gradient(
            student_hidden[cursor : cursor + count],
            [hidden[cursor : cursor + count] for hidden in teacher_hidden],
            head,
            temperature,
            chunk,
            teacher_probability_cache_bytes=cache_bytes,
        )
        final_cotangent[cursor : cursor + count] = cotangent / plan.num_steps
        loss += current_loss / plan.num_steps
        disagreement_sum += disagreement
        step_metrics.append(
            {
                "step_index": step_index,
                "step_id": record.step_ids[step_positions[0]],
                "n_tokens": count,
                "token_start": step_positions[0],
                "token_end": step_positions[-1] + 1,
                "js_mean": disagreement * math.log(len(adapter_names)),
                "js_units": "nats",
                "js_normalized": disagreement,
                "rho": disagreement,
                "step_kd_loss": current_loss,
                "temperature": temperature,
                "teacher_count": len(adapter_names),
            }
        )
        cursor += count
        del cotangent
        elapsed("target_and_hidden_gradient", started)

    started = tick()
    gradients = torch.autograd.grad(
        student_hidden,
        parameters,
        grad_outputs=final_cotangent,
        create_graph=False,
        allow_unused=False,
    )
    result_gradients = [value.detach().float() for value in gradients]
    elapsed("final_kd_gradient", started)
    if not all(bool(torch.isfinite(value).all()) for value in result_gradients):
        raise FloatingPointError(f"Non-finite output-space KD gradient for {record.sample_id}")
    return OutputSpaceGradientResult(
        result_gradients,
        loss,
        plan.num_steps,
        plan.num_steps,
        plan.discarded_steps,
        {"disagreement_sum": disagreement_sum, "rho_sum": disagreement_sum},
        timers,
        step_metrics,
    )
