from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from ..data.schema import PreparedRecord, TokenRegion
from ..models.chunked_head import (
    cross_entropy_hidden_gradient,
    decoder_and_lm_head,
    forward_hidden,
)
from ..models.multi_adapter import (
    adapter_parameter_map,
    create_multi_adapter_model,
    load_adapter_bundle,
    load_adapter_state,
    lora_config,
    require_same_model_source,
    set_active_adapter,
)
from ..signals.pag import record_pag_parts
from ..utils.manifest import read_json, require_file_sha256
from ..utils.training import zeros_like_parameters
from .geometry import task_anchored_weights
from .geometry_losses import blended_kd_hidden_gradient, teacher_kd_hidden_gradient


@dataclass
class Phase2RecordPlan:
    input_ids: list[int]
    step_positions: list[list[int]]
    prefix_ends: list[int]
    answer_prefix: list[int]
    solution: list[int]
    discarded_steps: int

    @property
    def num_steps(self) -> int:
        return len(self.step_positions)

    @property
    def max_anchor_length(self) -> int:
        return len(self.input_ids) + len(self.answer_prefix) + len(self.solution)


@dataclass
class RecordGradientResult:
    gradients: list[torch.Tensor]
    loss: float
    steps: int
    active_steps: int
    discarded_steps: int
    metrics: dict[str, float]
    timings: dict[str, float]


def plan_record(record: PreparedRecord, tokenizer: Any, max_length: int) -> Phase2RecordPlan:
    """Reserve the complete raw gold solution, retaining only complete R steps.

    KD positions contain REASONING content only; anchor prefixes also include
    each step's trailing delimiters. The gold answer marker is context only.
    """
    prefix, _, answer_prefix, solution = record_pag_parts(record, tokenizer)
    suffix_length = len(answer_prefix) + len(solution)
    if len(prefix) + suffix_length > max_length:
        raise ValueError(
            f"{record.sample_id}: prompt plus full gold solution exceeds "
            f"stage2.max_length={max_length}; gold solutions are never truncated"
        )
    step_ids = sorted({step for step in record.step_ids if step >= 0})
    positions, prefix_ends = [], []
    for step in step_ids:
        all_positions = [i for i, value in enumerate(record.step_ids) if value == step]
        end = max(all_positions) + 1
        if end + suffix_length > max_length:
            break
        content = [i for i in all_positions if record.region_ids[i] == int(TokenRegion.REASONING)]
        if not content or min(content) < 1:
            raise ValueError(f"{record.sample_id}: invalid reasoning content for step {step}")
        positions.append(content)
        prefix_ends.append(end)
    end = prefix_ends[-1] if prefix_ends else len(prefix)
    return Phase2RecordPlan(
        input_ids=record.input_ids[:end],
        step_positions=positions,
        prefix_ends=prefix_ends,
        answer_prefix=answer_prefix,
        solution=solution,
        discarded_steps=len(step_ids) - len(positions),
    )


def _select_adapter(model: torch.nn.Module, name: str, training: bool) -> None:
    set_active_adapter(model, name)
    if training:
        model.train()
    else:
        model.eval()
        # The shared helper makes the selected adapter trainable. Teachers must
        # remain frozen even when selected for their no-grad forward.
        for parameter in model.parameters():
            parameter.requires_grad_(False)


def create_online_model(config: dict[str, Any], distributed: Any):
    stage1_dir = Path(config["paths"]["stage1"])
    from .council_cache import load_council_cache

    prepared = read_json(Path(config["paths"]["prepared"]) / "manifest.json")
    stage1 = read_json(stage1_dir / "manifest.json")
    cache = load_council_cache(config, prepared, stage1)
    require_file_sha256(stage1_dir, stage1, "adapter_bundle", "adapter_bundle_sha256")
    require_file_sha256(stage1_dir, stage1, "config_file", "config_file_sha256")
    stage1_config = stage1["config"]
    require_same_model_source(config["model"], stage1_config["model"], "Phase 2/Stage 1")
    for key in ("rank", "alpha", "target_modules"):
        if config["lora"][key] != stage1_config["lora"][key]:
            raise ValueError(f"Best-expert cloning requires matching LoRA {key}")
    names = list(stage1["adapter_names"])
    if len(names) < 2:
        raise ValueError("Task-anchored MTKD requires at least two teachers")
    model, created_names = create_multi_adapter_model(
        config["model"],
        stage1_config["lora"],
        len(names),
        distributed.device,
        int(stage1_config["seed"]),
    )
    if created_names != names:
        raise RuntimeError("Teacher adapter names do not match Stage 1")
    bundle = load_adapter_bundle(stage1_dir / stage1["adapter_bundle"])
    for name in names:
        load_adapter_state(model, name, bundle[name])
    model.add_adapter("student", lora_config(config["lora"]))
    base_dtype = next(model.get_base_model().parameters()).dtype
    for parameter in adapter_parameter_map(model, "student").values():
        parameter.data = parameter.data.to(dtype=base_dtype)
    best_name = cache.manifest["selected_expert"]
    load_adapter_state(model, "student", bundle[best_name])
    if int(config["stage2"]["max_length"]) > int(model.config.max_position_embeddings):
        raise ValueError("Configured Phase-2 context exceeds the model context limit")
    _select_adapter(model, "student", training=True)
    parameters = list(adapter_parameter_map(model, "student").values())
    return model, names, parameters


def compute_record_gradient(
    model: torch.nn.Module,
    adapter_names: list[str],
    parameters: list[torch.nn.Parameter],
    record: PreparedRecord,
    tokenizer: Any,
    config: dict[str, Any],
    device: torch.device,
    profile: bool = False,
) -> RecordGradientResult:
    """Exact per-example geometry and KD-only VJP, with bounded dense logits.

    Frozen teachers share the student backbone. Only teacher hidden states are
    copied to CPU. At most one student trajectory graph and one anchor graph
    coexist; per-step parameter gradients are released before the next step.
    """
    plan = plan_record(record, tokenizer, int(config["stage2"]["max_length"]))
    timers: dict[str, float] = {}

    def tick() -> float:
        if profile and device.type == "cuda":
            torch.cuda.synchronize(device)
        return time.perf_counter()

    def elapsed(name: str, start: float) -> None:
        timers[name] = timers.get(name, 0.0) + tick() - start

    if not plan.num_steps:
        return RecordGradientResult(
            zeros_like_parameters(parameters),
            0.0,
            0,
            0,
            plan.discarded_steps,
            {"agreement_sum": 0.0, "weight_sum": 0.0, "consensus_sum": 0.0},
            timers,
        )
    chunk = int(config["runtime"]["lm_head_chunk_tokens"])
    geometry = config["geometry"]
    temperature = float(geometry["temperature"])
    ids = torch.tensor([plan.input_ids], device=device, dtype=torch.long)
    content_positions = [position for step in plan.step_positions for position in step]
    hidden_indices = torch.tensor(content_positions, device=device, dtype=torch.long) - 1
    teacher_hidden = []
    started = tick()
    for name in adapter_names:
        _select_adapter(model, name, training=False)
        with torch.no_grad():
            output = forward_hidden(model, ids, torch.ones_like(ids), use_cache=False)
            selected = output.last_hidden_state[0].index_select(0, hidden_indices)
            teacher_hidden.append(selected.cpu())
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
    cursor, active, loss = 0, 0, 0.0
    agreement_sum = weight_sum = consensus_sum = anchor_loss_sum = 0.0
    for positions, prefix_end in zip(plan.step_positions, plan.prefix_ends, strict=True):
        count = len(positions)
        current = student_hidden[cursor : cursor + count]
        teachers = [hidden[cursor : cursor + count] for hidden in teacher_hidden]
        gradients = []
        started = tick()
        for teacher in teachers:
            cotangent, _ = teacher_kd_hidden_gradient(current, teacher, head, temperature, chunk)
            values = torch.autograd.grad(
                current,
                parameters,
                grad_outputs=cotangent,
                retain_graph=True,
                create_graph=False,
                allow_unused=False,
            )
            gradients.append([value.detach() for value in values])
        del values, cotangent
        elapsed("teacher_kd_gradients", started)

        started = tick()
        anchor_ids = torch.tensor(
            [plan.input_ids[:prefix_end] + plan.answer_prefix + plan.solution],
            device=device,
            dtype=torch.long,
        )
        # No detached KV cache: h_s includes dependence through the full prefix.
        anchor_output = forward_hidden(model, anchor_ids, torch.ones_like(anchor_ids), False)
        answer_start = prefix_end + len(plan.answer_prefix)
        answer_hidden = anchor_output.last_hidden_state[
            0, answer_start - 1 : answer_start + len(plan.solution) - 1
        ]
        anchor_cotangent, anchor_loss, answer_count = cross_entropy_hidden_gradient(
            answer_hidden, head, torch.tensor(plan.solution, device=device), chunk
        )
        anchor_cotangent.div_(answer_count)
        anchor_gradients = torch.autograd.grad(
            answer_hidden,
            parameters,
            grad_outputs=anchor_cotangent,
            create_graph=False,
            allow_unused=False,
        )
        anchor_loss_sum += float(anchor_loss.item()) / answer_count
        del anchor_output, answer_hidden, anchor_cotangent, anchor_ids, anchor_loss
        elapsed("answer_anchor_gradients", started)

        started = tick()
        weights = task_anchored_weights(
            gradients,
            list(anchor_gradients),
            float(geometry["epsilon_a"]),
            float(geometry["epsilon_u"]),
        )
        del gradients, anchor_gradients
        agreement_sum += weights.agreement
        weight_sum += weights.step_weight
        consensus_sum += weights.consensus_weight
        elapsed("geometry", started)
        if weights.step_weight > 0.0:
            started = tick()
            cotangent, current_loss = blended_kd_hidden_gradient(
                current,
                teachers,
                head,
                weights.teacher_weights,
                weights.consensus_weight,
                temperature,
                chunk,
            )
            scale = weights.step_weight / plan.num_steps
            final_cotangent[cursor : cursor + count] = cotangent * scale
            loss += current_loss * scale
            active += 1
            del cotangent
            elapsed("target_and_hidden_gradient", started)
        cursor += count
        del current, teachers, weights

    started = tick()
    if active:
        final_gradients = torch.autograd.grad(
            student_hidden,
            parameters,
            grad_outputs=final_cotangent,
            create_graph=False,
            allow_unused=False,
        )
        result_gradients = [value.detach().float() for value in final_gradients]
    else:
        result_gradients = zeros_like_parameters(parameters)
    elapsed("final_kd_gradient", started)
    if not all(torch.isfinite(value).all().item() for value in result_gradients):
        raise FloatingPointError(f"Non-finite final KD gradient for {record.sample_id}")
    return RecordGradientResult(
        result_gradients,
        loss,
        plan.num_steps,
        active,
        plan.discarded_steps,
        {
            "agreement_sum": agreement_sum,
            "weight_sum": weight_sum,
            "consensus_sum": consensus_sum,
            "anchor_loss_sum": anchor_loss_sum,
        },
        timers,
    )
