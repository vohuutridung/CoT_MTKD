from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

from ..data.collator import LongCoTCollator, shifted_token_views
from ..data.dataset import JsonlRecordDataset
from ..data.prepare import tokenizer_fingerprint
from ..data.schema import TokenRegion
from ..models.chunked_head import (
    cross_entropy_hidden_gradient,
    decoder_and_lm_head,
    forward_hidden,
    full_vocab_probe,
    gather_hidden_positions,
    gather_support_logits,
    support_vjp_hidden_gradient,
)
from ..models.multi_adapter import (
    adapter_parameter_groups,
    create_multi_adapter_model,
    extract_adapter_state,
    load_adapter_state,
    load_tokenizer,
    require_same_model_source,
    save_adapter_bundle,
    set_active_adapter,
    set_all_adapters_trainable,
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
from ..utils.seed import derived_seed, deterministic_rng
from ..utils.training import (
    add_gradients_,
    assign_gradients,
    cosine_warmup_lambda,
    divide_gradients_,
    global_clip_grad_list_,
    interaction_scale,
    vector_norm,
    zeros_like_parameters,
)
from .dpp import DPPMetrics, normalized_support_features, step_dpp_loss
from .gac_gradient import GACDiagnostics, stable_gac_gradients
from .kneedle import build_union_support, capped_k_from_probe
from .rbf import BandwidthEMA, effective_update_distances, repulsion_updates

LOGGER = logging.getLogger(__name__)


@dataclass
class ProbeResult:
    support_ids: torch.Tensor
    support_mask: torch.Tensor
    dpp_logit_gradients: torch.Tensor
    dpp_loss_sum: float
    dpp_sample_count: int
    dpp_metrics: DPPMetrics
    mean_selected_k: float
    cap_rate: float
    probe_saturation_rate: float
    selection_count: int


def _batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: (
            value.to(device, non_blocking=True)
            if isinstance(value, torch.Tensor)
            else value
        )
        for key, value in batch.items()
    }


def _empty_probe(device: torch.device, expert_count: int) -> ProbeResult:
    return ProbeResult(
        support_ids=torch.empty((0, 0), dtype=torch.long, device=device),
        support_mask=torch.empty((0, 0), dtype=torch.bool, device=device),
        dpp_logit_gradients=torch.empty(
            (expert_count, 0, 0), dtype=torch.float32, device=device
        ),
        dpp_loss_sum=0.0,
        dpp_sample_count=0,
        dpp_metrics=DPPMetrics(0, 0, 0, 1.0e-4),
        mean_selected_k=0.0,
        cap_rate=0.0,
        probe_saturation_rate=0.0,
        selection_count=0,
    )


def probe_stage1_dpp(
    model: torch.nn.Module,
    adapter_names: list[str],
    batch: dict[str, Any],
    global_step: int,
    rng_stream: int,
    base_seed: int,
    config: dict[str, Any],
    device: torch.device,
    *,
    reasoning_hidden_by_expert: list[torch.Tensor] | None = None,
) -> ProbeResult:
    views = shifted_token_views(batch)
    token_count = int(views["reasoning_targets"].numel())
    if token_count == 0:
        return _empty_probe(device, len(adapter_names))
    _, head = decoder_and_lm_head(model)
    probe_hidden_device = torch.device(
        config["runtime"].get("probe_hidden_device", "cpu")
    )
    if probe_hidden_device.type == device.type and probe_hidden_device.index is None:
        probe_hidden_device = device
    hidden_by_expert: list[torch.Tensor] = []
    top_values: list[torch.Tensor] = []
    top_ids: list[torch.Tensor] = []
    minima: list[torch.Tensor] = []
    maxima: list[torch.Tensor] = []
    chunk_tokens = int(config["runtime"]["lm_head_chunk_tokens"])
    probe_k = int(config["kneedle"]["probe_k"])
    if reasoning_hidden_by_expert is not None and len(
        reasoning_hidden_by_expert
    ) != len(adapter_names):
        raise ValueError(
            "One-pass probe requires one reasoning hidden tensor per expert"
        )

    model.train()
    for expert, adapter_name in enumerate(adapter_names):
        if reasoning_hidden_by_expert is None:
            set_active_adapter(model, adapter_name)
            seed = derived_seed(base_seed, "stage1", global_step, rng_stream, expert)
            with deterministic_rng(seed, device), torch.no_grad():
                outputs = forward_hidden(
                    model, batch["input_ids"], batch["attention_mask"], use_cache=False
                )
                selected_hidden = gather_hidden_positions(
                    outputs.last_hidden_state,
                    views["reasoning_batch_indices"],
                    views["reasoning_hidden_indices"],
                ).to(probe_hidden_device)
        else:
            # Keep the decoder graph in the caller, but never build a graph
            # through the full-vocabulary Top-k and support selection.
            selected_hidden = (
                reasoning_hidden_by_expert[expert].detach().to(probe_hidden_device)
            )
        values, ids, minimum, maximum = full_vocab_probe(
            selected_hidden,
            head,
            views["reasoning_targets"],
            probe_k,
            chunk_tokens,
            output_device=device,
        )
        hidden_by_expert.append(selected_hidden)
        top_values.append(values)
        top_ids.append(ids)
        minima.append(minimum)
        maxima.append(maximum)

    selected_k = torch.stack(
        [
            capped_k_from_probe(
                values,
                minimum,
                maximum,
                vocab_size=head.weight.shape[0],
                min_k=int(config["kneedle"]["min_k"]),
                max_k=int(config["kneedle"]["max_k"]),
            )
            for values, minimum, maximum in zip(top_values, minima, maxima, strict=True)
        ]
    )
    support_ids, support_mask = build_union_support(
        torch.stack(top_ids), selected_k
    )
    support_logits = torch.stack(
        [
            gather_support_logits(
                hidden, head, support_ids, chunk_tokens, output_device=device
            )
            for hidden in hidden_by_expert
        ],
        dim=0,
    ).detach()
    support_logits.requires_grad_(True)
    features = normalized_support_features(support_logits, support_mask)
    loss, metrics = step_dpp_loss(
        features,
        views["reasoning_batch_indices"],
        views["reasoning_step_ids"],
        jitter=float(config["dpp"]["jitter"]),
        maximum_jitter=float(config["dpp"]["max_jitter"]),
        reduction="sum",
    )
    gradients = torch.autograd.grad(loss, support_logits)[0].detach()
    cap = int(config["kneedle"]["max_k"])
    probe_limit = int(config["kneedle"]["probe_k"])
    raw_elbow = torch.stack(
        [
            capped_k_from_probe(
                values,
                minimum,
                maximum,
                vocab_size=head.weight.shape[0],
                min_k=1,
                max_k=probe_limit,
            )
            for values, minimum, maximum in zip(top_values, minima, maxima, strict=True)
        ]
    )
    return ProbeResult(
        support_ids=support_ids,
        support_mask=support_mask,
        dpp_logit_gradients=gradients,
        dpp_loss_sum=float(loss.detach().item()),
        dpp_sample_count=metrics.samples,
        dpp_metrics=metrics,
        mean_selected_k=float(selected_k.float().mean().item()),
        cap_rate=float((raw_elbow > cap).float().mean().item()),
        probe_saturation_rate=float((raw_elbow == probe_limit).float().mean().item()),
        selection_count=int(selected_k.numel()),
    )


def replay_expert_gradients(
    model: torch.nn.Module,
    adapter_name: str,
    expert_index: int,
    parameters: list[torch.nn.Parameter],
    batch: dict[str, Any],
    probe: ProbeResult,
    global_step: int,
    rng_stream: int,
    base_seed: int,
    chunk_tokens: int,
    device: torch.device,
    combined_dpp_scale: float | None = None,
) -> tuple[list[torch.Tensor], list[torch.Tensor], float, int]:
    views = shifted_token_views(batch)
    set_active_adapter(model, adapter_name)
    model.train()
    seed = derived_seed(base_seed, "stage1", global_step, rng_stream, expert_index)
    with deterministic_rng(seed, device):
        outputs = forward_hidden(
            model, batch["input_ids"], batch["attention_mask"], use_cache=False
        )
    response_hidden = gather_hidden_positions(
        outputs.last_hidden_state,
        views["response_batch_indices"],
        views["response_hidden_indices"],
    )
    return _expert_gradients_from_hidden(
        model,
        response_hidden,
        parameters,
        batch,
        probe,
        expert_index,
        chunk_tokens,
        combined_dpp_scale,
    )


def _expert_gradients_from_hidden(
    model: torch.nn.Module,
    response_hidden: torch.Tensor,
    parameters: list[torch.nn.Parameter],
    batch: dict[str, Any],
    probe: ProbeResult,
    expert_index: int,
    chunk_tokens: int,
    combined_dpp_scale: float | None = None,
) -> tuple[list[torch.Tensor], list[torch.Tensor], float, int]:
    views = shifted_token_views(batch)
    _, head = decoder_and_lm_head(model)
    sft_hidden_gradient, sft_loss_sum, sft_count = cross_entropy_hidden_gradient(
        response_hidden, head, views["response_targets"], chunk_tokens
    )
    dpp_full_hidden_gradient = torch.zeros_like(response_hidden)
    if probe.support_ids.numel() > 0:
        response_regions = views["regions"][views["valid"]]
        response_reasoning = response_regions.eq(int(TokenRegion.REASONING))
        reasoning_hidden = response_hidden[response_reasoning]
        dpp_hidden_gradient = support_vjp_hidden_gradient(
            reasoning_hidden,
            head,
            probe.support_ids,
            probe.dpp_logit_gradients[expert_index],
            chunk_tokens,
        )
        dpp_full_hidden_gradient[response_reasoning] = dpp_hidden_gradient

    if combined_dpp_scale is not None:
        combined_parameter_gradients = torch.autograd.grad(
            response_hidden,
            parameters,
            grad_outputs=sft_hidden_gradient
            + float(combined_dpp_scale) * dpp_full_hidden_gradient,
            allow_unused=False,
        )
        return (
            [value.detach().float() for value in combined_parameter_gradients],
            [],
            float(sft_loss_sum.item()),
            int(sft_count),
        )
    sft_parameter_gradients = torch.autograd.grad(
        response_hidden,
        parameters,
        grad_outputs=sft_hidden_gradient,
        retain_graph=probe.support_ids.numel() > 0,
        allow_unused=False,
    )
    if probe.support_ids.numel() > 0:
        dpp_parameter_gradients = torch.autograd.grad(
            response_hidden,
            parameters,
            grad_outputs=dpp_full_hidden_gradient,
            retain_graph=False,
            allow_unused=False,
        )
    else:
        dpp_parameter_gradients = tuple(torch.zeros_like(value) for value in parameters)
    return (
        [value.detach().float() for value in sft_parameter_gradients],
        [value.detach().float() for value in dpp_parameter_gradients],
        float(sft_loss_sum.item()),
        int(sft_count),
    )


def one_pass_expert_gradients(
    model: torch.nn.Module,
    adapter_names: list[str],
    parameter_lists: list[list[torch.nn.Parameter]],
    batch: dict[str, Any],
    global_step: int,
    rng_stream: int,
    base_seed: int,
    config: dict[str, Any],
    device: torch.device,
    combined_dpp_scale: float | None = None,
) -> tuple[
    ProbeResult, list[tuple[list[torch.Tensor], list[torch.Tensor], float, int]]
]:
    """Forward each expert once, retaining its checkpointed graph until DPP is known."""
    views = shifted_token_views(batch)
    reasoning_in_response = views["regions"][views["valid"]].eq(
        int(TokenRegion.REASONING)
    )
    response_hiddens: list[torch.Tensor] = []
    reasoning_hiddens: list[torch.Tensor] = []
    model.train()
    for expert, adapter_name in enumerate(adapter_names):
        set_active_adapter(model, adapter_name)
        seed = derived_seed(base_seed, "stage1", global_step, rng_stream, expert)
        with deterministic_rng(seed, device):
            outputs = forward_hidden(
                model, batch["input_ids"], batch["attention_mask"], use_cache=False
            )
        response_hidden = gather_hidden_positions(
            outputs.last_hidden_state,
            views["response_batch_indices"],
            views["response_hidden_indices"],
        )
        response_hiddens.append(response_hidden)
        reasoning_hiddens.append(response_hidden[reasoning_in_response])
    probe = probe_stage1_dpp(
        model,
        adapter_names,
        batch,
        global_step,
        rng_stream,
        base_seed,
        config,
        device,
        reasoning_hidden_by_expert=reasoning_hiddens,
    )
    del reasoning_hiddens
    gradients = []
    for expert, (adapter_name, parameters, response_hidden) in enumerate(
        zip(adapter_names, parameter_lists, response_hiddens, strict=True)
    ):
        # Non-reentrant gradient checkpointing recomputes modules during backward.
        # Restore the adapter used by this expert's original forward first.
        set_active_adapter(model, adapter_name)
        gradients.append(
            _expert_gradients_from_hidden(
                model,
                response_hidden,
                parameters,
                batch,
                probe,
                expert,
                int(config["runtime"]["lm_head_chunk_tokens"]),
                combined_dpp_scale,
            )
        )
    return probe, gradients


def sft_only_expert_gradients(
    model: torch.nn.Module,
    adapter_name: str,
    expert_index: int,
    parameters: list[torch.nn.Parameter],
    batch: dict[str, Any],
    global_step: int,
    rng_stream: int,
    base_seed: int,
    chunk_tokens: int,
    device: torch.device,
) -> tuple[list[torch.Tensor], list[torch.Tensor], float, int]:
    """Warm-up path: one expert forward and one SFT transformer VJP."""
    views = shifted_token_views(batch)
    set_active_adapter(model, adapter_name)
    model.train()
    seed = derived_seed(base_seed, "stage1", global_step, rng_stream, expert_index)
    with deterministic_rng(seed, device):
        outputs = forward_hidden(
            model, batch["input_ids"], batch["attention_mask"], use_cache=False
        )
    response_hidden = gather_hidden_positions(
        outputs.last_hidden_state,
        views["response_batch_indices"],
        views["response_hidden_indices"],
    )
    _, head = decoder_and_lm_head(model)
    hidden_gradient, loss_sum, token_count = cross_entropy_hidden_gradient(
        response_hidden, head, views["response_targets"], chunk_tokens
    )
    gradients = torch.autograd.grad(
        response_hidden, parameters, grad_outputs=hidden_gradient, allow_unused=False
    )
    return (
        [value.detach().float() for value in gradients],
        [],
        float(loss_sum.item()),
        int(token_count),
    )


def _optimizer_and_scheduler(
    parameters: list[torch.nn.Parameter], config: dict[str, Any], total_steps: int
):
    optimizer_config = config["optimizer"]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(config["stage1"]["learning_rate"]),
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
    return optimizer, scheduler


def _stable_config(config: dict[str, Any]) -> dict[str, Any]:
    """Fingerprint semantic training settings while allowing a resume path to change."""
    value = {key: item for key, item in config.items() if not key.startswith("_")}
    value = {
        key: (dict(item) if isinstance(item, dict) else item)
        for key, item in value.items()
    }
    if isinstance(value.get("stage1"), dict):
        value["stage1"].pop("resume_from", None)
    return value


def _run_compatible_config(config: dict[str, Any]) -> dict[str, Any]:
    """A forward implementation switch does not change the training objective."""
    value = _stable_config(config)
    if isinstance(value.get("stage1"), dict):
        value["stage1"].pop("forward_mode", None)
    return value


def _validate_stage1_config(config: dict[str, Any]) -> None:
    if config["stage1"].get("forward_mode", "one_pass") not in {"one_pass", "two_pass"}:
        raise ValueError("stage1.forward_mode must be one_pass or two_pass")
    if str(config["optimizer"].get("name", "")).lower() != "adamw":
        raise ValueError("Stage 1 currently implements optimizer.name: adamw")
    if str(config["scheduler"].get("name", "")).lower() != "cosine":
        raise ValueError("Stage 1 currently implements scheduler.name: cosine")
    if int(config["stage1"]["epochs"]) <= 0:
        raise ValueError("stage1.epochs must be positive")
    if float(config["stage1"]["learning_rate"]) <= 0.0:
        raise ValueError("stage1.learning_rate must be positive")
    if float(config["stage1"]["dpp_weight"]) < 0.0:
        raise ValueError("stage1.dpp_weight must be non-negative")
    if float(config["stage1"]["rbf_weight"]) < 0.0:
        raise ValueError("stage1.rbf_weight must be non-negative")
    if float(config["stage1"]["max_grad_norm"]) <= 0.0:
        raise ValueError("stage1.max_grad_norm must be positive")
    experts = int(config["stage1"]["num_experts"])
    probe_k = int(config["kneedle"]["probe_k"])
    min_k = int(config["kneedle"]["min_k"])
    max_k = int(config["kneedle"]["max_k"])
    if experts < 2:
        raise ValueError("GAC Stage 1 requires at least two experts")
    if not (experts <= min_k <= max_k <= probe_k):
        raise ValueError(
            "Kneedle must satisfy num_experts <= min_k <= max_k <= probe_k"
        )
    off = float(config["stage1"]["interaction_off_until"])
    ramp = float(config["stage1"]["interaction_ramp_until"])
    if not (0.0 <= off < ramp <= 1.0):
        raise ValueError("Interaction schedule must satisfy 0 <= off < ramp <= 1")
    jitter = float(config["dpp"]["jitter"])
    maximum_jitter = float(config["dpp"]["max_jitter"])
    if not (0.0 < jitter <= maximum_jitter):
        raise ValueError("DPP jitter must be positive and no larger than max_jitter")
    decay = float(config["rbf"]["bandwidth_ema"])
    if not (0.0 <= decay < 1.0):
        raise ValueError("rbf.bandwidth_ema must be in [0, 1)")
    if float(config["rbf"]["bandwidth_floor"]) <= 0.0:
        raise ValueError("rbf.bandwidth_floor must be positive")
    if int(config["runtime"]["lm_head_chunk_tokens"]) <= 0:
        raise ValueError("runtime.lm_head_chunk_tokens must be positive")


def _planned_window_counts(
    dataset: JsonlRecordDataset,
    sampler: DistributedSampler,
    epochs: int,
    micro_batch: int,
    accumulation_steps: int,
) -> list[tuple[int, int]]:
    """Count SFT tokens and DPP samples before a window's transformer VJPs.

    A single full-phase VJP needs the global SFT-token/DPP-sample ratio. The
    sampler order is deterministic, so these exact denominators can be planned
    without holding any transformer graph across microbatches.
    """
    per_record: list[tuple[int, int]] = []
    for index in range(len(dataset)):
        record = dataset[index]
        sft_tokens = sum(label != -100 for label in record.labels[1:])
        reasoning_steps = [
            step
            for label, region, step in zip(
                record.labels[1:],
                record.region_ids[1:],
                record.step_ids[1:],
                strict=True,
            )
            if label != -100 and region == int(TokenRegion.REASONING)
        ]
        if any(step < 0 for step in reasoning_steps):
            raise ValueError(
                f"Prepared sample {record.sample_id} has an invalid reasoning step"
            )
        per_record.append((sft_tokens, int(bool(reasoning_steps))))
    micro_counts: list[tuple[int, int]] = []
    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        indices = list(iter(sampler))
        for start in range(0, len(indices), micro_batch):
            group = [
                per_record[index] for index in indices[start : start + micro_batch]
            ]
            micro_counts.append(
                (sum(value[0] for value in group), sum(value[1] for value in group))
            )
    return [
        (
            sum(value[0] for value in micro_counts[start : start + accumulation_steps]),
            sum(value[1] for value in micro_counts[start : start + accumulation_steps]),
        )
        for start in range(0, len(micro_counts), accumulation_steps)
    ]


def _save_training_checkpoint(
    path: Path,
    model: torch.nn.Module,
    adapter_names: list[str],
    optimizers: list[torch.optim.Optimizer],
    schedulers: list[torch.optim.lr_scheduler.LRScheduler],
    bandwidth: BandwidthEMA,
    global_step: int,
    epoch: int,
    batch_in_epoch: int,
    run_fingerprint: str,
    base_seed: int,
    world_size: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    value = {
        "adapter_states": {
            name: extract_adapter_state(model, name) for name in adapter_names
        },
        "optimizers": [optimizer.state_dict() for optimizer in optimizers],
        "schedulers": [scheduler.state_dict() for scheduler in schedulers],
        "bandwidth": bandwidth.state_dict(),
        "global_step": global_step,
        "epoch": epoch,
        "batch_in_epoch": batch_in_epoch,
        "run_fingerprint": run_fingerprint,
        "sampler_state": {
            "seed": base_seed,
            "epoch": epoch,
            "next_batch_in_epoch": batch_in_epoch,
            "world_size": world_size,
        },
        "expert_dropout_rng": {
            "scheme": "sha256(base_seed,stage1,global_step,epoch_batch_rank,expert)",
            "base_seed": base_seed,
        },
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
    }
    torch.save(value, temporary)
    temporary.replace(path)


def _load_training_checkpoint(
    path: Path,
    model: torch.nn.Module,
    adapter_names: list[str],
    optimizers: list[torch.optim.Optimizer],
    schedulers: list[torch.optim.lr_scheduler.LRScheduler],
    bandwidth: BandwidthEMA,
    expected_run_fingerprint: str,
) -> tuple[int, int, int]:
    value = torch.load(path, map_location="cpu", weights_only=False)
    if value["run_fingerprint"] != expected_run_fingerprint:
        raise RuntimeError("Refusing to resume Stage 1 from a different configuration")
    for name in adapter_names:
        load_adapter_state(model, name, value["adapter_states"][name])
    for optimizer, state in zip(optimizers, value["optimizers"], strict=True):
        optimizer.load_state_dict(state)
    for scheduler, state in zip(schedulers, value["schedulers"], strict=True):
        scheduler.load_state_dict(state)
    bandwidth.load_state_dict(value["bandwidth"])
    random.setstate(value["python_rng"])
    np.random.set_state(value["numpy_rng"])
    torch.set_rng_state(value["torch_rng"])
    if torch.cuda.is_available() and value["cuda_rng"] is not None:
        torch.cuda.set_rng_state_all(value["cuda_rng"])
    return int(value["global_step"]), int(value["epoch"]), int(value["batch_in_epoch"])


def train_stage1(
    config: dict[str, Any], distributed: DistributedContext
) -> dict[str, Any]:
    _validate_stage1_config(config)
    output_dir = Path(config["paths"]["output"])
    output_dir.mkdir(parents=True, exist_ok=True)
    config_snapshot = output_dir / "config.yaml"
    if distributed.is_main:
        write_config_snapshot(config_snapshot, _stable_config(config))
    barrier()
    prepared_manifest = read_json(Path(config["paths"]["prepared"]) / "manifest.json")
    require_file_sha256(
        config["paths"]["prepared"], prepared_manifest, "data_file", "data_file_sha256"
    )
    require_file_sha256(
        config["paths"]["prepared"],
        prepared_manifest,
        "config_file",
        "config_file_sha256",
    )
    tokenizer = load_tokenizer(config["model"])
    if tokenizer_fingerprint(tokenizer) != prepared_manifest["tokenizer_fingerprint"]:
        raise RuntimeError("Stage-1 tokenizer does not match the prepared dataset")
    prepared_model = prepared_manifest["config"]["model"]
    require_same_model_source(config["model"], prepared_model, "Stage 1/preprocessing")
    dataset = JsonlRecordDataset(config["paths"]["prepared"])
    if len(dataset) % distributed.world_size != 0:
        raise ValueError(
            "The prepared dataset size must be divisible by world_size so the "
            "distributed sampler never pads with duplicate samples"
        )
    # Use the epoch-addressable sampler even on one GPU. Unlike RandomSampler,
    # this recreates the exact permutation before skipping batches on resume.
    sampler = DistributedSampler(
        dataset,
        num_replicas=distributed.world_size,
        rank=distributed.rank,
        shuffle=True,
        seed=int(config["seed"]),
        drop_last=False,
    )
    micro_batch = int(config["stage1"]["micro_batch_size"])
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
    requested_global_batch = int(config["stage1"]["global_batch_size"])
    if requested_global_batch <= 0:
        raise ValueError("global_batch_size must be positive")
    configured_accumulation = config["stage1"].get("gradient_accumulation_steps")
    if configured_accumulation is None:
        divisor = micro_batch * distributed.world_size
        if requested_global_batch % divisor != 0:
            raise ValueError(
                "global_batch_size must be divisible by micro_batch_size * world_size"
            )
        accumulation_steps = requested_global_batch // divisor
    else:
        accumulation_steps = int(configured_accumulation)
        if accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        actual = accumulation_steps * micro_batch * distributed.world_size
        if actual != requested_global_batch:
            raise ValueError(
                f"Configured accumulation produces global batch {actual}, expected {requested_global_batch}"
            )
    epochs = int(config["stage1"]["epochs"])
    total_steps = math.ceil(epochs * len(dataloader) / accumulation_steps)
    planned_counts = _planned_window_counts(
        dataset, sampler, epochs, micro_batch, accumulation_steps
    )
    if len(planned_counts) != total_steps:
        raise RuntimeError("Planned optimizer windows do not match the dataloader")

    model, adapter_names = create_multi_adapter_model(
        config["model"],
        config["lora"],
        int(config["stage1"]["num_experts"]),
        distributed.device,
        int(config["seed"]),
    )
    _, stage1_head = decoder_and_lm_head(model)
    if int(config["kneedle"]["probe_k"]) > int(stage1_head.weight.shape[0]) - 1:
        raise ValueError(
            "kneedle.probe_k cannot exceed vocabulary_size - 1 non-target tokens"
        )
    groups = adapter_parameter_groups(model, adapter_names)
    parameter_lists = [list(group.values()) for group in groups]
    optimizers_and_schedulers = [
        _optimizer_and_scheduler(parameters, config, total_steps)
        for parameters in parameter_lists
    ]
    optimizers = [item[0] for item in optimizers_and_schedulers]
    schedulers = [item[1] for item in optimizers_and_schedulers]
    bandwidth = BandwidthEMA(
        decay=float(config["rbf"]["bandwidth_ema"]),
        floor=float(config["rbf"]["bandwidth_floor"]),
    )
    public_config = _stable_config(config)
    config_hash = fingerprint(_run_compatible_config(config))
    run_hash = fingerprint(
        {
            "config_fingerprint": config_hash,
            "prepared_manifest_fingerprint": fingerprint(prepared_manifest),
            "world_size": distributed.world_size,
        }
    )
    global_step = start_epoch = start_batch = 0
    resume = config["stage1"].get("resume_from")
    if resume:
        global_step, start_epoch, start_batch = _load_training_checkpoint(
            Path(resume),
            model,
            adapter_names,
            optimizers,
            schedulers,
            bandwidth,
            run_hash,
        )
        LOGGER.info(
            "Resumed Stage 1 at step=%d epoch=%d batch=%d",
            global_step,
            start_epoch,
            start_batch,
        )
        if start_epoch >= epochs:
            manifest_path = output_dir / "manifest.json"
            if not manifest_path.is_file():
                raise RuntimeError("Completed Stage-1 checkpoint has no final manifest")
            barrier()
            return read_json(manifest_path)
    metrics = JsonlLogger(
        output_dir / "metrics.jsonl",
        enabled=distributed.is_main,
        truncate=not bool(resume),
    )
    chunk_tokens = int(config["runtime"]["lm_head_chunk_tokens"])
    forward_mode = config["stage1"].get("forward_mode", "one_pass")
    scaling = float(config["lora"]["alpha"]) / float(config["lora"]["rank"])

    # Accumulation deliberately crosses epoch boundaries. For the canonical
    # 1,000 x 3 run this produces ceil(3,000 / 32) = 94 optimizer updates,
    # instead of flushing three undersized batches at each epoch boundary.
    sft_buffers = [zeros_like_parameters(parameters) for parameters in parameter_lists]
    dpp_buffers: list[list[torch.Tensor]] | None = None
    full_window_counts: tuple[int, int] | None = None
    sft_token_count = 0
    dpp_sample_count = 0
    accumulated_microbatches = 0
    accumulated_sft_loss = 0.0
    accumulated_dpp_loss = 0.0
    selected_k_sum = cap_hits = saturation_hits = 0.0
    support_selection_count = cholesky_fallbacks = 0

    for epoch in range(start_epoch, epochs):
        sampler.set_epoch(epoch)
        for batch_index, raw_batch in enumerate(dataloader):
            if epoch == start_epoch and batch_index < start_batch:
                continue
            batch = _batch_to_device(raw_batch, distributed.device)
            # The same stream id preserves exact two-pass dropout replay, while
            # distinct microbatches never reuse a mask.
            rng_stream = (
                epoch * len(dataloader) + batch_index
            ) * distributed.world_size + distributed.rank
            progress = global_step / max(total_steps - 1, 1)
            gamma = interaction_scale(
                progress,
                float(config["stage1"]["interaction_off_until"]),
                float(config["stage1"]["interaction_ramp_until"]),
            )
            phase = "sft" if gamma == 0.0 else "full" if gamma == 1.0 else "ramp"
            combined_dpp_scale: float | None = None
            if phase == "full":
                if full_window_counts is None:
                    local_counts = torch.tensor(
                        planned_counts[global_step],
                        device=distributed.device,
                        dtype=torch.float64,
                    )
                    all_reduce_tensor(local_counts)
                    full_window_counts = (
                        int(local_counts[0].item()),
                        int(local_counts[1].item()),
                    )
                if full_window_counts[0] <= 0:
                    raise ValueError("A full-phase optimizer window has no SFT tokens")
                combined_dpp_scale = (
                    float(config["stage1"]["dpp_weight"])
                    * full_window_counts[0]
                    / full_window_counts[1]
                    if full_window_counts[1] > 0
                    else 0.0
                )
            elif phase == "ramp" and dpp_buffers is None:
                dpp_buffers = [
                    zeros_like_parameters(parameters) for parameters in parameter_lists
                ]
            if phase == "sft":
                probe = _empty_probe(distributed.device, len(adapter_names))
                expert_results = (
                    sft_only_expert_gradients(
                        model,
                        adapter_name,
                        expert,
                        parameters,
                        batch,
                        global_step,
                        rng_stream,
                        int(config["seed"]),
                        chunk_tokens,
                        distributed.device,
                    )
                    for expert, (adapter_name, parameters) in enumerate(
                        zip(adapter_names, parameter_lists, strict=True)
                    )
                )
            elif forward_mode == "one_pass":
                probe, expert_results = one_pass_expert_gradients(
                    model,
                    adapter_names,
                    parameter_lists,
                    batch,
                    global_step,
                    rng_stream,
                    int(config["seed"]),
                    config,
                    distributed.device,
                    combined_dpp_scale=combined_dpp_scale,
                )
            else:
                probe = probe_stage1_dpp(
                    model,
                    adapter_names,
                    batch,
                    global_step,
                    rng_stream,
                    int(config["seed"]),
                    config,
                    distributed.device,
                )
                expert_results = (
                    replay_expert_gradients(
                        model,
                        adapter_name,
                        expert,
                        parameters,
                        batch,
                        probe,
                        global_step,
                        rng_stream,
                        int(config["seed"]),
                        chunk_tokens,
                        distributed.device,
                        combined_dpp_scale=combined_dpp_scale,
                    )
                    for expert, (adapter_name, parameters) in enumerate(
                        zip(adapter_names, parameter_lists, strict=True)
                    )
                )
            selected_k_sum += probe.mean_selected_k * probe.selection_count
            cap_hits += probe.cap_rate * probe.selection_count
            saturation_hits += probe.probe_saturation_rate * probe.selection_count
            support_selection_count += probe.selection_count
            cholesky_fallbacks += probe.dpp_metrics.cholesky_fallbacks
            for expert, (
                sft_gradient,
                dpp_gradient,
                sft_loss,
                token_count,
            ) in enumerate(expert_results):
                add_gradients_(sft_buffers[expert], sft_gradient)
                if phase == "ramp":
                    assert dpp_buffers is not None
                    add_gradients_(dpp_buffers[expert], dpp_gradient)
                accumulated_sft_loss += sft_loss
                sft_token_count += token_count if expert == 0 else 0
            accumulated_dpp_loss += probe.dpp_loss_sum
            dpp_sample_count += probe.dpp_sample_count
            accumulated_microbatches += 1
            final_microbatch = epoch + 1 == epochs and batch_index + 1 == len(
                dataloader
            )
            boundary = (
                accumulated_microbatches == accumulation_steps or final_microbatch
            )
            if not boundary:
                continue

            all_reduce_grad_lists(sft_buffers)
            if phase == "ramp":
                assert dpp_buffers is not None
                all_reduce_grad_lists(dpp_buffers)
            counts = torch.tensor(
                [sft_token_count, dpp_sample_count],
                device=distributed.device,
                dtype=torch.float64,
            )
            losses = torch.tensor(
                [accumulated_sft_loss, accumulated_dpp_loss],
                device=distributed.device,
                dtype=torch.float64,
            )
            all_reduce_tensor(counts)
            all_reduce_tensor(losses)
            probe_statistics = torch.tensor(
                [
                    selected_k_sum,
                    cap_hits,
                    saturation_hits,
                    support_selection_count,
                    cholesky_fallbacks,
                ],
                device=distributed.device,
                dtype=torch.float64,
            )
            all_reduce_tensor(probe_statistics)
            if (
                phase == "full"
                and (int(counts[0].item()), int(counts[1].item())) != full_window_counts
            ):
                raise RuntimeError(
                    "Precomputed full-phase loss denominators do not match"
                )
            for values in sft_buffers:
                divide_gradients_(values, float(counts[0].item()))
            if phase == "ramp":
                assert dpp_buffers is not None
                for values in dpp_buffers:
                    divide_gradients_(values, float(counts[1].item()))

            distances: torch.Tensor | None = None
            current_bandwidth: float | None = None
            if phase == "sft":
                final_gradients = sft_buffers
                task_norms = tuple(
                    float(vector_norm(values).item()) for values in sft_buffers
                )
                diagnostics = GACDiagnostics(0.0, task_norms, (), (), (), task_norms)
            else:
                set_all_adapters_trainable(model, adapter_names)
                distances_for_step = effective_update_distances(groups, scaling)
                current_bandwidth = bandwidth.update(distances_for_step)
                repulsion, kernel, distances = repulsion_updates(
                    groups, scaling, current_bandwidth, distances=distances_for_step
                )
                final_gradients, diagnostics = stable_gac_gradients(
                    sft_buffers,
                    dpp_buffers if phase == "ramp" else None,
                    repulsion,
                    kernel,
                    gamma,
                    float(config["stage1"]["dpp_weight"]),
                    float(config["stage1"]["rbf_weight"]),
                )
            preclip_norms = [
                global_clip_grad_list_(values, float(config["stage1"]["max_grad_norm"]))
                for values in final_gradients
            ]
            for parameters, gradients, optimizer, scheduler in zip(
                parameter_lists, final_gradients, optimizers, schedulers, strict=True
            ):
                assign_gradients(parameters, gradients)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            global_step += 1

            if (
                distributed.is_main
                and global_step % int(config["stage1"]["log_every_steps"]) == 0
            ):
                off_diagonal = (
                    distances[
                        torch.triu_indices(
                            len(adapter_names), len(adapter_names), 1
                        ).unbind()
                    ]
                    if distances is not None
                    else None
                )
                metrics.log(
                    "stage1_step",
                    step=global_step,
                    epoch=epoch,
                    forward_mode=forward_mode,
                    interaction_phase=phase,
                    task_gradient_kind=(
                        "sft_plus_weighted_dpp" if phase == "full" else "sft"
                    ),
                    probe_computed=phase != "sft",
                    rbf_computed=phase != "sft",
                    sft_nll=float(
                        losses[0].item()
                        / max(counts[0].item() * len(adapter_names), 1.0)
                    ),
                    dpp_loss=float(losses[1].item() / max(counts[1].item(), 1.0)),
                    interaction=gamma,
                    learning_rate=optimizers[0].param_groups[0]["lr"],
                    bandwidth=current_bandwidth,
                    mean_delta_w_distance=(
                        float(off_diagonal.mean().item())
                        if off_diagonal is not None
                        else None
                    ),
                    min_delta_w_distance=(
                        float(off_diagonal.min().item())
                        if off_diagonal is not None
                        else None
                    ),
                    mean_selected_k=float(
                        probe_statistics[0].item()
                        / max(probe_statistics[3].item(), 1.0)
                    ),
                    k_cap_rate=float(
                        probe_statistics[1].item()
                        / max(probe_statistics[3].item(), 1.0)
                    ),
                    probe_saturation_rate=float(
                        probe_statistics[2].item()
                        / max(probe_statistics[3].item(), 1.0)
                    ),
                    cholesky_fallbacks=int(probe_statistics[4].item()),
                    task_gradient_norms=diagnostics.task_norms,
                    dpp_gradient_norms=diagnostics.dpp_norms,
                    repulsion_cap_factors=diagnostics.repulsion_cap_factors,
                    preclip_gradient_norms=preclip_norms,
                )
            if (
                distributed.is_main
                and global_step % int(config["stage1"]["checkpoint_every_steps"]) == 0
            ):
                _save_training_checkpoint(
                    output_dir / "checkpoint.pt",
                    model,
                    adapter_names,
                    optimizers,
                    schedulers,
                    bandwidth,
                    global_step,
                    epoch,
                    batch_index + 1,
                    run_hash,
                    int(config["seed"]),
                    distributed.world_size,
                )

            sft_buffers = [
                zeros_like_parameters(parameters) for parameters in parameter_lists
            ]
            dpp_buffers = None
            full_window_counts = None
            sft_token_count = dpp_sample_count = accumulated_microbatches = 0
            accumulated_sft_loss = accumulated_dpp_loss = 0.0
            selected_k_sum = cap_hits = saturation_hits = 0.0
            support_selection_count = cholesky_fallbacks = 0
        start_batch = 0
        barrier()

    if distributed.is_main:
        final_dir = output_dir / "final"
        bundle_path = save_adapter_bundle(model, adapter_names, final_dir)
        _save_training_checkpoint(
            output_dir / "checkpoint.pt",
            model,
            adapter_names,
            optimizers,
            schedulers,
            bandwidth,
            global_step,
            epochs,
            0,
            run_hash,
            int(config["seed"]),
            distributed.world_size,
        )
        manifest = {
            "schema_version": 1,
            "artifact": "stage1_checkpoint",
            "global_step": global_step,
            "training_checkpoint": "checkpoint.pt",
            "training_checkpoint_sha256": file_sha256(output_dir / "checkpoint.pt"),
            "adapter_names": adapter_names,
            "adapter_bundle": str(bundle_path.relative_to(output_dir)),
            "adapter_bundle_sha256": file_sha256(bundle_path),
            "prepared_manifest_fingerprint": fingerprint(prepared_manifest),
            "tokenizer_fingerprint": prepared_manifest["tokenizer_fingerprint"],
            "bandwidth": bandwidth.state_dict(),
            "config": public_config,
            "config_fingerprint": config_hash,
            "config_file": config_snapshot.name,
            "config_file_sha256": file_sha256(config_snapshot),
            "run_fingerprint": run_hash,
            "metrics_file": "metrics.jsonl",
            "metrics_file_sha256": file_sha256(output_dir / "metrics.jsonl"),
            "runtime": runtime_metadata(config["_project_root"]),
        }
        write_json(output_dir / "manifest.json", manifest)
    barrier()
    return read_json(output_dir / "manifest.json")
