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
    chunked_probability_statistics,
    cross_entropy_hidden_gradient,
    decoder_and_lm_head,
    forward_hidden,
    gather_hidden_positions,
    gather_support_log_probabilities,
    support_logprob_vjp_hidden_gradient,
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
    zeros_like_parameters,
)
from .dpp import DPPMetrics, normalized_support_features, step_dpp_loss
from .dropout import step_level_token_weights
from .gac_gradient import apply_grassmann_force_, phase1_data_gradients
from .grassmann import grassmann_repulsion_updates
from .kneedle import council_kneedle_candidates, pad_candidate_support
from .rbf import BandwidthEMA

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
    selection_count: int
    valid_candidate_count: int


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
        dpp_metrics=DPPMetrics(0, 0, 0, 1.0e-5),
        mean_selected_k=0.0,
        selection_count=0,
        valid_candidate_count=0,
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
) -> ProbeResult:
    views = shifted_token_views(batch)
    token_count = int(views["reasoning_targets"].numel())
    if token_count == 0:
        return _empty_probe(device, len(adapter_names))
    _, head = decoder_and_lm_head(model)
    probe_hidden_device = torch.device(
        config["runtime"].get("probe_hidden_device", "cpu")
    )
    hidden_by_expert: list[torch.Tensor] = []
    chunk_tokens = int(config["runtime"]["lm_head_chunk_tokens"])

    model.train()
    for expert, adapter_name in enumerate(adapter_names):
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
        hidden_by_expert.append(selected_hidden)

    candidates: list[torch.Tensor] = []
    selected_k: list[torch.Tensor] = []

    def select_council_candidates(
        start: int,
        end: int,
        probabilities: list[torch.Tensor],
        _log_probabilities: list[torch.Tensor],
    ) -> None:
        council = torch.stack(probabilities).mean(dim=0)
        current, elbows = council_kneedle_candidates(
            council,
            views["reasoning_targets"][start:end].to(council.device),
        )
        candidates.extend(current)
        selected_k.append(elbows)

    chunked_probability_statistics(
        hidden_by_expert,
        head,
        views["reasoning_targets"],
        int(config["runtime"]["council_chunk_tokens"]),
        1.0,
        select_council_candidates,
    )
    support_ids_cpu, support_mask_cpu = pad_candidate_support(candidates)
    support_ids = support_ids_cpu.to(device, non_blocking=True)
    support_mask = support_mask_cpu.to(device, non_blocking=True)
    eligible = support_mask.any(dim=-1)
    if eligible.any():
        support_log_probabilities = torch.stack(
            [
                gather_support_log_probabilities(
                    hidden, head, support_ids_cpu, chunk_tokens, output_device=device
                )
                for hidden in hidden_by_expert
            ],
            dim=0,
        ).detach().requires_grad_(True)
        features = normalized_support_features(support_log_probabilities, support_mask)
        loss, metrics = step_dpp_loss(
            features[:, eligible],
            views["reasoning_batch_indices"][eligible],
            views["reasoning_step_ids"][eligible],
            jitter=float(config["dpp"]["jitter"]),
            maximum_jitter=float(config["dpp"]["max_jitter"]),
            reduction="sum",
        )
        gradients = torch.autograd.grad(loss, support_log_probabilities)[0].detach()
    else:
        loss = torch.zeros((), device=device)
        metrics = DPPMetrics(0, 0, 0, float(config["dpp"]["jitter"]))
        gradients = torch.empty(
            (len(adapter_names), token_count, 0), device=device, dtype=torch.float32
        )
    return ProbeResult(
        support_ids=support_ids,
        support_mask=support_mask,
        dpp_logit_gradients=gradients,
        dpp_loss_sum=float(loss.detach().item()),
        dpp_sample_count=metrics.samples,
        dpp_metrics=metrics,
        mean_selected_k=float(torch.cat(selected_k).float().mean().item()),
        selection_count=token_count,
        valid_candidate_count=int(eligible.sum().item()),
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
    drop_probability: float,
    chunk_tokens: int,
    device: torch.device,
) -> tuple[list[torch.Tensor], list[torch.Tensor], float, int, int]:
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
    token_weights, sft_count, dropped_steps = step_level_token_weights(
        batch, expert_index, base_seed, global_step, rng_stream, drop_probability
    )
    # `sft_count` is the number of contributing segments: retained reasoning
    # steps plus always-on answer/format blocks. Dropped steps are excluded.
    sft_hidden_gradient, sft_loss_sum, _ = cross_entropy_hidden_gradient(
        response_hidden,
        head,
        views["response_targets"],
        chunk_tokens,
        token_weights=token_weights,
    )
    dpp_full_hidden_gradient = torch.zeros_like(response_hidden)
    if probe.support_ids.numel() > 0:
        response_regions = views["regions"][views["valid"]]
        response_reasoning = response_regions.eq(int(TokenRegion.REASONING))
        reasoning_hidden = response_hidden[response_reasoning]
        dpp_hidden_gradient = support_logprob_vjp_hidden_gradient(
            reasoning_hidden,
            head,
            probe.support_ids,
            probe.dpp_logit_gradients[expert_index],
            chunk_tokens,
        )
        dpp_full_hidden_gradient[response_reasoning] = dpp_hidden_gradient

    sft_parameter_gradients = torch.autograd.grad(
        response_hidden,
        parameters,
        grad_outputs=sft_hidden_gradient,
        retain_graph=True,
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
        dropped_steps,
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


def _validate_stage1_config(config: dict[str, Any]) -> None:
    if str(config["optimizer"].get("name", "")).lower() != "adamw":
        raise ValueError("Stage 1 currently implements optimizer.name: adamw")
    if str(config["scheduler"].get("name", "")).lower() != "cosine":
        raise ValueError("Stage 1 currently implements scheduler.name: cosine")
    if int(config["stage1"]["epochs"]) <= 0:
        raise ValueError("stage1.epochs must be positive")
    if float(config["stage1"]["learning_rate"]) <= 0.0:
        raise ValueError("stage1.learning_rate must be positive")
    if float(config["stage1"]["diversity_weight"]) < 0.0:
        raise ValueError("stage1.diversity_weight must be non-negative")
    if float(config["stage1"]["repulsion_weight"]) < 0.0:
        raise ValueError("stage1.repulsion_weight must be non-negative")
    if not 0 <= float(config["stage1"]["step_drop_probability"]) < 1:
        raise ValueError("stage1.step_drop_probability must be in [0, 1)")
    if float(config["stage1"]["max_grad_norm"]) <= 0.0:
        raise ValueError("stage1.max_grad_norm must be positive")
    experts = int(config["stage1"]["num_experts"])
    if experts < 2:
        raise ValueError("Grassmann Stage 1 requires at least two experts")
    jitter = float(config["dpp"]["jitter"])
    maximum_jitter = float(config["dpp"]["max_jitter"])
    if not (0.0 < jitter <= maximum_jitter):
        raise ValueError("DPP jitter must be positive and no larger than max_jitter")
    if float(config["grassmann"]["rank_epsilon"]) <= 0.0:
        raise ValueError("grassmann.rank_epsilon must be positive")
    if not 0 < float(config["grassmann"]["angle_epsilon"]) < 1:
        raise ValueError("grassmann.angle_epsilon must be in (0, 1)")
    if float(config["grassmann"]["bandwidth_floor"]) <= 0.0:
        raise ValueError("grassmann.bandwidth_floor must be positive")
    if int(config["runtime"]["lm_head_chunk_tokens"]) <= 0:
        raise ValueError("runtime.lm_head_chunk_tokens must be positive")
    if int(config["runtime"]["council_chunk_tokens"]) <= 0:
        raise ValueError("runtime.council_chunk_tokens must be positive")
    benchmark_epoch = config["stage1"].get("benchmark_checkpoint_epoch")
    if benchmark_epoch is not None:
        benchmark_epoch = int(benchmark_epoch)
        if not 1 <= benchmark_epoch <= int(config["stage1"]["epochs"]):
            raise ValueError(
                "stage1.benchmark_checkpoint_epoch must be in 1..stage1.epochs"
            )


def _save_training_checkpoint(
    path: Path,
    model: torch.nn.Module,
    adapter_names: list[str],
    optimizers: list[torch.optim.Optimizer],
    schedulers: list[torch.optim.lr_scheduler.LRScheduler],
    bandwidth: BandwidthEMA | None,
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
        "bandwidth": bandwidth.state_dict() if bandwidth is not None else None,
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
        "step_dropout_rng": {
            "scheme": "sha256(base_seed,stage1_step_dropout,global_step,epoch_batch_rank,expert)",
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
    bandwidth: BandwidthEMA | None,
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
    if bandwidth is not None:
        if value.get("bandwidth") is None:
            raise RuntimeError("Checkpoint is missing the legacy bandwidth state")
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
    LOGGER.info(
        "Stage 1 setup: samples=%d epochs=%d micro_batch=%d "
        "accumulation=%d optimizer_steps=%d",
        len(dataset),
        epochs,
        micro_batch,
        accumulation_steps,
        total_steps,
    )
    benchmark_epoch = config["stage1"].get("benchmark_checkpoint_epoch")
    benchmark_step = (
        math.ceil(int(benchmark_epoch) * len(dataset) / requested_global_batch)
        if benchmark_epoch is not None
        else None
    )

    model, adapter_names = create_multi_adapter_model(
        config["model"],
        config["lora"],
        int(config["stage1"]["num_experts"]),
        distributed.device,
        int(config["seed"]),
    )
    groups = adapter_parameter_groups(model, adapter_names)
    parameter_lists = [list(group.values()) for group in groups]
    optimizers_and_schedulers = [
        _optimizer_and_scheduler(parameters, config, total_steps)
        for parameters in parameter_lists
    ]
    optimizers = [item[0] for item in optimizers_and_schedulers]
    schedulers = [item[1] for item in optimizers_and_schedulers]
    bandwidth = None
    public_config = _stable_config(config)
    config_hash = fingerprint(_stable_config(config))
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

    # Accumulation deliberately crosses epoch boundaries. For the canonical
    # 1,000 x 3 run this produces ceil(3,000 / 8) = 375 optimizer updates,
    # instead of flushing three undersized batches at each epoch boundary.
    sft_buffers = [zeros_like_parameters(parameters) for parameters in parameter_lists]
    dpp_buffers = [zeros_like_parameters(parameters) for parameters in parameter_lists]
    sft_segment_counts = [0] * len(adapter_names)
    dpp_sample_count = 0
    accumulated_microbatches = 0
    accumulated_sft_losses = [0.0] * len(adapter_names)
    accumulated_dpp_loss = 0.0
    selected_k_sum = 0.0
    support_selection_count = valid_candidate_count = cholesky_fallbacks = 0
    dropped_step_count = 0

    for epoch in range(start_epoch, epochs):
        sampler.set_epoch(epoch)
        for batch_index, raw_batch in enumerate(dataloader):
            if epoch == start_epoch and batch_index < start_batch:
                continue
            batch = _batch_to_device(raw_batch, distributed.device)
            if epoch == start_epoch and batch_index == start_batch:
                LOGGER.info(
                    "Stage 1 first microbatch: tokens=%d seq_len=%d",
                    int(batch["attention_mask"].sum().item()),
                    int(batch["input_ids"].shape[1]),
                )
            # The same stream id is used for the no-grad probe and replay so dropout
            # masks match exactly, while distinct microbatches never reuse a mask.
            rng_stream = (
                epoch * len(dataloader) + batch_index
            ) * distributed.world_size + distributed.rank
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
            selected_k_sum += probe.mean_selected_k * probe.selection_count
            support_selection_count += probe.selection_count
            valid_candidate_count += probe.valid_candidate_count
            cholesky_fallbacks += probe.dpp_metrics.cholesky_fallbacks
            for expert, (adapter_name, parameters) in enumerate(
                zip(adapter_names, parameter_lists, strict=True)
            ):
                sft_gradient, dpp_gradient, sft_loss, segment_count, dropped_steps = (
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
                        float(config["stage1"]["step_drop_probability"]),
                        chunk_tokens,
                        distributed.device,
                    )
                )
                add_gradients_(sft_buffers[expert], sft_gradient)
                add_gradients_(dpp_buffers[expert], dpp_gradient)
                accumulated_sft_losses[expert] += sft_loss
                sft_segment_counts[expert] += segment_count
                dropped_step_count += dropped_steps
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
            all_reduce_grad_lists(dpp_buffers)
            counts = torch.tensor(
                [*sft_segment_counts, dpp_sample_count],
                device=distributed.device,
                dtype=torch.float64,
            )
            losses = torch.tensor(
                [*accumulated_sft_losses, accumulated_dpp_loss],
                device=distributed.device,
                dtype=torch.float64,
            )
            all_reduce_tensor(counts)
            all_reduce_tensor(losses)
            probe_statistics = torch.tensor(
                [
                    selected_k_sum,
                    support_selection_count,
                    valid_candidate_count,
                    cholesky_fallbacks,
                    dropped_step_count,
                ],
                device=distributed.device,
                dtype=torch.float64,
            )
            all_reduce_tensor(probe_statistics)
            for expert, values in enumerate(sft_buffers):
                divide_gradients_(values, max(float(counts[expert].item()), 1.0))
            for values in dpp_buffers:
                divide_gradients_(values, max(float(counts[-1].item()), 1.0))

            set_all_adapters_trainable(model, adapter_names)
            repulsion, _kernel, distances, current_bandwidth = grassmann_repulsion_updates(
                groups,
                rank_epsilon=float(config["grassmann"]["rank_epsilon"]),
                angle_epsilon=float(config["grassmann"]["angle_epsilon"]),
                bandwidth_floor=float(config["grassmann"]["bandwidth_floor"]),
            )
            final_gradients, diagnostics = phase1_data_gradients(
                sft_buffers,
                dpp_buffers,
                repulsion,
                float(config["stage1"]["diversity_weight"]),
            )
            preclip_norms = [
                global_clip_grad_list_(values, float(config["stage1"]["max_grad_norm"]))
                for values in final_gradients
            ]
            for parameters, gradients, force, optimizer, scheduler in zip(
                parameter_lists, final_gradients, repulsion, optimizers, schedulers, strict=True
            ):
                assign_gradients(parameters, gradients)
                learning_rate = float(optimizer.param_groups[0]["lr"])
                optimizer.step()
                apply_grassmann_force_(
                    parameters,
                    force,
                    learning_rate,
                    float(config["stage1"]["repulsion_weight"]),
                )
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            global_step += 1

            if (
                distributed.is_main
                and global_step % int(config["stage1"]["log_every_steps"]) == 0
            ):
                off_diagonal = distances[
                    torch.triu_indices(
                        len(adapter_names), len(adapter_names), 1
                    ).unbind()
                ]
                metrics.log(
                    "stage1_step",
                    step=global_step,
                    epoch=epoch,
                    sft_nll=float(
                        sum(
                            losses[expert].item() / max(counts[expert].item(), 1.0)
                            for expert in range(len(adapter_names))
                        )
                        / max(len(adapter_names), 1)
                    ),
                    dpp_loss=float(losses[-1].item() / max(counts[-1].item(), 1.0)),
                    dropped_reasoning_steps=int(probe_statistics[4].item()),
                    learning_rate=optimizers[0].param_groups[0]["lr"],
                    grassmann_bandwidth=current_bandwidth,
                    mean_grassmann_distance=float(off_diagonal.sqrt().mean().item()),
                    min_grassmann_distance=float(off_diagonal.sqrt().min().item()),
                    mean_selected_k=float(
                        probe_statistics[0].item()
                        / max(probe_statistics[1].item(), 1.0)
                    ),
                    candidate_valid_rate=float(
                        probe_statistics[2].item()
                        / max(probe_statistics[1].item(), 1.0)
                    ),
                    cholesky_fallbacks=int(probe_statistics[3].item()),
                    task_gradient_norms=diagnostics.task_norms,
                    dpp_gradient_norms=diagnostics.dpp_norms,
                    repulsion_gradient_norms=diagnostics.repulsion_norms,
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
            if distributed.is_main and benchmark_step == global_step:
                benchmark_dir = output_dir / "benchmark"
                LOGGER.info(
                    "Saving Stage 1 benchmark checkpoint at step=%d (~epoch %s)",
                    global_step,
                    benchmark_epoch,
                )
                save_adapter_bundle(model, adapter_names, benchmark_dir)
                _save_training_checkpoint(
                    benchmark_dir / "checkpoint.pt",
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
            dpp_buffers = [
                zeros_like_parameters(parameters) for parameters in parameter_lists
            ]
            sft_segment_counts = [0] * len(adapter_names)
            accumulated_sft_losses = [0.0] * len(adapter_names)
            dpp_sample_count = accumulated_microbatches = 0
            accumulated_dpp_loss = 0.0
            selected_k_sum = 0.0
            support_selection_count = valid_candidate_count = cholesky_fallbacks = 0
            dropped_step_count = 0
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
            "benchmark_checkpoint_epoch": benchmark_epoch,
            "benchmark_step": benchmark_step,
            "benchmark_adapter_bundle": (
                "benchmark/adapter_states.pt"
                if (output_dir / "benchmark" / "adapter_states.pt").is_file()
                else None
            ),
            "expert_files": {
                name: {
                    "adapter_config": f"final/adapters/{name}/adapter_config.json",
                    "adapter_config_sha256": file_sha256(
                        final_dir / "adapters" / name / "adapter_config.json"
                    ),
                    "adapter_weights": f"final/adapters/{name}/adapter_model.safetensors",
                    "adapter_weights_sha256": file_sha256(
                        final_dir / "adapters" / name / "adapter_model.safetensors"
                    ),
                }
                for name in adapter_names
            },
            "prepared_manifest_fingerprint": fingerprint(prepared_manifest),
            "tokenizer_fingerprint": prepared_manifest["tokenizer_fingerprint"],
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
