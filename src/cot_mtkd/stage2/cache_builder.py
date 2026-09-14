from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from ..data.dataset import JsonlRecordDataset, load_jsonl_files
from ..data.schema import PreparedRecord, TokenRegion
from ..models.chunked_head import decoder_and_lm_head, forward_hidden
from ..models.multi_adapter import (
    create_multi_adapter_model,
    load_adapter_bundle,
    load_adapter_state,
    require_same_model_source,
    set_active_adapter,
)
from ..utils.distributed import (
    DistributedContext,
    all_reduce_tensor,
    barrier,
    shard_indices,
)
from ..utils.manifest import (
    file_sha256,
    files_fingerprint,
    fingerprint,
    read_json,
    require_file_sha256,
    runtime_metadata,
    write_config_snapshot,
    write_json,
)
from .cache import SparseCacheWriter, probability_dtype
from .tail_kl import sparse_topk_with_tail, tail_bucket_kl

LOGGER = logging.getLogger(__name__)


def _response_metadata(record: PreparedRecord):
    labels = torch.tensor(record.labels[1:], dtype=torch.long)
    regions = torch.tensor(record.region_ids[1:], dtype=torch.long)
    steps = torch.tensor(record.step_ids[1:], dtype=torch.long)
    valid = labels.ne(-100)
    shifted_positions = valid.nonzero(as_tuple=True)[0]
    return labels[valid], regions[valid], steps[valid], shifted_positions + 1, valid


def _token_teacher_weights(
    regions: torch.Tensor,
    steps: torch.Tensor,
    signal: dict[str, Any],
    expert_count: int,
) -> torch.Tensor:
    reasoning = torch.tensor(signal["teacher_weights"], dtype=torch.float32).reshape(
        -1, expert_count
    )
    answer = torch.tensor(signal["answer_weights"], dtype=torch.float32)
    if answer.shape != (expert_count,):
        raise ValueError("Answer teacher weights have the wrong expert dimension")
    if not torch.isfinite(reasoning).all() or not torch.isfinite(answer).all():
        raise FloatingPointError("Teacher weights contain non-finite values")
    if (reasoning < 0).any() or (answer < 0).any():
        raise ValueError("Teacher weights must be non-negative")
    if reasoning.numel() and not torch.allclose(
        reasoning.sum(dim=-1), torch.ones(reasoning.shape[0]), atol=1.0e-5
    ):
        raise ValueError("Reasoning teacher weights must sum to one")
    if not torch.allclose(answer.sum(), torch.tensor(1.0), atol=1.0e-5):
        raise ValueError("Answer teacher weights must sum to one")
    uniform = torch.full((expert_count,), 1.0 / expert_count)
    rows: list[torch.Tensor] = []
    for region, step in zip(regions.tolist(), steps.tolist(), strict=True):
        if region == int(TokenRegion.ASSISTANT_CONTROL):
            rows.append(uniform)
        elif region in (int(TokenRegion.REASONING), int(TokenRegion.DELIMITER)):
            if step < 0 or step >= reasoning.shape[0]:
                raise ValueError(f"Teacher weights are missing reasoning step {step}")
            rows.append(reasoning[step])
        else:
            rows.append(answer)
    return torch.stack(rows)


def _teacher_hidden(
    model: torch.nn.Module,
    adapter_names: list[str],
    record: PreparedRecord,
    valid: torch.Tensor,
    device: torch.device,
) -> list[torch.Tensor]:
    input_ids = torch.tensor([record.input_ids], device=device, dtype=torch.long)
    attention = torch.ones_like(input_ids)
    hidden: list[torch.Tensor] = []
    model.eval()
    for name in adapter_names:
        set_active_adapter(model, name)
        with torch.no_grad():
            output = forward_hidden(model, input_ids, attention, use_cache=False)
            selected = output.last_hidden_state[:, :-1, :][0, valid.to(device)]
        hidden.append(selected.to("cpu"))
    return hidden


def compile_sample_target(
    model: torch.nn.Module,
    adapter_names: list[str],
    record: PreparedRecord,
    signal: dict[str, Any],
    device: torch.device,
    top_k: int,
    temperature: float,
    chunk_tokens: int,
    storage_dtype: torch.dtype,
    epsilon: float,
    audit_dense_kl: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    _, regions, steps, positions, valid = _response_metadata(record)
    weights = _token_teacher_weights(regions, steps, signal, len(adapter_names))
    hidden_by_expert = _teacher_hidden(model, adapter_names, record, valid, device)
    _, head = decoder_and_lm_head(model)
    head_dtype = next(head.parameters()).dtype
    all_ids, all_probabilities, all_tail = [], [], []
    # [tokens, tail sum, expert-token comparisons, dense KL sum,
    #  bucket KL sum, absolute gap sum, lower-bound violations]
    audit = torch.zeros(7, dtype=torch.float64)
    with torch.no_grad():
        for start in range(0, positions.numel(), chunk_tokens):
            end = min(positions.numel(), start + chunk_tokens)
            mixture = None
            for expert, hidden in enumerate(hidden_by_expert):
                logits = head(hidden[start:end].to(device, dtype=head_dtype)).float()
                probabilities = F.softmax(logits / temperature, dim=-1)
                weighted = probabilities * weights[start:end, expert].to(
                    device
                ).unsqueeze(-1)
                mixture = weighted if mixture is None else mixture + weighted
            assert mixture is not None
            mixture = mixture / mixture.sum(dim=-1, keepdim=True).clamp_min(epsilon)
            ids, probabilities, tail = sparse_topk_with_tail(mixture, top_k)
            cached_probabilities = probabilities.to(storage_dtype).float()
            audit[0] += end - start
            audit[1] += tail.double().sum().cpu()
            if audit_dense_kl:
                mixture_log = mixture.clamp_min(epsilon).log()
                for hidden in hidden_by_expert:
                    logits = head(
                        hidden[start:end].to(device, dtype=head_dtype)
                    ).float()
                    expert_log = F.log_softmax(logits / temperature, dim=-1)
                    dense = float(temperature) ** 2 * (
                        mixture * (mixture_log - expert_log)
                    ).sum(dim=-1).clamp_min(0.0)
                    bucket = tail_bucket_kl(
                        logits,
                        ids,
                        cached_probabilities,
                        tail,
                        temperature,
                        epsilon,
                    )
                    gap = dense - bucket
                    audit[2] += dense.numel()
                    audit[3] += dense.double().sum().cpu()
                    audit[4] += bucket.double().sum().cpu()
                    audit[5] += gap.abs().double().sum().cpu()
                    audit[6] += gap.lt(-1.0e-5).sum().double().cpu()
            all_ids.append(ids.cpu())
            all_probabilities.append(probabilities.cpu())
            all_tail.append(tail.cpu())
    return (
        positions,
        torch.cat(all_ids),
        torch.cat(all_probabilities),
        torch.cat(all_tail),
        audit,
    )


def build_teacher_cache(
    config: dict[str, Any], distributed: DistributedContext
) -> dict[str, Any]:
    if int(config["cache"]["top_k"]) <= 0:
        raise ValueError("cache.top_k must be positive")
    if float(config["cache"]["temperature"]) <= 0.0:
        raise ValueError("cache.temperature must be positive")
    epsilon = float(config["cache"]["clamp_epsilon"])
    if not (0.0 < epsilon < 1.0):
        raise ValueError("cache.clamp_epsilon must be in (0, 1)")
    if int(config["cache"]["tokens_per_shard"]) <= 0:
        raise ValueError("cache.tokens_per_shard must be positive")
    if int(config["runtime"]["lm_head_chunk_tokens"]) <= 0:
        raise ValueError("runtime.lm_head_chunk_tokens must be positive")
    if int(config["cache"].get("audit_samples", 0)) < 0:
        raise ValueError("cache.audit_samples must be non-negative")
    prepared_dir = Path(config["paths"]["prepared"])
    stage1_dir = Path(config["paths"]["stage1"])
    supervision_dir = Path(config["paths"]["supervision"])
    output_dir = Path(config["paths"]["cache"])
    output_dir.mkdir(parents=True, exist_ok=True)
    public_config = {
        key: value for key, value in config.items() if not key.startswith("_")
    }
    config_snapshot = output_dir / "config.yaml"
    if distributed.is_main:
        write_config_snapshot(config_snapshot, public_config)
    barrier()
    prepared_manifest = read_json(prepared_dir / "manifest.json")
    stage1_manifest = read_json(stage1_dir / "manifest.json")
    supervision_manifest = read_json(supervision_dir / "manifest.json")
    require_file_sha256(
        prepared_dir, prepared_manifest, "data_file", "data_file_sha256"
    )
    require_file_sha256(
        prepared_dir, prepared_manifest, "config_file", "config_file_sha256"
    )
    require_file_sha256(
        stage1_dir, stage1_manifest, "adapter_bundle", "adapter_bundle_sha256"
    )
    require_file_sha256(
        stage1_dir, stage1_manifest, "config_file", "config_file_sha256"
    )
    require_file_sha256(
        supervision_dir,
        supervision_manifest,
        "config_file",
        "config_file_sha256",
    )
    signal_paths = [supervision_dir / name for name in supervision_manifest["shards"]]
    if files_fingerprint(signal_paths) != supervision_manifest["signals_fingerprint"]:
        raise RuntimeError("Supervision shard content does not match its manifest")
    if supervision_manifest["prepared_manifest_fingerprint"] != fingerprint(
        prepared_manifest
    ):
        raise RuntimeError("Supervision/prepared dataset mismatch")
    if supervision_manifest["stage1_manifest_fingerprint"] != fingerprint(
        stage1_manifest
    ):
        raise RuntimeError("Supervision/Stage-1 checkpoint mismatch")
    adapter_names = list(stage1_manifest["adapter_names"])
    stage1_config = stage1_manifest["config"]
    require_same_model_source(config["model"], stage1_config["model"], "Cache/Stage 1")
    model, created_names = create_multi_adapter_model(
        config["model"],
        stage1_config["lora"],
        len(adapter_names),
        distributed.device,
        int(stage1_config["seed"]),
    )
    if created_names != adapter_names:
        raise RuntimeError("Adapter-name mismatch while compiling sparse targets")
    bundle = load_adapter_bundle(stage1_dir / stage1_manifest["adapter_bundle"])
    for name in adapter_names:
        load_adapter_state(model, name, bundle[name])
    _, head = decoder_and_lm_head(model)
    if int(config["cache"]["top_k"]) > int(head.weight.shape[0]):
        raise ValueError("cache.top_k cannot exceed the model vocabulary size")
    signals = load_jsonl_files(
        supervision_dir / name for name in supervision_manifest["shards"]
    )
    dataset = JsonlRecordDataset(prepared_dir)
    cache_config = config["cache"]
    storage_dtype = probability_dtype(str(cache_config["probability_dtype"]))
    writer = SparseCacheWriter(
        output_dir,
        distributed.rank,
        int(cache_config["tokens_per_shard"]),
        storage_dtype,
        distributed.world_size,
    )
    processed = 0
    audit_samples = int(cache_config.get("audit_samples", 0))
    audit_totals = torch.zeros(7, device=distributed.device, dtype=torch.float64)
    audited_records = torch.zeros((), device=distributed.device, dtype=torch.int64)
    for index in tqdm(
        shard_indices(len(dataset), distributed.rank, distributed.world_size),
        desc=f"Teacher cache rank {distributed.rank}",
    ):
        record = dataset[index]
        positions, top_ids, top_probabilities, tail, sample_audit = (
            compile_sample_target(
                model,
                adapter_names,
                record,
                signals[record.sample_id],
                distributed.device,
                int(cache_config["top_k"]),
                float(cache_config["temperature"]),
                int(config["runtime"]["lm_head_chunk_tokens"]),
                storage_dtype,
                float(cache_config["clamp_epsilon"]),
                audit_dense_kl=index < audit_samples,
            )
        )
        audit_totals += sample_audit.to(distributed.device)
        audited_records += int(index < audit_samples)
        writer.add(record.sample_id, positions, top_ids, top_probabilities, tail)
        processed += 1
    writer.close()
    all_reduce_tensor(audit_totals)
    all_reduce_tensor(audited_records)
    barrier()
    if distributed.is_main:
        combined: dict[str, Any] = {}
        index_files = [
            output_dir / f"index-rank{rank:05d}-of{distributed.world_size:05d}.json"
            for rank in range(distributed.world_size)
        ]
        for path in index_files:
            with path.open("r", encoding="utf-8") as handle:
                shard_index = json.load(handle)
            duplicate = set(combined).intersection(shard_index)
            if duplicate:
                raise RuntimeError(
                    f"Duplicate samples across cache ranks: {sorted(duplicate)}"
                )
            combined.update(shard_index)
        if len(combined) != int(prepared_manifest["records"]):
            raise RuntimeError(
                "Sparse teacher cache does not cover every prepared training sample"
            )
        write_json(output_dir / "index.json", combined)
        cache_paths = [
            output_dir / name
            for name in sorted({item["file"] for item in combined.values()})
        ]
        token_count = max(float(audit_totals[0].item()), 1.0)
        comparison_count = max(float(audit_totals[2].item()), 1.0)
        manifest = {
            "schema_version": 1,
            "artifact": "sparse_teacher_cache",
            "records": len(combined),
            "top_k": int(cache_config["top_k"]),
            "temperature": float(cache_config["temperature"]),
            "probability_dtype": str(cache_config["probability_dtype"]),
            "shards": [path.name for path in cache_paths],
            "cache_files_fingerprint": files_fingerprint(cache_paths),
            "index_file": "index.json",
            "index_file_sha256": file_sha256(output_dir / "index.json"),
            "teacher_weighting_fingerprint": supervision_manifest[
                "signals_fingerprint"
            ],
            "cache_statistics": {
                "mean_retained_mass": 1.0 - float(audit_totals[1].item()) / token_count,
                "mean_tail_mass": float(audit_totals[1].item()) / token_count,
            },
            "dense_sparse_audit": {
                "selection": "first_global_indices",
                "requested_samples": audit_samples,
                "audited_samples": int(audited_records.item()),
                "expert_token_comparisons": int(audit_totals[2].item()),
                "mean_full_kl": float(audit_totals[3].item()) / comparison_count,
                "mean_tail_bucket_kl": float(audit_totals[4].item()) / comparison_count,
                "mean_absolute_gap": float(audit_totals[5].item()) / comparison_count,
                "lower_bound_violations_over_tolerance": int(audit_totals[6].item()),
            },
            "prepared_manifest_fingerprint": fingerprint(prepared_manifest),
            "stage1_manifest_fingerprint": fingerprint(stage1_manifest),
            "supervision_manifest_fingerprint": fingerprint(supervision_manifest),
            "config": public_config,
            "config_fingerprint": fingerprint(public_config),
            "config_file": config_snapshot.name,
            "config_file_sha256": file_sha256(config_snapshot),
            "runtime": runtime_metadata(config["_project_root"]),
        }
        write_json(output_dir / "manifest.json", manifest)
        LOGGER.info("Compiled sparse teacher cache for %d samples", len(combined))
    barrier()
    return read_json(output_dir / "manifest.json")
