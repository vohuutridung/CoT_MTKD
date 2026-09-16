from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm

from ..data.dataset import JsonlRecordDataset
from ..data.prepare import tokenizer_fingerprint
from ..models.multi_adapter import (
    create_multi_adapter_model,
    load_adapter_bundle,
    load_adapter_state,
    load_tokenizer,
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
from .group_importance import group_step_importance
from .medoid import functional_medoid
from .pag import score_reference_solution_pag
from .predictive import score_predictive_signals
from .teacher_features import (
    answer_teacher_weights,
    reasoning_teacher_weights,
    zscore_across_experts,
)

LOGGER = logging.getLogger(__name__)


def _as_list(value: torch.Tensor) -> list[Any]:
    return value.detach().cpu().tolist()


def build_supervision(
    config: dict[str, Any], distributed: DistributedContext
) -> dict[str, Any]:
    signal_config = config["signals"]
    if not (0.0 <= float(signal_config["uniform_teacher_mass"]) <= 1.0):
        raise ValueError("uniform_teacher_mass must be in [0, 1]")
    if float(signal_config["temperature_features"]) <= 0.0:
        raise ValueError("temperature_features must be positive")
    if float(signal_config["temperature_medoid"]) <= 0.0:
        raise ValueError("temperature_medoid must be positive")
    clip = tuple(float(value) for value in signal_config["importance_clip"])
    if len(clip) != 2 or not (0.0 < clip[0] <= clip[1]):
        raise ValueError("importance_clip must contain two ordered positive values")
    if not (0.0 <= float(signal_config["importance_consensus_floor"]) <= 1.0):
        raise ValueError("importance_consensus_floor must be in [0, 1]")
    if float(signal_config["feature_epsilon"]) <= 0.0:
        raise ValueError("feature_epsilon must be positive")
    if signal_config.get("pag_nonfinite_fallback") != "neutral":
        raise ValueError("The implemented PAG non-finite fallback is 'neutral'")
    if int(config["runtime"]["lm_head_chunk_tokens"]) <= 0:
        raise ValueError("runtime.lm_head_chunk_tokens must be positive")
    prepared_dir = Path(config["paths"]["prepared"])
    stage1_dir = Path(config["paths"]["stage1"])
    output_dir = Path(config["paths"]["output"])
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
    if stage1_manifest["prepared_manifest_fingerprint"] != fingerprint(
        prepared_manifest
    ):
        raise RuntimeError(
            "Stage-1 checkpoint was produced from a different prepared dataset"
        )
    stage1_config = stage1_manifest["config"]
    require_same_model_source(
        config["model"], stage1_config["model"], "Signals/Stage 1"
    )
    adapter_names = list(stage1_manifest["adapter_names"])
    model, created_names = create_multi_adapter_model(
        config["model"],
        stage1_config["lora"],
        len(adapter_names),
        distributed.device,
        int(stage1_config["seed"]),
    )
    if created_names != adapter_names:
        raise RuntimeError(
            f"Adapter-name mismatch: {created_names} versus {adapter_names}"
        )
    bundle = load_adapter_bundle(stage1_dir / stage1_manifest["adapter_bundle"])
    for name in adapter_names:
        load_adapter_state(model, name, bundle[name])
    tokenizer = load_tokenizer(config["model"])
    if tokenizer_fingerprint(tokenizer) != prepared_manifest["tokenizer_fingerprint"]:
        raise RuntimeError("Signal tokenizer does not match the prepared dataset")
    dataset = JsonlRecordDataset(prepared_dir)
    shard_path = output_dir / (
        f"signals-rank{distributed.rank:05d}-of{distributed.world_size:05d}.jsonl"
    )
    temporary = shard_path.with_suffix(".jsonl.tmp")
    medoid_sum = torch.zeros(
        len(adapter_names), device=distributed.device, dtype=torch.float64
    )
    medoid_tokens = torch.zeros((), device=distributed.device, dtype=torch.float64)
    nonfinite_pag = 0
    kneedle = stage1_config["kneedle"]
    chunk_tokens = int(config["runtime"]["lm_head_chunk_tokens"])
    processed = 0
    with temporary.open("w", encoding="utf-8") as handle:
        iterator = shard_indices(len(dataset), distributed.rank, distributed.world_size)
        for index in tqdm(iterator, desc=f"Signals rank {distributed.rank}"):
            record = dataset[index]
            predictive = score_predictive_signals(
                model,
                adapter_names,
                record,
                distributed.device,
                chunk_tokens,
                kneedle,
                float(signal_config["dpp_jitter"]),
                float(signal_config["temperature_features"]),
                float(signal_config["temperature_medoid"]),
            )
            pag_rows: list[torch.Tensor] = []
            pag_nonfinite_rows: list[torch.Tensor] = []
            for name in adapter_names:
                set_active_adapter(model, name)
                values = score_reference_solution_pag(
                    model, record, tokenizer, distributed.device, chunk_tokens
                )
                nonfinite = ~torch.isfinite(values)
                nonfinite_pag += int(nonfinite.sum().item())
                pag_nonfinite_rows.append(nonfinite)
                values = torch.where(nonfinite, torch.zeros_like(values), values)
                pag_rows.append(values)
            pag = (
                torch.stack(pag_rows)
                if pag_rows
                else torch.empty((len(adapter_names), 0))
            )
            importance, importance_parts = group_step_importance(
                pag,
                predictive.js_disagreement,
                predictive.token_counts,
                consensus_floor=float(signal_config["importance_consensus_floor"]),
                js_scale=float(signal_config["importance_js_scale"]),
                clip=tuple(float(value) for value in signal_config["importance_clip"]),
                mad_scale=float(signal_config["robust_mad_scale"]),
                epsilon=float(signal_config["feature_epsilon"]),
            )
            reasoning_weights, reasoning_scores = reasoning_teacher_weights(
                predictive.competence,
                predictive.agreement,
                predictive.uniqueness,
                uniform_mass=float(signal_config["uniform_teacher_mass"]),
                epsilon=float(signal_config["feature_epsilon"]),
            )
            answer_weights, answer_scores = answer_teacher_weights(
                predictive.answer_competence,
                predictive.answer_agreement,
                uniform_mass=float(signal_config["uniform_teacher_mass"]),
                epsilon=float(signal_config["feature_epsilon"]),
            )
            standardized_competence = zscore_across_experts(
                predictive.competence, float(signal_config["feature_epsilon"])
            )
            standardized_agreement = zscore_across_experts(
                predictive.agreement, float(signal_config["feature_epsilon"])
            )
            standardized_uniqueness = zscore_across_experts(
                predictive.uniqueness, float(signal_config["feature_epsilon"])
            )
            standardized_answer_competence = zscore_across_experts(
                predictive.answer_competence.unsqueeze(0),
                float(signal_config["feature_epsilon"]),
            ).squeeze(0)
            standardized_answer_agreement = zscore_across_experts(
                predictive.answer_agreement.unsqueeze(0),
                float(signal_config["feature_epsilon"]),
            ).squeeze(0)
            value = {
                "schema_version": 1,
                "sample_id": record.sample_id,
                "step_ids": list(range(record.kept_steps)),
                "step_token_counts": _as_list(predictive.token_counts),
                "pag": _as_list(pag),
                "pag_nonfinite_mask": _as_list(torch.stack(pag_nonfinite_rows)),
                "group_gain": _as_list(importance_parts["group_gain"]),
                "gain_consensus": _as_list(importance_parts["gain_consensus"]),
                "js_disagreement": _as_list(predictive.js_disagreement),
                "mean_uncertainty": _as_list(predictive.mean_uncertainty),
                "disagreement": _as_list(predictive.js_disagreement),
                "relative_disagreement": _as_list(predictive.relative_disagreement),
                "standardized_group_gain": _as_list(
                    importance_parts["standardized_gain"]
                ),
                "standardized_js_disagreement": _as_list(
                    importance_parts["standardized_js"]
                ),
                "importance": _as_list(importance),
                "competence": _as_list(predictive.competence),
                "agreement": _as_list(predictive.agreement),
                "uniqueness": _as_list(predictive.uniqueness),
                "standardized_competence": _as_list(standardized_competence),
                "standardized_agreement": _as_list(standardized_agreement),
                "standardized_uniqueness": _as_list(standardized_uniqueness),
                "teacher_scores": _as_list(reasoning_scores),
                "teacher_weights": _as_list(reasoning_weights),
                "answer_competence": _as_list(predictive.answer_competence),
                "answer_agreement": _as_list(predictive.answer_agreement),
                "standardized_answer_competence": _as_list(
                    standardized_answer_competence
                ),
                "standardized_answer_agreement": _as_list(
                    standardized_answer_agreement
                ),
                "answer_scores": _as_list(answer_scores),
                "answer_weights": _as_list(answer_weights),
            }
            handle.write(json.dumps(value, ensure_ascii=False) + "\n")
            medoid_sum += predictive.medoid_kl_sum.to(distributed.device)
            medoid_tokens += predictive.medoid_token_count
            processed += 1
    temporary.replace(shard_path)
    all_reduce_tensor(medoid_sum)
    all_reduce_tensor(medoid_tokens)
    nonfinite_tensor = torch.tensor(
        nonfinite_pag, device=distributed.device, dtype=torch.int64
    )
    processed_tensor = torch.tensor(
        processed, device=distributed.device, dtype=torch.int64
    )
    all_reduce_tensor(nonfinite_tensor)
    all_reduce_tensor(processed_tensor)
    barrier()
    if distributed.is_main:
        if int(processed_tensor.item()) != int(prepared_manifest["records"]):
            raise RuntimeError(
                "Signal shards do not cover every prepared training sample"
            )
        signal_paths = [
            output_dir / f"signals-rank{rank:05d}-of{distributed.world_size:05d}.jsonl"
            for rank in range(distributed.world_size)
        ]
        missing_shards = [str(path) for path in signal_paths if not path.is_file()]
        if missing_shards:
            raise RuntimeError(f"Missing signal shards: {missing_shards}")
        medoid_scores = medoid_sum / medoid_tokens.clamp_min(1.0)
        medoid_index = functional_medoid(medoid_scores)
        manifest = {
            "schema_version": 1,
            "artifact": "step_signals",
            "records": int(processed_tensor.item()),
            "shards": [path.name for path in signal_paths],
            "signals_fingerprint": files_fingerprint(signal_paths),
            "adapter_names": adapter_names,
            "functional_medoid_index": medoid_index,
            "functional_medoid_adapter": adapter_names[medoid_index],
            "functional_medoid_kl": _as_list(medoid_scores),
            "pag_nonfinite_fallbacks": int(nonfinite_tensor.item()),
            "prepared_manifest_fingerprint": fingerprint(prepared_manifest),
            "stage1_manifest_fingerprint": fingerprint(stage1_manifest),
            "config": public_config,
            "config_fingerprint": fingerprint(public_config),
            "config_file": config_snapshot.name,
            "config_file_sha256": file_sha256(config_snapshot),
            "runtime": runtime_metadata(config["_project_root"]),
        }
        write_json(output_dir / "manifest.json", manifest)
        LOGGER.info("Functional medoid: %s", adapter_names[medoid_index])
    barrier()
    return read_json(output_dir / "manifest.json")
