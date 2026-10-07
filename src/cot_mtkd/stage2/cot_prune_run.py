"""Prune prepared CoT traces with the frozen Phase-1 council, then publish them."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Sequence

import torch
from tqdm.auto import tqdm

from ..data.prepare import write_prepared_dataset
from ..data.schema import PreparedRecord, TokenRegion
from ..data.serialize import reasoning_character_segments
from ..models.chunked_head import (
    cross_entropy_from_hidden_no_grad,
    decoder_and_lm_head,
    forward_hidden,
)
from ..models.multi_adapter import (
    create_multi_adapter_model,
    load_adapter_bundle,
    load_adapter_state,
    load_tokenizer,
    require_same_model_source,
)
from ..signals.pag import record_pag_parts
from ..utils.distributed import DistributedContext
from ..utils.local_logging import JsonlLogger
from ..utils.manifest import read_json, require_file_sha256, write_json
from .cot_prune import (
    ScoreCache,
    candidate_token_ids,
    hierarchical_group_prune,
    sample_compression,
    token_cut_summary,
)
from .online import _select_adapter
from .teachers import ensure_stage2_teachers

LOGGER = logging.getLogger(__name__)


def reasoning_step_texts(thinking: str, step_pattern: str) -> list[str]:
    grouped: dict[int, list[str]] = {}
    for segment in reasoning_character_segments(thinking, step_pattern):
        if segment.region != TokenRegion.REASONING:
            continue
        grouped.setdefault(segment.step_id, []).append(thinking[segment.start : segment.end])
    if not grouped:
        return []
    last = max(grouped)
    if set(grouped) != set(range(last + 1)):
        raise ValueError("Reasoning step ids are not contiguous")
    return ["".join(grouped[index]) for index in range(last + 1)]


def validate_pruning_config(config: dict[str, Any]) -> dict[str, Any]:
    pruning = config.get("cot_pruning")
    if not isinstance(pruning, dict) or pruning.get("enabled") is not True:
        raise ValueError("cot_pruning.enabled must be true")
    if pruning["ensemble"].get("aggregation") != "geometric_mean":
        raise ValueError("cot_pruning.ensemble.aggregation must be geometric_mean")
    if pruning["ensemble"].get("score_space") != "log_probability":
        raise ValueError("cot_pruning.ensemble.score_space must be log_probability")
    if pruning.get("method") != "hierarchical_group_pruning":
        raise ValueError("cot_pruning.method must be hierarchical_group_pruning")
    search = pruning["search"]
    if search.get("method") != "hierarchical_group_pruning":
        raise ValueError("cot_pruning.search.method must be hierarchical_group_pruning")
    max_depth = search.get("max_depth", None)
    if max_depth is not None and int(max_depth) < 0:
        raise ValueError("cot_pruning.search.max_depth must be null or nonnegative")
    if int(pruning.get("min_steps", 1)) < 0:
        raise ValueError("cot_pruning.min_steps must be nonnegative")
    batch_size = search.get("candidate_batch_size", pruning.get("candidate_batch_size", 8))
    if int(batch_size) < 1:
        raise ValueError("cot_pruning.search.candidate_batch_size must be positive")
    if pruning.get("cache", {}).get("enabled") is not True:
        raise ValueError("cot_pruning.cache.enabled must be true")
    if not pruning.get("hf_repo"):
        raise ValueError("cot_pruning.hf_repo is required")
    if not pruning.get("data_config") or not pruning.get("source_prepared"):
        raise ValueError("cot_pruning needs data_config and source_prepared")
    if not pruning.get("export_dir"):
        raise ValueError("cot_pruning.export_dir is required")
    return pruning


def _answer_logprob_sums(
    model: torch.nn.Module,
    sequences: Sequence[Sequence[int]],
    solution_lengths: Sequence[int],
    device: torch.device,
    chunk_tokens: int,
) -> list[float]:
    """Sum of token log-probabilities of the trailing gold-answer span."""
    if len(sequences) != len(solution_lengths) or not sequences:
        raise ValueError("Each candidate needs one answer length")
    width = max(len(sequence) for sequence in sequences)
    token_ids = torch.zeros((len(sequences), width), dtype=torch.long, device=device)
    mask = torch.zeros_like(token_ids)
    for row, sequence in enumerate(sequences):
        if solution_lengths[row] < 1 or solution_lengths[row] >= len(sequence):
            raise ValueError("Gold answer must be a nonempty strict suffix of the candidate")
        token_ids[row, : len(sequence)] = torch.tensor(sequence, dtype=torch.long, device=device)
        mask[row, : len(sequence)] = 1
    with torch.inference_mode():
        hidden = forward_hidden(model, token_ids, mask, use_cache=False).last_hidden_state
    _, head = decoder_and_lm_head(model)
    scores: list[float] = []
    for row, length in enumerate(solution_lengths):
        sequence_length = int(mask[row].sum().item())
        start = sequence_length - length
        predicted = hidden[row, start - 1 : start + length - 1]
        targets = token_ids[row, start : start + length]
        loss_sum, count = cross_entropy_from_hidden_no_grad(
            predicted, head, targets, chunk_tokens
        )
        if count != length:
            raise RuntimeError("Answer log-probability did not cover every gold token")
        scores.append(-loss_sum)
    return scores


class FrozenCouncil:
    def __init__(self, config: dict[str, Any], distributed: DistributedContext) -> None:
        pruning = validate_pruning_config(config)
        self.pruning = pruning
        self.device = distributed.device
        self.chunk_tokens = int(config["runtime"]["lm_head_chunk_tokens"])
        search = pruning["search"]
        self.batch_size = int(
            search.get("candidate_batch_size", pruning.get("candidate_batch_size", 8))
        )
        self.min_steps = int(pruning.get("min_steps", 1))
        max_depth = search.get("max_depth", None)
        self.max_depth = None if max_depth is None else int(max_depth)
        stage1_dir = Path(config["paths"]["stage1"])
        teachers = ensure_stage2_teachers(config)
        bundle_path = require_file_sha256(
            stage1_dir, teachers, "adapter_bundle", "adapter_bundle_sha256"
        )
        require_same_model_source(
            config["model"], teachers["config"]["model"], "CoT pruning/teachers"
        )
        self.names = list(teachers["adapter_names"])
        if len(self.names) < 2:
            raise ValueError("CoT pruning needs at least two frozen experts")
        self.tokenizer = load_tokenizer(config["model"])
        self.model, created = create_multi_adapter_model(
            config["model"],
            teachers["config"]["lora"],
            len(self.names),
            distributed.device,
            int(teachers["config"].get("seed", config["seed"])),
        )
        if created != self.names:
            raise RuntimeError("Council adapter names mismatch")
        bundle = load_adapter_bundle(bundle_path)
        for name in self.names:
            load_adapter_state(self.model, name, bundle[name])
        self.model.eval()
        base = self.model.get_base_model() if hasattr(self.model, "get_base_model") else self.model
        if hasattr(base, "gradient_checkpointing_disable"):
            base.gradient_checkpointing_disable()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def score_many(self, sequences: Sequence[Sequence[int]], answer_length: int) -> list[float]:
        totals = [0.0] * len(sequences)
        for name in self.names:
            _select_adapter(self.model, name, training=False)
            for start in range(0, len(sequences), self.batch_size):
                chunk = sequences[start : start + self.batch_size]
                values = _answer_logprob_sums(
                    self.model,
                    chunk,
                    [answer_length] * len(chunk),
                    self.device,
                    self.chunk_tokens,
                )
                for offset, value in enumerate(values):
                    totals[start + offset] += value
        count = len(self.names)
        return [value / count for value in totals]


def _prepare_config(config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    from ..config import load_config

    data_config = load_config(config["cot_pruning"]["data_config"])
    data_config["output_dir"] = str(output_dir)
    data_config["dataset"]["allow_empty_thinking"] = True
    data_config["dataset"]["exclude_sample_ids"] = []
    data_config["dataset"]["expected_records"] = None
    data_config["dataset"]["expected_prepared_records"] = None
    return data_config


def _upload(rows: list[dict[str, Any]], repo_id: str, private: bool) -> None:
    if os.environ.get("HF_HUB_OFFLINE") == "1":
        raise RuntimeError(
            f"cot_pruning.hf_repo is {repo_id}, but HF_HUB_OFFLINE=1 blocks the upload"
        )
    from datasets import Dataset

    Dataset.from_list(rows).push_to_hub(repo_id, private=private)
    LOGGER.info("Uploaded %d pruned samples to %s", len(rows), repo_id)


def prune_dataset(config: dict[str, Any], distributed: DistributedContext) -> dict[str, Any]:
    """Score and shorten every sample, write prepared Phase-2 data, and publish it."""
    pruning = validate_pruning_config(config)
    if distributed.world_size != 1:
        raise RuntimeError("CoT pruning runs in one process; set NPROC_PER_NODE=1")
    source_dir = Path(pruning["source_prepared"])
    output_dir = Path(config["paths"]["prepared"])
    export_dir = Path(pruning["export_dir"])
    export_dir.mkdir(parents=True, exist_ok=True)
    source_manifest = read_json(source_dir / "manifest.json")
    source_data = require_file_sha256(source_dir, source_manifest, "data_file", "data_file_sha256")
    from ..data.dataset import JsonlRecordDataset

    dataset = JsonlRecordDataset(source_data)
    if len(dataset) != int(source_manifest["records"]):
        raise RuntimeError("Source prepared dataset record count mismatch")
    council = FrozenCouncil(config, distributed)
    step_pattern = str(source_manifest["config"]["serialization"]["step_pattern"])
    eta = float(pruning["eta"])
    logger = JsonlLogger(export_dir / "metrics.jsonl", truncate=True)
    rows: list[dict[str, Any]] = []
    seen_original = 0
    seen_deleted = 0
    progress = tqdm(dataset, total=len(dataset), desc="Pruning CoT")
    for record in progress:
        row = _prune_record(record, council, step_pattern, eta, pruning, logger)
        rows.append(row)
        seen_original += int(row["pruning"]["original_reasoning_tokens"])
        seen_deleted += int(row["pruning"]["deleted_reasoning_tokens"])
        running_cut = 0.0 if seen_original == 0 else 100.0 * seen_deleted / seen_original
        progress.set_postfix(sample=record.sample_id, token_cut=f"{running_cut:.1f}%")
    export_path = export_dir / "train.jsonl"
    temporary = export_path.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(export_path)
    prepared = write_prepared_dataset(rows, council.tokenizer, _prepare_config(config, output_dir))
    cuts = token_cut_summary(
        [
            (
                int(row["pruning"]["original_reasoning_tokens"]),
                int(row["pruning"]["deleted_reasoning_tokens"]),
            )
            for row in rows
        ]
    )
    count = len(rows)
    summary = {
        "records": count,
        "method": "hierarchical_group_pruning",
        "mean_deleted_steps": sum(row["pruning"]["num_deleted_steps"] for row in rows) / count,
        "mean_num_score_evaluations": sum(row["pruning"]["num_score_evaluations"] for row in rows)
        / count,
        "mean_step_reduction": sum(row["pruning"]["step_reduction"] for row in rows) / count,
        "mean_token_reduction": sum(row["pruning"]["token_reduction"] for row in rows) / count,
        "mean_fidelity": sum(row["pruning"]["fidelity"] for row in rows) / count,
        "mean_token_cut_percent": cuts["mean_token_cut_percent"],
        "overall_token_cut_percent": cuts["overall_token_cut_percent"],
        "total_original_reasoning_tokens": int(cuts["total_original_reasoning_tokens"]),
        "total_deleted_reasoning_tokens": int(cuts["total_deleted_reasoning_tokens"]),
        "prepared_manifest": prepared,
        "export_file": str(export_path),
        "hf_repo": pruning["hf_repo"],
    }
    write_json(export_dir / "summary.json", summary)
    _upload(rows, str(pruning["hf_repo"]), bool(pruning.get("hf_private", False)))
    LOGGER.info(
        "Pruned %d samples into %s: mean step reduction %.2f%%, "
        "mean token cut %.2f%%, overall token cut %.2f%% "
        "(%d/%d reasoning tokens), mean fidelity %.4f",
        count,
        output_dir,
        100.0 * summary["mean_step_reduction"],
        summary["mean_token_cut_percent"],
        summary["overall_token_cut_percent"],
        summary["total_deleted_reasoning_tokens"],
        summary["total_original_reasoning_tokens"],
        summary["mean_fidelity"],
    )
    return summary


def _prune_record(
    record: PreparedRecord,
    council: FrozenCouncil,
    step_pattern: str,
    eta: float,
    pruning: dict[str, Any],
    logger: JsonlLogger,
) -> dict[str, Any]:
    texts = reasoning_step_texts(record.thinking, step_pattern)
    prefix, steps, answer_prefix, solution = record_pag_parts(record, council.tokenizer)
    if len(texts) != len(steps):
        raise RuntimeError(
            f"{record.sample_id}: text steps ({len(texts)}) != token steps ({len(steps)})"
        )
    answer_length = len(solution)

    def encode(kept: tuple[int, ...], include_answer: bool) -> list[int]:
        return candidate_token_ids(
            prefix,
            steps,
            kept,
            answer_prefix if include_answer else (),
            solution if include_answer else (),
        )

    def score_many(keys: list[tuple[int, ...]]) -> list[float]:
        for key in keys:
            if key != tuple(sorted(set(key))) or any(not 0 <= index < len(steps) for index in key):
                raise RuntimeError(f"{record.sample_id}: search requested an invalid subset")
        sequences = [encode(key, True) for key in keys]
        prompt_ids = list(prefix)
        answer_ids = list(solution)
        for sequence in sequences:
            if sequence[: len(prompt_ids)] != prompt_ids:
                raise RuntimeError(f"{record.sample_id}: candidate changed the prompt tokens")
            if sequence[-answer_length:] != answer_ids:
                raise RuntimeError(f"{record.sample_id}: candidate changed the answer target")
        return council.score_many(sequences, answer_length)

    cache = ScoreCache(lambda kept: score_many([kept])[0])
    result = hierarchical_group_prune(
        len(steps),
        cache,
        eta,
        min_steps=council.min_steps,
        max_depth=council.max_depth,
        score_many=score_many,
    )
    chosen = result.kept
    original_tokens = sum(len(step) for step in steps)
    kept_tokens = sum(len(steps[index]) for index in chosen)
    metrics = sample_compression(
        len(steps),
        len(chosen),
        original_tokens,
        kept_tokens,
        result.original_score,
        result.final_score,
    )
    short_text = "\n\n".join(texts[index] for index in chosen)
    payload = {
        "id": record.sample_id,
        "question": record.question,
        "prompt": record.question,
        "solution": record.solution,
        "answer": record.solution,
        "deepseek_grade": record.deepseek_grade,
        "deepseek_thinking_trajectory": short_text,
        "reasoning_original": texts
        if pruning["output"].get("save_original_reasoning", True)
        else [],
        "reasoning_short": [texts[index] for index in chosen]
        if pruning["output"].get("save_pruned_reasoning", True)
        else [],
        "pruning": {
            "algorithm": "hierarchical_group_pruning",
            "method": "hierarchical_group_pruning",
            "eta": eta,
            "original_score": result.original_score,
            "final_score": result.final_score,
            "threshold": result.threshold,
            "original_num_steps": len(steps),
            "final_num_steps": len(chosen),
            "num_deleted_steps": len(steps) - len(chosen),
            "deleted_indices": list(result.deleted_indices),
            "num_model_evaluations": result.num_score_evaluations,
            "num_score_evaluations": result.num_score_evaluations,
            "num_unique_subsets_evaluated": result.num_unique_subsets_evaluated,
            "cache_hits": result.cache_hits,
            "original_reasoning_tokens": original_tokens,
            "final_reasoning_tokens": kept_tokens,
            "original_num_reasoning_tokens": original_tokens,
            "final_num_reasoning_tokens": kept_tokens,
            "deleted_reasoning_tokens": original_tokens - kept_tokens,
            "answer_tokens": answer_length,
            "step_reduction": metrics["step_reduction"],
            "token_reduction": metrics["token_reduction"],
            "token_reduction_ratio": metrics["token_reduction"],
            "fidelity": metrics["fidelity"],
            "ensemble_likelihood_ratio": metrics["fidelity"],
        },
    }
    details = payload["pruning"]
    LOGGER.info(
        "sample_id=%s original_num_steps=%d final_num_steps=%d "
        "original_num_reasoning_tokens=%d final_num_reasoning_tokens=%d "
        "token_reduction_ratio=%.4f answer_tokens=%d original_score=%.6f "
        "final_score=%.6f threshold=%.6f eta=%s num_deleted_steps=%d "
        "num_model_evaluations=%d cache_hits=%d ensemble_likelihood_ratio=%.6g",
        record.sample_id,
        details["original_num_steps"],
        details["final_num_steps"],
        details["original_num_reasoning_tokens"],
        details["final_num_reasoning_tokens"],
        details["token_reduction_ratio"],
        details["answer_tokens"],
        details["original_score"],
        details["final_score"],
        details["threshold"],
        eta,
        details["num_deleted_steps"],
        details["num_model_evaluations"],
        details["cache_hits"],
        details["ensemble_likelihood_ratio"],
    )
    logger.log("cot_prune", sample_id=record.sample_id, **details)
    return payload
