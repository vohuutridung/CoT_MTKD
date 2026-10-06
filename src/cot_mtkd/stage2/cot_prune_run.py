"""Prune prepared CoT traces with the frozen Phase-1 council, then publish them."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Sequence

import torch

from ..data.prepare import write_prepared_dataset
from ..data.schema import PreparedRecord, TokenRegion
from ..data.serialize import reasoning_character_segments
from ..evaluation.grading import grade_math, normalize_answer
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
from ..signals.pag import answer_continuation_ids, record_pag_parts
from ..utils.distributed import DistributedContext
from ..utils.local_logging import JsonlLogger
from ..utils.manifest import read_json, require_file_sha256, write_json
from .cot_prune import ScoreCache, candidate_token_ids, choose_fallback, greedy_delete
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
    if pruning["search"].get("method") != "greedy_deletion":
        raise ValueError("cot_pruning.search.method must be greedy_deletion")
    if pruning["search"].get("rerank_after_each_deletion") is not True:
        raise ValueError("cot_pruning.search.rerank_after_each_deletion must be true")
    correctness = pruning["correctness"]
    if (
        correctness.get("enabled") is not True
        or correctness.get("check_after_pruning_only") is not True
    ):
        raise ValueError("Correctness is checked only after greedy deletion stops")
    if (
        correctness.get("method") != "majority_vote"
        or correctness.get("fallback_on_failure") is not True
    ):
        raise ValueError("cot_pruning.correctness requires majority_vote and fallback_on_failure")
    if int(pruning.get("candidate_batch_size", 1)) < 1:
        raise ValueError("cot_pruning.candidate_batch_size must be positive")
    if int(correctness.get("max_new_tokens", 1)) < 1:
        raise ValueError("cot_pruning.correctness.max_new_tokens must be positive")
    if not pruning.get("hf_repo"):
        raise ValueError("cot_pruning.hf_repo is required")
    if not pruning.get("data_config") or not pruning.get("source_prepared"):
        raise ValueError("cot_pruning needs data_config and source_prepared")
    if not pruning.get("export_dir"):
        raise ValueError("cot_pruning.export_dir is required")
    return pruning


def majority_vote_correct(
    predictions: Sequence[str], reference: str, timeout_seconds: float
) -> tuple[bool, str | None]:
    """Return whether the unique most common answer matches the ground truth."""
    groups: dict[str, list[str]] = {}
    for prediction in predictions:
        groups.setdefault(normalize_answer(prediction), []).append(prediction)
    if not groups:
        return False, None
    counts = [len(values) for values in groups.values()]
    best = max(counts)
    winners = [key for key, values in groups.items() if len(values) == best]
    if len(winners) != 1:
        return False, None
    chosen = groups[winners[0]][0]
    return grade_math(chosen, reference, True, timeout_seconds), chosen


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
        self.batch_size = int(pruning.get("candidate_batch_size", 1))
        self.max_new_tokens = int(pruning["correctness"]["max_new_tokens"])
        self.grade_timeout = float(pruning["correctness"].get("timeout_seconds", 10))
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

    def generate(self, name: str, prompt_ids: Sequence[int]) -> str:
        _select_adapter(self.model, name, training=False)
        self.model.eval()
        room = int(self.model.config.max_position_embeddings) - len(prompt_ids)
        if room < 1:
            return ""
        self.model.config.use_cache = True
        token_ids = torch.tensor([list(prompt_ids)], dtype=torch.long, device=self.device)
        with torch.inference_mode():
            output = self.model.generate(
                input_ids=token_ids,
                attention_mask=torch.ones_like(token_ids),
                max_new_tokens=min(self.max_new_tokens, room),
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        continuation = output[0, token_ids.shape[1] :]
        return self.tokenizer.decode(continuation, skip_special_tokens=True)


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
    for record in dataset:
        rows.append(_prune_record(record, council, step_pattern, eta, pruning, logger))
    export_path = export_dir / "train.jsonl"
    temporary = export_path.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(export_path)
    prepared = write_prepared_dataset(rows, council.tokenizer, _prepare_config(config, output_dir))
    summary = {
        "records": len(rows),
        "mean_deleted": sum(row["pruning"]["num_deleted"] for row in rows) / len(rows),
        "fallback_count": sum(bool(row["pruning"]["fallback_used"]) for row in rows),
        "correct_count": sum(bool(row["pruning"]["correct_after_pruning"]) for row in rows),
        "prepared_manifest": prepared,
        "export_file": str(export_path),
        "hf_repo": pruning["hf_repo"],
    }
    write_json(export_dir / "summary.json", summary)
    _upload(rows, str(pruning["hf_repo"]), bool(pruning.get("hf_private", False)))
    LOGGER.info(
        "Pruned %d samples (mean deleted %.2f, fallbacks %d) into %s",
        len(rows),
        summary["mean_deleted"],
        summary["fallback_count"],
        output_dir,
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
        sequences = [encode(key, True) for key in keys]
        if any(sequence[: len(prefix)] != list(prefix) for sequence in sequences):
            raise RuntimeError(f"{record.sample_id}: candidate changed the prompt tokens")
        return council.score_many(sequences, answer_length)

    cache = ScoreCache(lambda kept: score_many([kept])[0])
    result = greedy_delete(len(steps), cache, eta, score_many=score_many)
    correctness_calls: list[tuple[int, ...]] = []

    def is_correct(kept: tuple[int, ...]) -> bool:
        correctness_calls.append(kept)
        predictions = [
            council.generate(name, encode(kept, False)) for name in council.names
        ]
        accepted, _vote = majority_vote_correct(predictions, record.solution, council.grade_timeout)
        return accepted

    chosen, correct, fallback_used = choose_fallback(result.states, is_correct)
    if not correctness_calls or correctness_calls[0] != result.kept:
        raise RuntimeError("Correctness must start at the final pruned trace")
    short_text = "\n\n".join(texts[index] for index in chosen)
    history = [item.__dict__ for item in result.history] if pruning["output"].get(
        "save_deletion_history", True
    ) else []
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
            "eta": eta,
            "original_score": result.original_score,
            "final_score": result.final_score if chosen == result.kept else cache(chosen),
            "threshold": result.threshold,
            "original_num_steps": len(steps),
            "final_num_steps": len(chosen),
            "num_deleted": len(steps) - len(chosen),
            "correct_after_pruning": correct,
            "fallback_used": fallback_used,
            "deletion_history": history,
            "accepted_steps": list(chosen),
        },
    }
    details = payload["pruning"]
    LOGGER.info(
        "sample_id=%s original_num_steps=%d final_num_steps=%d original_score=%.6f "
        "final_score=%.6f threshold=%.6f eta=%s num_deleted=%d "
        "correct_after_pruning=%s fallback_used=%s",
        record.sample_id,
        details["original_num_steps"],
        details["final_num_steps"],
        details["original_score"],
        details["final_score"],
        details["threshold"],
        eta,
        details["num_deleted"],
        details["correct_after_pruning"],
        details["fallback_used"],
    )
    logger.log("cot_prune", sample_id=record.sample_id, **details)
    return payload
