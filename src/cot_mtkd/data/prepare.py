from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Iterable

from tqdm.auto import tqdm

from ..utils.manifest import (
    file_sha256,
    fingerprint,
    runtime_metadata,
    write_config_snapshot,
    write_json,
)
from .schema import PreparedRecord, TokenRegion
from .serialize import serialize_record
from .token_spans import (
    assign_token_regions,
    complete_step_truncate,
    first_region_index,
    validate_token_contract,
)

LOGGER = logging.getLogger(__name__)


def tokenizer_fingerprint(tokenizer: Any) -> str:
    payload = {
        "class": tokenizer.__class__.__name__,
        "name_or_path": getattr(tokenizer, "name_or_path", None),
        "vocab_size": len(tokenizer),
        "special_tokens_map": getattr(tokenizer, "special_tokens_map", {}),
        "added_vocab": getattr(tokenizer, "get_added_vocab", lambda: {})(),
    }
    return fingerprint(payload)


def prepare_one(
    raw: dict[str, Any],
    tokenizer: Any,
    sample_index: int,
    max_length: int,
    system_prompt: str,
    step_pattern: str,
    tokenizer_hash: str,
) -> PreparedRecord:
    question = str(raw["question"])
    thinking = (
        str(raw["deepseek_thinking_trajectory"])
        .replace("\r\n", "\n")
        .replace("\r", "\n")
    )
    attempt = str(raw["deepseek_attempt"])
    solution = str(raw["solution"])
    serialized = serialize_record(
        question, thinking, attempt, system_prompt, step_pattern
    )
    encoded = tokenizer(
        serialized.text,
        add_special_tokens=False,
        return_attention_mask=False,
        return_offsets_mapping=True,
        truncation=False,
    )
    input_ids = list(encoded["input_ids"])
    offsets = [tuple(pair) for pair in encoded["offset_mapping"]]
    regions, steps = assign_token_regions(offsets, serialized.segments)
    labels = [
        token if region != int(TokenRegion.PROMPT) else -100
        for token, region in zip(input_ids, regions)
    ]
    original_answer_start = first_region_index(
        regions, TokenRegion.ANSWER_MARKER, len(input_ids)
    )
    original_length = len(input_ids)
    input_ids, labels, regions, steps, stats = complete_step_truncate(
        input_ids, labels, regions, steps, max_length
    )
    if bool(stats["truncated"]):
        suffix_length = original_length - original_answer_start
        prefix_length = len(input_ids) - suffix_length
        offsets = offsets[:prefix_length] + offsets[original_answer_start:]
    validate_token_contract(input_ids, labels, regions, steps)
    if len(offsets) != len(input_ids):
        raise RuntimeError("Offset mapping became misaligned during truncation")
    identifier = raw.get("id")
    if identifier is None:
        identifier = raw.get("problem_id")
    if identifier is None:
        identifier = f"s1k-{sample_index:04d}"
    sample_id = str(identifier)
    answer_start = first_region_index(
        regions, TokenRegion.ANSWER_MARKER, len(input_ids)
    )
    reasoning_start = first_region_index(regions, TokenRegion.REASONING, answer_start)
    return PreparedRecord(
        sample_id=sample_id,
        input_ids=input_ids,
        labels=labels,
        attention_mask=[1] * len(input_ids),
        offset_mapping=offsets,
        region_ids=regions,
        step_ids=steps,
        question=question,
        thinking=thinking,
        attempt=attempt,
        solution=solution,
        deepseek_grade=(
            None
            if raw.get("deepseek_grade") is None
            else str(raw.get("deepseek_grade"))
        ),
        original_length=int(stats["original_length"]),
        kept_length=int(stats["kept_length"]),
        original_steps=int(stats["original_steps"]),
        kept_steps=int(stats["kept_steps"]),
        truncated=bool(stats["truncated"]),
        answer_start=answer_start,
        reasoning_start=reasoning_start,
        tokenizer_fingerprint=tokenizer_hash,
    )


def write_prepared_dataset(
    records: Iterable[dict[str, Any]], tokenizer: Any, config: dict[str, Any]
) -> dict[str, Any]:
    if config["dataset"].get("keep_all_grades") is not True:
        raise ValueError(
            "Canonical s1K-1.1 preprocessing requires keep_all_grades: true"
        )
    tokenization = config["tokenization"]
    if tokenization.get("padding_side") != "right":
        raise ValueError("Canonical preprocessing requires right padding")
    if tokenization.get("packing") is not False:
        raise ValueError("Canonical preprocessing does not permit sequence packing")
    if tokenization.get("add_special_tokens") is not False:
        raise ValueError("Canonical serialization already contains its special tokens")
    if tokenization.get("truncation") != "complete_reasoning_step_prefix":
        raise ValueError("Unsupported truncation policy")
    if int(tokenization["max_length"]) <= 0:
        raise ValueError("tokenization.max_length must be positive")
    destination = Path(config["output_dir"])
    destination.mkdir(parents=True, exist_ok=True)
    public_config = {
        key: value for key, value in config.items() if not key.startswith("_")
    }
    config_path = write_config_snapshot(destination / "config.yaml", public_config)
    output_file = destination / "data.jsonl"
    temporary = output_file.with_suffix(".jsonl.tmp")
    tokenizer_hash = tokenizer_fingerprint(tokenizer)
    count = truncated_count = token_count = 0
    sample_ids: list[str] = []
    seen_sample_ids: set[str] = set()
    with temporary.open("w", encoding="utf-8") as handle:
        for index, raw in enumerate(tqdm(records, desc="Preparing s1K-1.1")):
            prepared = prepare_one(
                raw,
                tokenizer,
                index,
                int(config["tokenization"]["max_length"]),
                str(config["serialization"]["system_prompt"]),
                str(config["serialization"]["step_pattern"]),
                tokenizer_hash,
            )
            if prepared.sample_id in seen_sample_ids:
                raise ValueError(
                    f"Duplicate sample id in source dataset: {prepared.sample_id}"
                )
            seen_sample_ids.add(prepared.sample_id)
            handle.write(json.dumps(prepared.to_dict(), ensure_ascii=False) + "\n")
            count += 1
            truncated_count += int(prepared.truncated)
            token_count += prepared.kept_length
            sample_ids.append(prepared.sample_id)
    expected_records = config["dataset"].get("expected_records")
    if expected_records is not None and count != int(expected_records):
        raise RuntimeError(
            f"Expected {int(expected_records)} source records, but prepared {count}"
        )
    temporary.replace(output_file)
    manifest = {
        "schema_version": 1,
        "artifact": "prepared_dataset",
        "records": count,
        "tokens": token_count,
        "truncated_records": truncated_count,
        "tokenizer_fingerprint": tokenizer_hash,
        "sample_order_fingerprint": hashlib.sha256(
            "\n".join(sample_ids).encode()
        ).hexdigest(),
        "data_file": output_file.name,
        "data_file_sha256": file_sha256(output_file),
        "config": public_config,
        "config_fingerprint": fingerprint(public_config),
        "config_file": config_path.name,
        "config_file_sha256": file_sha256(config_path),
        "runtime": runtime_metadata(config["_project_root"]),
    }
    write_json(destination / "manifest.json", manifest)
    LOGGER.info(
        "Prepared %d records (%d truncated) at %s", count, truncated_count, destination
    )
    return manifest
