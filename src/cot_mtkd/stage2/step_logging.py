from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..utils.local_logging import JsonlLogger


def reasoning_log_filename(rank: int, world_size: int) -> str:
    if world_size == 1:
        return "reasoning_steps.jsonl"
    return f"reasoning_steps.rank{rank:05d}.jsonl"


def trim_jsonl_to_checkpoint(
    path: Path,
    run_fingerprint: str,
    data_step: int,
    *,
    before_update: bool,
) -> None:
    """Remove entries produced beyond a saved effective-batch cursor on resume.

    Step rows are emitted before updates: keep data_step_before < data_step.
    Aggregate rows follow updates: keep data_step <= the checkpoint cursor.
    A final interrupted write is discarded; corrupt complete rows fail loudly.
    """
    if not path.exists():
        return
    temporary = path.with_suffix(path.suffix + ".recovering")
    key = "data_step_before" if before_update else "data_step"
    try:
        with (
            path.open("r", encoding="utf-8") as source,
            temporary.open("w", encoding="utf-8") as destination,
        ):
            for line_number, line in enumerate(source, start=1):
                if not line.endswith("\n"):
                    # JsonlLogger always writes a newline, so this is a partial
                    # final record from an interrupted process.
                    break
                try:
                    value = json.loads(line)
                    if value["run_fingerprint"] != run_fingerprint:
                        raise RuntimeError(f"Refusing to append to logs from another run: {path}")
                    cursor = value[key]
                    if isinstance(cursor, bool) or not isinstance(cursor, int) or cursor < 0:
                        raise ValueError("Invalid data-step cursor")
                except (ValueError, KeyError, TypeError) as error:
                    raise RuntimeError(f"Invalid log record at {path}:{line_number}") from error
                keep = cursor < data_step if before_update else cursor <= data_step
                if keep:
                    destination.write(line)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def create_reasoning_logger(
    output_dir: Path,
    rank: int,
    world_size: int,
    run_fingerprint: str,
    data_step: int,
    resume: bool,
) -> JsonlLogger:
    path = output_dir / reasoning_log_filename(rank, world_size)
    if resume:
        trim_jsonl_to_checkpoint(path, run_fingerprint, data_step, before_update=True)
    return JsonlLogger(path, truncate=not resume)


def log_record_steps(
    logger: JsonlLogger,
    result: Any,
    sample_id: str,
    *,
    run_fingerprint: str,
    rank: int,
    epoch: int,
    batch_in_epoch: int,
    sample_in_batch: int,
    data_step_before: int,
    global_step_before: int,
) -> None:
    for values in result.step_metrics:
        logger.log(
            "stage2_reasoning_step",
            run_fingerprint=run_fingerprint,
            rank=rank,
            epoch=epoch,
            batch_in_epoch=batch_in_epoch,
            sample_in_batch=sample_in_batch,
            data_step_before=data_step_before,
            global_step_before=global_step_before,
            sample_id=sample_id,
            sample_total_loss=result.loss,
            sample_sft_loss=result.metrics.get("sft_loss", 0.0),
            sample_kl_loss=result.metrics.get("kl_loss", 0.0),
            sample_mass_loss=result.metrics.get("mass_loss", 0.0),
            retained_steps=result.steps,
            discarded_steps=result.discarded_steps,
            **values,
        )
