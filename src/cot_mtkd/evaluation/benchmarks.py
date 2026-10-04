"""Benchmark loading identical to the exp_s1k P-ALIGN evaluation (``eval/benchmarks.py``).

AMC12 is the 83-problem set shipped with the P-ALIGN repository; its questions
and answers are vendored in ``data/eval/amc12_p_align.jsonl`` (prompt prefix
stripped, ``142.0`` written as ``142``) so evaluation needs no extra checkout.
AIME 2025 is opencompass Part I followed by Part II.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass(frozen=True)
class EvalItem:
    index: int
    question: str
    answer: str


def _question(row: dict[str, Any]) -> str:
    for key in ("problem", "question", "Question", "query"):
        if row.get(key):
            return str(row[key])
    return ""


def _answer(row: dict[str, Any]) -> Optional[str]:
    for key in ("answer", "Answer", "solution", "expected_answer"):
        if key in row and row[key] not in (None, ""):
            return str(row[key])
    return None


def _rows(specification: dict[str, Any], project_root: Path) -> list[dict[str, Any]]:
    local_json = specification.get("local_json")
    if local_json:
        path = Path(local_json).expanduser()
        if not path.is_absolute():
            path = project_root / path
        with path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    from datasets import load_dataset

    rows: list[dict[str, Any]] = []
    for subset in specification.get("subsets") or [None]:
        dataset = load_dataset(
            specification["dataset"],
            subset,
            split=specification["split"],
            revision=specification.get("revision"),
        )
        rows.extend(dict(row) for row in dataset)
    return rows


def load_benchmark(
    specification: dict[str, Any], project_root: Path, limit: Optional[int] = None
) -> list[EvalItem]:
    items: list[EvalItem] = []
    for index, row in enumerate(_rows(specification, project_root)):
        question, answer = _question(row), _answer(row)
        if not question or answer is None:
            continue
        items.append(EvalItem(index=index, question=question, answer=answer))
        if limit and len(items) >= limit:
            break
    expected = specification.get("expected_records")
    if limit is None and expected is not None and len(items) != int(expected):
        raise RuntimeError(
            f"Benchmark {specification['name']} has {len(items)} problems; "
            f"expected {int(expected)}"
        )
    return items
