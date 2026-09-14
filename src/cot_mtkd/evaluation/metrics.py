from __future__ import annotations

from typing import Any


def pass_metrics(records: list[dict[str, Any]]) -> dict[str, float | int]:
    if not records:
        return {"problems": 0, "pass_at_1": 0.0, "pass_at_3": 0.0}
    first = sum(bool(record["correct"][0]) for record in records)
    any_correct = sum(
        any(bool(value) for value in record["correct"][:3]) for record in records
    )
    count = len(records)
    return {
        "problems": count,
        "pass_at_1": first / count,
        "pass_at_3": any_correct / count,
    }


def macro_average(metrics: dict[str, dict[str, float | int]]) -> dict[str, float]:
    available = [value for value in metrics.values() if int(value["problems"]) > 0]
    if not available:
        return {"pass_at_1": 0.0, "pass_at_3": 0.0}
    return {
        "pass_at_1": sum(float(value["pass_at_1"]) for value in available)
        / len(available),
        "pass_at_3": sum(float(value["pass_at_3"]) for value in available)
        / len(available),
    }
