from __future__ import annotations

from typing import Any


def pass_metrics(records: list[dict[str, Any]]) -> dict[str, float | int]:
    """Pass@1 averages every sample; Pass@3 is whether any sample is correct.

    For problem i with correctness flags y1, y2, y3, Pass@1 uses (y1+y2+y3)/3
    and then averages over problems. With exactly three generations this equals
    the total number of correct generations divided by 3N. Pass@3 is 1 when any
    of those three flags is true.
    """
    if not records:
        return {"problems": 0, "pass_at_1": 0.0, "pass_at_3": 0.0}
    sample_means: list[float] = []
    any_correct = 0
    for record in records:
        flags = [bool(value) for value in record["correct"][:3]]
        if len(flags) != 3:
            raise ValueError(
                f"Pass@1/Pass@3 expect exactly 3 generations, got {len(flags)}"
            )
        sample_means.append(sum(flags) / 3)
        any_correct += int(any(flags))
    count = len(records)
    return {
        "problems": count,
        "pass_at_1": sum(sample_means) / count,
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
