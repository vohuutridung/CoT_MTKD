from __future__ import annotations

import statistics
from typing import Any, Sequence


def problem_scores(correct: Sequence[bool]) -> tuple[float, float]:
    """Pass@1 is the mean accuracy over the k samples; Pass@k is any-correct."""
    if not correct:
        return 0.0, 0.0
    return sum(map(float, correct)) / len(correct), float(any(correct))


def benchmark_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("Cannot score an empty benchmark")
    k = len(records[0]["correct"])
    pairs = [problem_scores(record["correct"]) for record in records]
    lengths = [length for record in records for length in record["token_counts"]]
    return {
        "pass@1": sum(pair[0] for pair in pairs) / len(pairs),
        f"pass@{k}": sum(pair[1] for pair in pairs) / len(pairs),
        "n_problems": len(records),
        "k": k,
        "mean_tokens": sum(lengths) / len(lengths) if lengths else 0.0,
        "truncated_fraction": (
            sum(
                flag for record in records for flag in record["truncated"]
            )
            / len(lengths)
            if lengths
            else 0.0
        ),
        "per_problem": [pair[0] for pair in pairs],
        "per_problem_passk": [pair[1] for pair in pairs],
    }


def macro_average(benchmarks: dict[str, dict[str, Any]]) -> dict[str, float]:
    """Unweighted mean over benchmarks, as reported by P-ALIGN and exp_s1k."""
    pass1 = [value["pass@1"] for value in benchmarks.values()]
    passk = [
        value[key]
        for value in benchmarks.values()
        for key in value
        if key.startswith("pass@") and key != "pass@1"
    ]
    return {
        "avg": sum(pass1) / len(pass1) if pass1 else 0.0,
        "avg_passk": sum(passk) / len(passk) if passk else 0.0,
    }


def _mean_std(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def seed_summary(results: dict[int, dict[str, Any]]) -> dict[str, Any]:
    """Mean and sample standard deviation over seeds of every reported score."""
    seeds = sorted(results)
    first = results[seeds[0]]
    benchmarks: dict[str, Any] = {}
    for name in first["benchmarks"]:
        keys = [key for key in first["benchmarks"][name] if key.startswith("pass@")]
        benchmarks[name] = {
            key: _mean_std([results[seed]["benchmarks"][name][key] for seed in seeds])
            for key in keys
        }
    return {
        "seeds": seeds,
        "avg": _mean_std([results[seed]["avg"] for seed in seeds]),
        "avg_passk": _mean_std([results[seed]["avg_passk"] for seed in seeds]),
        "benchmarks": benchmarks,
    }
