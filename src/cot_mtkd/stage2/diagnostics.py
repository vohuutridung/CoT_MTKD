"""End-of-run step summaries, computed from existing logs without model work."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from .council_cache import distribution_summary, threshold_summary


def summarize_reasoning_logs(paths: list[Path], run_fingerprint: str) -> dict:
    keys = (
        "token_js_mean",
        "step_disagreement",
        "rho",
        "support_size_mean",
        "target_support_mass",
        "target_tail_mass",
        "student_support_mass",
        "student_tail_mass",
        "step_kd_loss",
        "step_sft_loss",
        "weighted_sft_loss",
        "step_total_loss",
    )
    values = {key: [] for key in keys}
    for path in paths:
        with path.open(encoding="utf-8") as source:
            for line in source:
                row = json.loads(line)
                if row["run_fingerprint"] != run_fingerprint:
                    raise RuntimeError(f"Foreign run in Phase-2 diagnostics: {path}")
                for key in keys:
                    values[key].append(row[key])
    tensors = {k: torch.tensor(v, dtype=torch.float64) for k, v in values.items()}
    result = {k: distribution_summary(v) for k, v in tensors.items()}
    for key in ("target_tail_mass", "student_tail_mass"):
        result[key] = threshold_summary(tensors[key], (0.10, 0.25, 0.50))
    result["rho"] = threshold_summary(tensors["rho"], (0.25, 0.50, 0.75))
    return {
        "scope": "observed_training_step_means_including_restored_logs",
        "num_reasoning_steps": len(values["rho"]),
        "statistics": result,
    }
