from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from ..utils.training import vector_norm

STAGE1_METHOD = "sft_dpp_rbf"


@dataclass(frozen=True)
class ObjectiveDiagnostics:
    task_norms: tuple[float, ...]
    rbf_norms: tuple[float, ...]
    weighted_rbf_norms: tuple[float, ...]
    rbf_task_ratios: tuple[float, ...]


def compose_objective_gradients(
    task_gradients: list[list[torch.Tensor]],
    rbf_gradients: list[list[torch.Tensor]],
    rbf_weight: float,
) -> tuple[list[list[torch.Tensor]], ObjectiveDiagnostics]:
    """Add each expert's own gradients of SFT + lambda_D DPP + lambda_R RBF.

    Task gradients already have global SFT-token/DPP-sample normalization.
    RBF gradients are positive loss gradients, evaluated once per optimizer
    window with detached bandwidth. There is no sharing or repulsion norm cap;
    the caller clips only the final gradient before AdamW.
    """
    if not math.isfinite(rbf_weight) or rbf_weight < 0.0:
        raise ValueError("rbf_weight must be finite and non-negative")
    if not task_gradients or len(task_gradients) != len(rbf_gradients):
        raise ValueError("Inconsistent expert dimensions in objective gradients")
    final: list[list[torch.Tensor]] = []
    for task, rbf in zip(task_gradients, rbf_gradients, strict=True):
        if not task or len(task) != len(rbf):
            raise ValueError("Inconsistent parameter counts in objective gradients")
        current = []
        for local, regularizer in zip(task, rbf, strict=True):
            if local.shape != regularizer.shape or local.device != regularizer.device:
                raise ValueError("Objective gradients must have matching shapes and devices")
            current.append(local.float() + rbf_weight * regularizer.float())
        final.append(current)
    task_norms = tuple(float(vector_norm(values).item()) for values in task_gradients)
    rbf_norms = tuple(float(vector_norm(values).item()) for values in rbf_gradients)
    weighted = tuple(rbf_weight * value for value in rbf_norms)
    diagnostics = ObjectiveDiagnostics(
        task_norms=task_norms,
        rbf_norms=rbf_norms,
        weighted_rbf_norms=weighted,
        rbf_task_ratios=tuple(
            regularizer / max(task, 1.0e-12)
            for regularizer, task in zip(weighted, task_norms, strict=True)
        ),
    )
    return final, diagnostics
