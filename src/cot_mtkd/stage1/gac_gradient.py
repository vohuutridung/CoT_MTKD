from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from ..utils.training import vector_norm

STAGE1_METHOD = "sft_dpp_rbf_local_gac"


@dataclass(frozen=True)
class GACDiagnostics:
    task_norms: tuple[float, ...]
    mixed_task_norms: tuple[float, ...]
    cross_coefficients: tuple[float, ...]
    self_coefficients: tuple[float, ...]
    repulsion_norms_before_cap: tuple[float, ...]
    repulsion_cap_factors: tuple[float, ...]
    capped_repulsion_norms: tuple[float, ...]
    weighted_repulsion_norms: tuple[float, ...]
    final_norms: tuple[float, ...]


def stable_gac_gradients(
    task_gradients: list[list[torch.Tensor]],
    repulsion: list[list[torch.Tensor]],
    gac_kernel: torch.Tensor,
    beta: float = 0.5,
    rbf_weight: float = 0.5,
) -> tuple[list[list[torch.Tensor]], GACDiagnostics]:
    """Local GAC sharing followed by capped outward RBF repulsion.

    Tasks already contain globally normalized g_SFT + lambda_D * g_DPP.
    Neighbor j != i contributes beta/(M-1) * K_G[j,i], without normalization.
    The remaining coefficient belongs to the expert's own task. Repulsion is
    -grad(mean unordered-pair K_R), computed separately using h_R.
    """
    if not math.isfinite(beta) or not 0.0 <= beta <= 1.0:
        raise ValueError("gac_beta must be finite and in [0, 1]")
    if not math.isfinite(rbf_weight) or rbf_weight < 0.0:
        raise ValueError("rbf_weight must be finite and non-negative")
    count = len(task_gradients)
    if count < 2 or len(repulsion) != count or gac_kernel.shape != (count, count):
        raise ValueError("Inconsistent expert dimensions in GAC inputs")
    for task, outward in zip(task_gradients, repulsion, strict=True):
        if not task or len(task) != len(outward) or len(task) != len(task_gradients[0]):
            raise ValueError("Inconsistent parameter counts in GAC inputs")
        for index, (local, repel) in enumerate(zip(task, outward, strict=True)):
            reference = task_gradients[0][index]
            if local.shape != reference.shape or local.shape != repel.shape:
                raise ValueError("GAC gradients must have matching parameter shapes")
            if local.device != reference.device or local.device != repel.device:
                raise ValueError("GAC gradients must have matching devices")

    # Only this small M-by-M matrix leaves the device. Its diagonal is excluded
    # before summation; neither K_R nor a normalized neighbor distribution enters.
    neighbors = gac_kernel.detach().to(device="cpu", dtype=torch.float64).clone()
    neighbors.fill_diagonal_(0.0)
    if not bool(torch.isfinite(neighbors).all()) or bool(((neighbors < 0) | (neighbors > 1)).any()):
        raise ValueError("Off-diagonal GAC kernels must be finite and in [0, 1]")
    neighbors.mul_(beta / (count - 1))
    cross_coefficients = neighbors.sum(dim=0).tolist()
    self_coefficients = [1.0 - value for value in cross_coefficients]
    neighbor_coefficients = neighbors.tolist()

    mixed_tasks: list[list[torch.Tensor]] = []
    for target in range(count):
        current = []
        for parameter_index, local in enumerate(task_gradients[target]):
            value = local.float() * self_coefficients[target]
            for source in range(count):
                if source == target:
                    continue
                coefficient = neighbor_coefficients[source][target]
                if coefficient:
                    value.add_(task_gradients[source][parameter_index].float(), alpha=coefficient)
            current.append(value)
        mixed_tasks.append(current)

    task_norms = tuple(float(vector_norm(values).item()) for values in task_gradients)
    mixed_norms = tuple(float(vector_norm(values).item()) for values in mixed_tasks)
    raw_norms = tuple(float(vector_norm(values).item()) for values in repulsion)
    cap_factors = tuple(
        min(1.0, mixed / (raw + 1.0e-12))
        for mixed, raw in zip(mixed_norms, raw_norms, strict=True)
    )
    final = [
        [mixed - rbf_weight * factor * outward.float()
         for mixed, outward in zip(current, directions, strict=True)]
        for current, directions, factor in zip(mixed_tasks, repulsion, cap_factors, strict=True)
    ]
    capped_norms = tuple(raw * cap for raw, cap in zip(raw_norms, cap_factors, strict=True))
    return final, GACDiagnostics(
        task_norms=task_norms,
        mixed_task_norms=mixed_norms,
        cross_coefficients=tuple(cross_coefficients),
        self_coefficients=tuple(self_coefficients),
        repulsion_norms_before_cap=raw_norms,
        repulsion_cap_factors=cap_factors,
        capped_repulsion_norms=capped_norms,
        weighted_repulsion_norms=tuple(rbf_weight * value for value in capped_norms),
        final_norms=tuple(float(vector_norm(values).item()) for values in final),
    )
