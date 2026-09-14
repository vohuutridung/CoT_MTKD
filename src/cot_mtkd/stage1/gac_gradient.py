from __future__ import annotations

from dataclasses import dataclass

import torch

from ..utils.training import vector_norm


@dataclass(frozen=True)
class GACDiagnostics:
    interaction_scale: float
    task_norms: tuple[float, ...]
    dpp_norms: tuple[float, ...]
    repulsion_norms_before_cap: tuple[float, ...]
    repulsion_cap_factors: tuple[float, ...]
    final_norms: tuple[float, ...]


def stable_gac_gradients(
    sft_gradients: list[list[torch.Tensor]],
    dpp_gradients: list[list[torch.Tensor]],
    repulsion: list[list[torch.Tensor]],
    kernel: torch.Tensor,
    interaction: float,
    dpp_weight: float,
    rbf_weight: float,
) -> tuple[list[list[torch.Tensor]], GACDiagnostics]:
    count = len(sft_gradients)
    if not (
        len(dpp_gradients)
        == len(repulsion)
        == count
        == kernel.shape[0]
        == kernel.shape[1]
    ):
        raise ValueError("Inconsistent expert dimensions in GAC inputs")
    weights = kernel.float() / kernel.float().sum(dim=0, keepdim=True).clamp_min(
        1.0e-12
    )
    mixed_tasks: list[list[torch.Tensor]] = []
    for target in range(count):
        current: list[torch.Tensor] = []
        for parameter_index in range(len(sft_gradients[target])):
            value = torch.zeros_like(
                sft_gradients[target][parameter_index], dtype=torch.float32
            )
            for source in range(count):
                task = sft_gradients[source][parameter_index].float()
                diversity = dpp_gradients[source][parameter_index].float()
                value.add_(
                    task + dpp_weight * diversity, alpha=float(weights[source, target])
                )
            current.append(value)
        mixed_tasks.append(current)

    capped_repulsion: list[list[torch.Tensor]] = []
    repulsion_norms: list[float] = []
    cap_factors: list[float] = []
    for expert in range(count):
        reference_norm = float(vector_norm(mixed_tasks[expert]).item())
        repulsion_norm = float(vector_norm(repulsion[expert]).item())
        factor = min(1.0, reference_norm / (repulsion_norm + 1.0e-12))
        capped_repulsion.append([value.float() * factor for value in repulsion[expert]])
        repulsion_norms.append(repulsion_norm)
        cap_factors.append(factor)

    final: list[list[torch.Tensor]] = []
    for expert in range(count):
        current = []
        for local, mixed, repel in zip(
            sft_gradients[expert],
            mixed_tasks[expert],
            capped_repulsion[expert],
            strict=True,
        ):
            full = mixed - rbf_weight * repel
            current.append((1.0 - interaction) * local.float() + interaction * full)
        final.append(current)
    diagnostics = GACDiagnostics(
        interaction_scale=float(interaction),
        task_norms=tuple(float(vector_norm(values).item()) for values in sft_gradients),
        dpp_norms=tuple(float(vector_norm(values).item()) for values in dpp_gradients),
        repulsion_norms_before_cap=tuple(repulsion_norms),
        repulsion_cap_factors=tuple(cap_factors),
        final_norms=tuple(float(vector_norm(values).item()) for values in final),
    )
    return final, diagnostics
