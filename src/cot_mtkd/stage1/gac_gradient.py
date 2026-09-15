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


@dataclass(frozen=True)
class Phase1Diagnostics:
    task_norms: tuple[float, ...]
    dpp_norms: tuple[float, ...]
    repulsion_norms: tuple[float, ...]
    final_norms: tuple[float, ...]


def apply_grassmann_force_(
    parameters: list[torch.nn.Parameter],
    outward_force: list[torch.Tensor],
    learning_rate: float,
    repulsion_weight: float,
) -> None:
    """Apply Eq. (23)'s +eta*lambda_rep*F_rep after the data optimizer step."""
    if len(parameters) != len(outward_force):
        raise ValueError("Grassmann force does not match LoRA parameters")
    with torch.no_grad():
        for parameter, force in zip(parameters, outward_force, strict=True):
            parameter.add_(
                force.to(device=parameter.device, dtype=parameter.dtype),
                alpha=learning_rate * repulsion_weight,
            )


def phase1_data_gradients(
    task: list[list[torch.Tensor]],
    diversity: list[list[torch.Tensor]],
    outward_force: list[list[torch.Tensor]],
    diversity_weight: float,
) -> tuple[list[list[torch.Tensor]], Phase1Diagnostics]:
    """The data-loss gradient of Eq. (23), kept separate from Grassmann force."""
    if not len(task) == len(diversity) == len(outward_force):
        raise ValueError("Inconsistent expert dimensions in Phase 1 gradients")
    final = []
    for local, dpp, force in zip(task, diversity, outward_force, strict=True):
        if not len(local) == len(dpp) == len(force):
            raise ValueError("Inconsistent LoRA parameter dimensions")
        final.append(
            [
                sft.float() + diversity_weight * div.float()
                for sft, div in zip(local, dpp, strict=True)
            ]
        )
    return final, Phase1Diagnostics(
        task_norms=tuple(float(vector_norm(values).item()) for values in task),
        dpp_norms=tuple(float(vector_norm(values).item()) for values in diversity),
        repulsion_norms=tuple(float(vector_norm(values).item()) for values in outward_force),
        final_norms=tuple(float(vector_norm(values).item()) for values in final),
    )


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
