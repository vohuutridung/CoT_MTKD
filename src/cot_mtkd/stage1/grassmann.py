from __future__ import annotations

import math
from collections import OrderedDict

import torch


def _factor_pairs(group: OrderedDict[str, torch.nn.Parameter]):
    for key, a_factor in group.items():
        if "lora_A" not in key:
            continue
        b_key = key.replace("lora_A", "lora_B")
        if b_key not in group:
            raise KeyError(f"Missing LoRA B factor for {key}")
        yield key, a_factor, group[b_key]


def _basis(factor: torch.Tensor, epsilon: float) -> torch.Tensor:
    matrix = factor.float()
    rows, rank = matrix.shape
    if rows < rank:
        raise ValueError("LoRA factor has fewer rows than its rank")
    # PEFT starts every B at zero. A fixed full-rank scaffold keeps QR
    # differentiable until the learned factor itself has full rank. Leave
    # full-rank factors untouched so their subspaces are basis-invariant.
    if float(torch.linalg.svdvals(matrix.detach())[-1].item()) < epsilon:
        scaffold = torch.zeros_like(matrix)
        scaffold[:rank, :] = torch.eye(rank, device=matrix.device, dtype=matrix.dtype)
        matrix = matrix + epsilon * scaffold
    return torch.linalg.qr(matrix, mode="reduced").Q


def _principal_angle_squared(
    left: torch.Tensor, right: torch.Tensor, angle_epsilon: float
) -> torch.Tensor:
    singular = torch.linalg.svdvals(left.T @ right)
    cosine = singular.clamp(min=0.0, max=1.0 - angle_epsilon)
    return torch.arccos(cosine).square().sum()


def grassmann_squared_distances(
    groups: list[OrderedDict[str, torch.nn.Parameter]],
    rank_epsilon: float = 1.0e-4,
    angle_epsilon: float = 1.0e-6,
) -> torch.Tensor:
    """Pairwise LoRA A-row/B-column subspace distances, averaged over modules."""
    if len(groups) < 2:
        raise ValueError("Grassmann repulsion requires at least two experts")
    if rank_epsilon <= 0 or not (0 < angle_epsilon < 1):
        raise ValueError("Grassmann numerical epsilons must be positive")
    reference = list(_factor_pairs(groups[0]))
    module_keys = [key for key, _, _ in reference]
    if not module_keys:
        raise ValueError("No LoRA factor pairs found")
    bases: list[dict[str, tuple[torch.Tensor, torch.Tensor]]] = []
    for group in groups:
        current = list(_factor_pairs(group))
        if [key for key, _, _ in current] != module_keys:
            raise ValueError("Expert LoRA module structures differ")
        bases.append(
            {
                key: (_basis(a.T, rank_epsilon), _basis(b, rank_epsilon))
                for key, a, b in current
            }
        )
    count = len(groups)
    distances = torch.zeros(
        (count, count), device=next(iter(groups[0].values())).device, dtype=torch.float32
    )
    for left in range(count):
        for right in range(left + 1, count):
            module_distances = []
            for key in module_keys:
                a_left, b_left = bases[left][key]
                a_right, b_right = bases[right][key]
                module_distances.append(
                    _principal_angle_squared(a_left, a_right, angle_epsilon)
                    + _principal_angle_squared(b_left, b_right, angle_epsilon)
                )
            value = torch.stack(module_distances).mean()
            distances[left, right] = value
            distances[right, left] = value
    return distances


def grassmann_repulsion_updates(
    groups: list[OrderedDict[str, torch.nn.Parameter]],
    rank_epsilon: float = 1.0e-4,
    angle_epsilon: float = 1.0e-6,
    bandwidth_floor: float = 1.0e-8,
) -> tuple[list[list[torch.Tensor]], torch.Tensor, torch.Tensor, float]:
    """Return F_rep = mean_j K(m,j) grad_m d²(m,j), for direct outward updates."""
    distances = grassmann_squared_distances(groups, rank_epsilon, angle_epsilon)
    count = len(groups)
    upper = torch.triu_indices(count, count, 1, device=distances.device)
    pairwise = distances.detach()[upper[0], upper[1]]
    median = torch.quantile(pairwise.sqrt(), 0.5)
    bandwidth = max(float(median.square().item()) / math.log(count), bandwidth_floor)
    kernel = torch.exp(-distances.detach() / bandwidth)
    updates = []
    for expert, group in enumerate(groups):
        potential = sum(
            kernel[expert, other] * distances[expert, other]
            for other in range(count)
            if other != expert
        ) / count
        parameters = list(group.values())
        gradients = torch.autograd.grad(
            potential, parameters, retain_graph=expert + 1 < count, allow_unused=False
        )
        updates.append([gradient.detach().float() for gradient in gradients])
    return updates, kernel, distances.detach(), bandwidth
