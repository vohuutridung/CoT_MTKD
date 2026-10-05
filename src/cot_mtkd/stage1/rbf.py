from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass

import torch


def _factor_pairs(group: OrderedDict[str, torch.nn.Parameter]):
    for key, a_factor in group.items():
        if "lora_A" not in key:
            continue
        b_key = key.replace("lora_A", "lora_B")
        if b_key not in group:
            raise KeyError(f"Missing LoRA B factor for {key}")
        yield key, a_factor, group[b_key]


def low_rank_squared_distance(
    a_i: torch.Tensor,
    b_i: torch.Tensor,
    a_j: torch.Tensor,
    b_j: torch.Tensor,
    scaling: float,
) -> torch.Tensor:
    """Exact ||scale*B_i A_i - scale*B_j A_j||_F^2 without materialization."""
    a_i32, b_i32 = a_i.float(), b_i.float()
    a_j32, b_j32 = a_j.float(), b_j.float()
    norm_i = torch.sum((b_i32.T @ b_i32) * (a_i32 @ a_i32.T))
    norm_j = torch.sum((b_j32.T @ b_j32) * (a_j32 @ a_j32.T))
    inner = torch.sum((b_i32.T @ b_j32) * (a_i32 @ a_j32.T))
    return (float(scaling) ** 2) * (norm_i + norm_j - 2.0 * inner).clamp_min(0.0)


def effective_update_distances(
    groups: list[OrderedDict[str, torch.nn.Parameter]], scaling: float
) -> torch.Tensor:
    count = len(groups)
    distance = torch.zeros((count, count), device=next(iter(groups[0].values())).device)
    reference_pairs = list(_factor_pairs(groups[0]))
    module_keys = [item[0] for item in reference_pairs]
    for left in range(count):
        for right in range(left + 1, count):
            module_distances: list[torch.Tensor] = []
            for a_key in module_keys:
                b_key = a_key.replace("lora_A", "lora_B")
                a_i, b_i = groups[left][a_key], groups[left][b_key]
                a_j, b_j = groups[right][a_key], groups[right][b_key]
                squared = low_rank_squared_distance(a_i, b_i, a_j, b_j, scaling)
                element_count = b_i.shape[0] * a_i.shape[1]
                module_distances.append(squared / float(element_count))
            value = torch.stack(module_distances).mean()
            distance[left, right] = value
            distance[right, left] = value
    return distance


def rbf_kernel(
    distance_squared: torch.Tensor, bandwidth: torch.Tensor | float
) -> torch.Tensor:
    if isinstance(bandwidth, torch.Tensor):
        bandwidth = bandwidth.to(distance_squared.device, dtype=torch.float32)
    return torch.exp(-distance_squared.float() / bandwidth)


def interaction_bandwidths(
    base_bandwidth: float,
    gac_scale: float = 0.5,
    rbf_scale: float = 1.0,
) -> tuple[float, float]:
    """Scale the already-floored EMA bandwidth; GAC is no wider than RBF.

    Distances already are normalized squared Frobenius quantities. Kernels
    consume them directly as exp(-D/h), without another square or bandwidth floor.
    """
    if not math.isfinite(base_bandwidth) or base_bandwidth <= 0.0:
        raise ValueError("Base bandwidth must be finite and positive")
    if not (math.isfinite(gac_scale) and math.isfinite(rbf_scale)
            and 0.0 < gac_scale <= rbf_scale):
        raise ValueError("Bandwidth scales require 0 < gac_bandwidth_scale <= rbf_bandwidth_scale")
    h_gac, h_rbf = base_bandwidth * gac_scale, base_bandwidth * rbf_scale
    if not (math.isfinite(h_gac) and math.isfinite(h_rbf) and h_gac > 0 and h_rbf > 0):
        raise ValueError("Scaled bandwidths must be finite and positive")
    return h_gac, h_rbf


@dataclass
class BandwidthEMA:
    decay: float = 0.9
    floor: float = 1.0e-12
    value: float | None = None

    def update(self, distance_squared: torch.Tensor) -> float:
        count = distance_squared.shape[0]
        pairwise = distance_squared.detach()[
            torch.triu_indices(count, count, offset=1).unbind()
        ]
        estimate = (
            self.floor
            if pairwise.numel() == 0
            else float(torch.quantile(pairwise.float(), 0.5).item())
            / math.log(count + 1)
        )
        estimate = max(estimate, self.floor)
        self.value = (
            estimate
            if self.value is None
            else self.decay * self.value + (1.0 - self.decay) * estimate
        )
        return max(self.value, self.floor)

    def state_dict(self) -> dict[str, float | None]:
        return {"decay": self.decay, "floor": self.floor, "value": self.value}

    def load_state_dict(self, value: dict[str, float | None]) -> None:
        self.decay = float(value["decay"])
        self.floor = float(value["floor"])
        self.value = None if value.get("value") is None else float(value["value"])


def rbf_repulsion_loss(
    distances: torch.Tensor, bandwidth: torch.Tensor | float
) -> torch.Tensor:
    """Mean unordered-pair kernel potential; minimize it to separate updates.

    Bandwidth is held constant during differentiation, including when the
    caller supplies a tensor estimated from current effective-update distances.
    """
    if distances.ndim != 2 or distances.shape[0] != distances.shape[1] or distances.shape[0] < 2:
        raise ValueError("RBF repulsion requires a square distance matrix for at least two experts")
    fixed_bandwidth = float(bandwidth.detach().item()) if isinstance(bandwidth, torch.Tensor) else float(bandwidth)
    if not math.isfinite(fixed_bandwidth) or fixed_bandwidth <= 0.0:
        raise ValueError("RBF bandwidth must be finite and positive")
    kernel = rbf_kernel(distances, fixed_bandwidth)
    upper = torch.triu_indices(kernel.shape[0], kernel.shape[0], offset=1, device=kernel.device)
    return kernel[upper[0], upper[1]].mean()


@dataclass(frozen=True)
class RBFGradients:
    gradients: list[list[torch.Tensor]]
    loss: float
    kernel: torch.Tensor
    distances: torch.Tensor


def rbf_repulsion_gradients(
    groups: list[OrderedDict[str, torch.nn.Parameter]],
    scaling: float,
    bandwidth: float,
    distances: torch.Tensor | None = None,
) -> RBFGradients:
    """Differentiate the scalar RBF loss into each expert's own A/B factors.

    Returns positive loss gradients for addition to the SFT/DPP gradients.
    No full-size BA matrix, task-gradient sharing, or special norm cap is used.
    """
    if distances is None:
        distances = effective_update_distances(groups, scaling)
    kernel = rbf_kernel(distances, bandwidth)
    potential = rbf_repulsion_loss(distances, bandwidth)
    flat_parameters = [parameter for group in groups for parameter in group.values()]
    gradients = torch.autograd.grad(potential, flat_parameters, allow_unused=False)
    grouped_gradients: list[list[torch.Tensor]] = []
    cursor = 0
    for group in groups:
        current: list[torch.Tensor] = []
        for _ in group.values():
            current.append(gradients[cursor].detach().float())
            cursor += 1
        grouped_gradients.append(current)
    return RBFGradients(grouped_gradients, float(potential.detach().item()), kernel.detach(), distances.detach())


def repulsion_updates(
    groups: list[OrderedDict[str, torch.nn.Parameter]],
    scaling: float,
    bandwidth: float,
    distances: torch.Tensor | None = None,
) -> tuple[list[list[torch.Tensor]], torch.Tensor, torch.Tensor]:
    """Return outward directions -grad(mean pairwise RBF) for the GAC update.

    GAC subtracts these directions from its descent gradient after norm capping.
    """
    result = rbf_repulsion_gradients(groups, scaling, bandwidth, distances=distances)
    updates = [[-value for value in current] for current in result.gradients]
    return updates, result.kernel, result.distances
