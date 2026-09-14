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


def repulsion_updates(
    groups: list[OrderedDict[str, torch.nn.Parameter]], scaling: float, bandwidth: float
) -> tuple[list[list[torch.Tensor]], torch.Tensor, torch.Tensor]:
    """Return outward update directions `-grad(sum pairwise RBF)` for each expert."""
    distances = effective_update_distances(groups, scaling)
    kernel = rbf_kernel(distances, bandwidth)
    count = len(groups)
    upper = torch.triu_indices(count, count, offset=1, device=kernel.device)
    potential = kernel[upper[0], upper[1]].mean()
    flat_parameters = [parameter for group in groups for parameter in group.values()]
    gradients = torch.autograd.grad(potential, flat_parameters, allow_unused=False)
    updates: list[list[torch.Tensor]] = []
    cursor = 0
    for group in groups:
        current: list[torch.Tensor] = []
        for _ in group.values():
            current.append(-gradients[cursor].detach().float())
            cursor += 1
        updates.append(current)
    return updates, kernel.detach(), distances.detach()
