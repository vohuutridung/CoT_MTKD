from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class GeometryWeights:
    agreement: float
    utilities: list[float]
    mean_utility: float
    step_weight: float
    teacher_weights: torch.Tensor
    consensus_weight: float


def task_anchored_weights(
    teacher_gradients: list[list[torch.Tensor]],
    anchor_gradients: list[torch.Tensor],
    epsilon_a: float = 1.0e-12,
    epsilon_u: float = 1.0e-12,
) -> GeometryWeights:
    """Detached, whole-parameter geometry from proposal equations (3)--(13).

    Parameter blocks stay separate: a flattened copy of the complete LoRA
    gradient is never needed. The raw council mean is unweighted, including
    teachers whose task utility is subsequently nonpositive.
    """
    for name, value in (("epsilon_a", epsilon_a), ("epsilon_u", epsilon_u)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    expert_count = len(teacher_gradients)
    if expert_count < 2:
        raise ValueError("Task-anchored geometry requires at least two teachers")
    if not anchor_gradients:
        raise ValueError("The LoRA gradient must contain at least one parameter block")
    if any(len(gradient) != len(anchor_gradients) for gradient in teacher_gradients):
        raise ValueError("Teacher and anchor parameter blocks are misaligned")

    # Each device accumulates only scalar statistics. Double precision avoids
    # losing the small common direction of nearly opposing raw gradients.
    statistics: dict[torch.device, torch.Tensor] = {}
    with torch.no_grad():
        for block_index, anchor in enumerate(anchor_gradients):
            current = [gradient[block_index] for gradient in teacher_gradients]
            for value in [anchor, *current]:
                if not isinstance(value, torch.Tensor) or not value.is_floating_point():
                    raise TypeError("Gradient blocks must be real floating-point tensors")
                if value.shape != anchor.shape:
                    raise ValueError("Teacher and anchor gradient shapes are misaligned")
                if not bool(torch.isfinite(value).all()):
                    raise ValueError("Gradient blocks must contain only finite values")
            device = anchor.device if anchor.device.type != "mps" else torch.device("cpu")
            h = anchor.detach().to(device=device, dtype=torch.float64)
            mean = torch.zeros_like(h)
            for value in current:
                mean.add_(value.detach().to(device=device, dtype=torch.float64))
            mean.div_(expert_count)
            # [||h||², ||gbar||², mean ||gm-gbar||², <h,gbar>,
            #  ||g1||²,...,||gM||², <h,g1>,...,<h,gM>]
            block = torch.zeros(4 + 2 * expert_count, device=device, dtype=torch.float64)
            block[0] = h.square().sum()
            block[1] = mean.square().sum()
            block[3] = (h * mean).sum()
            for expert_index, value in enumerate(current):
                gradient = value.detach().to(device=device, dtype=torch.float64)
                block[2].add_((gradient - mean).square().sum() / expert_count)
                block[4 + expert_index] = gradient.square().sum()
                block[4 + expert_count + expert_index] = (h * gradient).sum()
            if device not in statistics:
                statistics[device] = block
            else:
                statistics[device].add_(block)
        totals = torch.zeros(4 + 2 * expert_count, dtype=torch.float64)
        for value in statistics.values():
            totals.add_(value.cpu())
    if not bool(torch.isfinite(totals).all()):
        raise ValueError("Gradient geometry overflowed its finite scalar reductions")
    values = totals.tolist()
    anchor_norm = math.sqrt(values[0])
    mean_norm = math.sqrt(values[1])
    common_energy = (expert_count - 1) * values[1]
    agreement = common_energy / (common_energy + values[2] + epsilon_a)
    utilities = [
        values[4 + expert_count + index]
        / (anchor_norm * math.sqrt(values[4 + index]) + epsilon_u)
        for index in range(expert_count)
    ]
    mean_utility = values[3] / (anchor_norm * mean_norm + epsilon_u)
    if not all(math.isfinite(value) for value in [agreement, mean_utility, *utilities]):
        raise ValueError("Gradient geometry produced nonfinite agreement or utility")
    # Cosines can cross a mathematical endpoint by one floating-point ulp.
    # Keep selector weights inside their defined range after that roundoff.
    utilities = [max(-1.0, min(1.0, value)) for value in utilities]
    mean_utility = max(-1.0, min(1.0, mean_utility))
    positive_utilities = [max(value, 0.0) for value in utilities]
    utility_sum = math.fsum(positive_utilities)
    step_weight = utility_sum / expert_count
    if utility_sum == 0.0:
        teacher_weights = torch.zeros(expert_count, dtype=torch.float32)
    else:
        teacher_weights = torch.tensor(
            [value / utility_sum for value in positive_utilities], dtype=torch.float32
        )
    return GeometryWeights(
        agreement=agreement,
        utilities=utilities,
        mean_utility=mean_utility,
        step_weight=step_weight,
        teacher_weights=teacher_weights,
        consensus_weight=agreement * max(mean_utility, 0.0),
    )
