from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

import torch


def cosine_warmup_lambda(
    step: int, total_steps: int, warmup_ratio: float, min_lr_ratio: float = 0.0
) -> float:
    if total_steps <= 0:
        return 1.0
    warmup_steps = max(1, round(total_steps * warmup_ratio))
    if step < warmup_steps:
        return float(step + 1) / float(warmup_steps)
    progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


def interaction_scale(
    progress: float, off_until: float = 0.1, ramp_until: float = 0.3
) -> float:
    if progress < off_until:
        return 0.0
    if progress >= ramp_until:
        return 1.0
    return (progress - off_until) / max(ramp_until - off_until, 1.0e-12)


def zeros_like_parameters(
    parameters: Sequence[torch.nn.Parameter], dtype: torch.dtype = torch.float32
):
    return [
        torch.zeros_like(parameter, dtype=dtype, memory_format=torch.preserve_format)
        for parameter in parameters
    ]


def add_gradients_(
    destination: Sequence[torch.Tensor], source: Sequence[torch.Tensor]
) -> None:
    for output, value in zip(destination, source, strict=True):
        output.add_(value)


def divide_gradients_(gradients: Sequence[torch.Tensor], denominator: float) -> None:
    scale = 1.0 / max(float(denominator), 1.0)
    for gradient in gradients:
        gradient.mul_(scale)


def vector_norm(tensors: Iterable[torch.Tensor]) -> torch.Tensor:
    values = [tensor.float().pow(2).sum() for tensor in tensors]
    if not values:
        return torch.tensor(0.0)
    return torch.stack(values).sum().sqrt()


def assign_gradients(
    parameters: Sequence[torch.nn.Parameter], gradients: Sequence[torch.Tensor]
) -> None:
    for parameter, gradient in zip(parameters, gradients, strict=True):
        parameter.grad = gradient.to(device=parameter.device, dtype=parameter.dtype)


def global_clip_grad_list_(gradients: Sequence[torch.Tensor], max_norm: float) -> float:
    norm = vector_norm(gradients)
    coefficient = min(1.0, max_norm / (float(norm.item()) + 1.0e-12))
    if coefficient < 1.0:
        for gradient in gradients:
            gradient.mul_(coefficient)
    return float(norm.item())
