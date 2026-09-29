"""Closed-form projection-distance repulsion for Phase-1 LoRA experts.

Experts are separate PEFT adapters. This module stacks their A/B factors by
shape, builds r x r Gram blocks with one GEMM per shape group, and returns a
force in the same parameter order as ``adapter_parameter_groups``. The legacy
geodesic path stays in ``grassmann.py`` and is selected with
``rep_metric="geodesic_autograd"``.
"""

from __future__ import annotations

import math
import time
from collections import OrderedDict
from dataclasses import dataclass

import torch

from .grassmann import grassmann_repulsion_updates


@dataclass(frozen=True)
class RepulsionStats:
    updates: list[list[torch.Tensor]]
    kernel: torch.Tensor
    distances: torch.Tensor
    bandwidth: float
    mean_f: float
    min_f: float
    max_f: float
    b_active: bool
    seconds: float


def b_side_active(
    step: int,
    min_b_norm: float,
    start_step: int,
    min_norm: float,
) -> bool:
    """Turn on col(B) only after ``start_step`` and once every expert has left 0."""
    return int(step) >= int(start_step) and float(min_b_norm) > float(min_norm)


def gram_stats(
    factors: torch.Tensor,
    eps_rel: float = 1.0e-4,
    eps_abs: float = 1.0e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batched Gram statistics for ``factors`` of shape ``(L, M, r, d)``.

    Returns ``Ct_inv (L,M,r,r)``, ``C (L,M,M,r,r)``, ``T (L,M,M,r,r)``,
    ``f (L,M,M)``. Every inverse uses the relative ridge
    ``C + (eps_rel * tr(C)/r + eps_abs) I``.
    """
    if factors.ndim != 4:
        raise ValueError("factors must have shape (L, M, r, d)")
    with torch.no_grad():
        values = factors.detach()
        if values.dtype != torch.float64:
            values = values.float()
        layers, experts, rank, _width = values.shape
        if rank <= 0 or experts < 1:
            raise ValueError("rank and expert count must be positive")
        flat = values.reshape(layers, experts * rank, values.shape[-1])
        gram = flat @ flat.transpose(1, 2)
        blocks = gram.reshape(layers, experts, rank, experts, rank)
        cross = blocks.permute(0, 1, 3, 2, 4).contiguous()
        diagonal = cross[:, torch.arange(experts), torch.arange(experts)]
        trace = diagonal.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        ridge = float(eps_rel) * trace / float(rank) + float(eps_abs)
        eye = torch.eye(rank, device=values.device, dtype=values.dtype)
        regularized = diagonal + ridge[..., None, None] * eye
        inverse = torch.linalg.inv(regularized)
        transport = cross @ inverse[:, None, :, :, :]
        overlap = inverse[:, :, None] @ transport @ cross.transpose(1, 2)
        score = overlap.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        return inverse, cross, transport, score


def pairwise_d2(
    overlap_a: torch.Tensor,
    overlap_b: torch.Tensor | None,
    rank: int,
    n_modules: int,
) -> torch.Tensor:
    """Mean projection distance. ``overlap_*`` is ``(L, M, M)``."""
    if n_modules <= 0:
        raise ValueError("n_modules must be positive")
    distance_a = float(rank) - overlap_a
    total = distance_a.sum(dim=0)
    if overlap_b is not None:
        total = total + (float(rank) - overlap_b).sum(dim=0)
    return total / float(n_modules)


def kernel_from_d2(
    distances: torch.Tensor,
    bandwidth_floor: float = 1.0e-8,
) -> tuple[torch.Tensor, float]:
    """Median-heuristic Gaussian kernel. Diagonal is zero. ``h <= 0`` is floored."""
    count = distances.shape[0]
    if count < 2:
        return torch.zeros_like(distances), 0.0
    upper = distances[torch.triu(torch.ones(count, count, dtype=torch.bool, device=distances.device), diagonal=1)]
    median = upper.clamp_min(0.0).sqrt().median()
    bandwidth = float(median.square().item()) / math.log(count)
    if bandwidth <= 0.0:
        bandwidth = float(bandwidth_floor)
    kernel = torch.exp(-distances / bandwidth)
    kernel = kernel.clone()
    kernel.fill_diagonal_(0.0)
    return kernel.detach(), bandwidth


def repulsion_force(
    factors: torch.Tensor,
    stats: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    kernel: torch.Tensor,
    n_modules: int,
) -> torch.Tensor:
    """Closed-form force on ``X``, same shape ``(L, M, r, d)``. Ascends ``d^2``."""
    if n_modules <= 0:
        raise ValueError("n_modules must be positive")
    inverse, cross, transport, _score = stats
    with torch.no_grad():
        values = factors.detach()
        if values.dtype != torch.float64:
            values = values.float()
        _layers, experts, _rank, _width = values.shape
        solved = inverse @ values
        cross_qp = cross.permute(0, 2, 1, 3, 4)
        subtracted = cross_qp @ solved[:, :, None, :, :]
        residual = values[:, None, :, :, :] - subtracted
        mapped = inverse[:, :, None, :, :] @ transport
        gradient = 2.0 * (mapped @ residual)
        weights = kernel.to(dtype=gradient.dtype, device=gradient.device)
        weighted = weights[None, :, :, None, None] * gradient
        scale = -1.0 / (float(experts) * float(n_modules))
        return scale * weighted.sum(dim=2)


def _factor_keys(group: OrderedDict[str, torch.nn.Parameter]) -> list[tuple[str, str]]:
    pairs = []
    for key in group:
        if "lora_A" not in key:
            continue
        b_key = key.replace("lora_A", "lora_B")
        if b_key not in group:
            raise KeyError(f"Missing LoRA B factor for {key}")
        pairs.append((key, b_key))
    if not pairs:
        raise ValueError("No LoRA factor pairs found")
    return pairs


def _min_b_norm(groups: list[OrderedDict[str, torch.nn.Parameter]]) -> float:
    keys = _factor_keys(groups[0])
    norms = []
    for group in groups:
        total = torch.zeros((), dtype=torch.float32)
        for _a_key, b_key in keys:
            factor = group[b_key].detach().float()
            total = total + factor.pow(2).sum()
        norms.append(float(total.sqrt().item()))
    return min(norms)


def _stack_side(
    groups: list[OrderedDict[str, torch.nn.Parameter]],
    keys: list[str],
    *,
    transpose: bool,
) -> dict[tuple[int, ...], tuple[torch.Tensor, list[str]]]:
    buckets: dict[tuple[int, ...], list[tuple[str, list[torch.Tensor]]]] = {}
    for key in keys:
        samples = [group[key].detach() for group in groups]
        matrix = samples[0]
        shape = tuple(matrix.transpose(0, 1).shape if transpose else matrix.shape)
        buckets.setdefault(shape, []).append((key, samples))
    stacked: dict[tuple[int, ...], tuple[torch.Tensor, list[str]]] = {}
    for shape, items in buckets.items():
        layers = []
        names = []
        for key, samples in items:
            expert_stack = []
            for sample in samples:
                current = sample.float()
                if transpose:
                    current = current.transpose(0, 1)
                expert_stack.append(current.contiguous())
            layers.append(torch.stack(expert_stack, dim=0))
            names.append(key)
        stacked[shape] = (torch.stack(layers, dim=0), names)
    return stacked


def projection_repulsion_updates(
    groups: list[OrderedDict[str, torch.nn.Parameter]],
    *,
    eps_rel: float = 1.0e-4,
    eps_abs: float = 1.0e-8,
    b_active: bool = False,
    bandwidth_floor: float = 1.0e-8,
) -> RepulsionStats:
    """Projection-distance force from one consistent parameter snapshot."""
    started = time.perf_counter()
    if len(groups) < 2:
        raise ValueError("Projection repulsion requires at least two experts")
    pairs = _factor_keys(groups[0])
    for group in groups[1:]:
        if _factor_keys(group) != pairs:
            raise ValueError("Expert LoRA module structures differ")
    n_modules = len(pairs)
    a_keys = [a_key for a_key, _b_key in pairs]
    b_keys = [b_key for _a_key, b_key in pairs]
    stored_a = [
        (factors, names, gram_stats(factors, eps_rel, eps_abs))
        for factors, names in _stack_side(groups, a_keys, transpose=False).values()
    ]
    overlap_a_cat = torch.cat([stats[3] for _factors, _names, stats in stored_a], dim=0)
    stored_b: list[tuple[torch.Tensor, list[str], tuple]] = []
    overlap_b_cat = None
    if b_active:
        stored_b = [
            (factors, names, gram_stats(factors, eps_rel, eps_abs))
            for factors, names in _stack_side(groups, b_keys, transpose=True).values()
        ]
        overlap_b_cat = torch.cat([stats[3] for _factors, _names, stats in stored_b], dim=0)
    rank = int(stored_a[0][0].shape[2])
    distances = pairwise_d2(overlap_a_cat, overlap_b_cat, rank, n_modules)
    kernel, bandwidth = kernel_from_d2(distances, bandwidth_floor)
    stacked_forces: dict[str, torch.Tensor] = {}
    for factors, names, stats in stored_a:
        update = repulsion_force(factors, stats, kernel, n_modules)
        for layer, key in enumerate(names):
            stacked_forces[key] = update[layer]
    for factors, names, stats in stored_b:
        update = repulsion_force(factors, stats, kernel, n_modules)
        for layer, key in enumerate(names):
            stacked_forces[key] = update[layer].transpose(-1, -2).contiguous()
    updates = []
    for expert, group in enumerate(groups):
        current = []
        for key in group:
            value = stacked_forces.get(key)
            if (
                value is not None
                and value.shape[0] == len(groups)
                and value.shape[1:] == group[key].shape
            ):
                current.append(value[expert].detach().float())
            else:
                current.append(torch.zeros_like(group[key], dtype=torch.float32))
        updates.append(current)
    off = torch.triu(torch.ones_like(distances, dtype=torch.bool), diagonal=1)
    pair_f = overlap_a_cat
    if overlap_b_cat is not None:
        pair_f = torch.cat([overlap_a_cat, overlap_b_cat], dim=0)
    pair_values = pair_f[:, off].reshape(-1) if pair_f.numel() else pair_f.reshape(-1)
    finite = torch.isfinite(pair_values).all() and all(
        torch.isfinite(item).all() for expert in updates for item in expert
    )
    if not finite:
        raise FloatingPointError("Projection repulsion produced a non-finite value")
    elapsed = time.perf_counter() - started
    return RepulsionStats(
        updates=updates,
        kernel=kernel.detach(),
        distances=distances.detach(),
        bandwidth=bandwidth,
        mean_f=float(pair_values.mean().item()) if pair_values.numel() else 0.0,
        min_f=float(pair_values.min().item()) if pair_values.numel() else 0.0,
        max_f=float(pair_values.max().item()) if pair_values.numel() else 0.0,
        b_active=b_active,
        seconds=elapsed,
    )


def compute_repulsion(
    groups: list[OrderedDict[str, torch.nn.Parameter]],
    *,
    metric: str,
    step: int,
    eps_rel: float,
    eps_abs: float,
    b_start_step: int,
    b_min_norm: float,
    bandwidth_floor: float,
    rank_epsilon: float,
    angle_epsilon: float,
) -> RepulsionStats:
    """Dispatch the closed-form projection force or the legacy geodesic force."""
    if metric == "projection_closed_form":
        active = b_side_active(step, _min_b_norm(groups), b_start_step, b_min_norm)
        return projection_repulsion_updates(
            groups,
            eps_rel=eps_rel,
            eps_abs=eps_abs,
            b_active=active,
            bandwidth_floor=bandwidth_floor,
        )
    if metric == "geodesic_autograd":
        started = time.perf_counter()
        updates, kernel, distances, bandwidth = grassmann_repulsion_updates(
            groups,
            rank_epsilon=rank_epsilon,
            angle_epsilon=angle_epsilon,
            bandwidth_floor=bandwidth_floor,
        )
        return RepulsionStats(
            updates=updates,
            kernel=kernel.detach(),
            distances=distances.detach(),
            bandwidth=bandwidth,
            mean_f=float("nan"),
            min_f=float("nan"),
            max_f=float("nan"),
            b_active=True,
            seconds=time.perf_counter() - started,
        )
    raise ValueError(
        f"Unknown rep_metric {metric!r}; expected projection_closed_form or geodesic_autograd"
    )
