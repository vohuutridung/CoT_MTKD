from __future__ import annotations

import torch
import torch.nn.functional as F


def robust_standardize(
    values: torch.Tensor, mad_scale: float = 1.4826, epsilon: float = 1.0e-6
) -> torch.Tensor:
    if values.numel() == 0:
        return values.float()
    values = values.float()
    median = torch.quantile(values, 0.5)
    mad = torch.quantile((values - median).abs(), 0.5)
    if float(mad.item()) < epsilon:
        return torch.zeros_like(values)
    return (values - median) / (mad_scale * mad)


def group_step_importance(
    pag: torch.Tensor,
    js_disagreement: torch.Tensor,
    token_counts: torch.Tensor,
    consensus_floor: float = 0.5,
    js_scale: float = 1.0,
    clip: tuple[float, float] = (0.25, 4.0),
    mad_scale: float = 1.4826,
    epsilon: float = 1.0e-6,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """PAG + gain-consensus + JS multiplicative importance map."""
    if pag.ndim != 2:
        raise ValueError("PAG must have shape [experts, steps]")
    if pag.shape[1] != js_disagreement.numel() or token_counts.numel() != pag.shape[1]:
        raise ValueError("Step dimensions do not match")
    if pag.shape[1] == 0:
        return torch.empty(0), {
            "group_gain": torch.empty(0),
            "gain_consensus": torch.empty(0),
            "standardized_gain": torch.empty(0),
            "standardized_js": torch.empty(0),
        }
    group_gain = pag.float().median(dim=0).values
    consensus = pag.gt(0).float().mean(dim=0)
    gain_z = robust_standardize(group_gain, mad_scale, epsilon)
    js_z = robust_standardize(js_disagreement.float(), mad_scale, epsilon)
    raw = (
        F.softplus(gain_z)
        * (consensus_floor + (1.0 - consensus_floor) * consensus)
        * (1.0 + js_scale * torch.sigmoid(js_z))
    )
    token_counts = token_counts.float().clamp_min(1.0)
    mean = (raw * token_counts).sum() / token_counts.sum()
    importance = raw / mean.clamp_min(epsilon)
    importance = importance.clamp(min=float(clip[0]), max=float(clip[1]))
    renormalized_mean = (importance * token_counts).sum() / token_counts.sum()
    importance = importance / renormalized_mean.clamp_min(epsilon)
    return importance, {
        "group_gain": group_gain,
        "gain_consensus": consensus,
        "standardized_gain": gain_z,
        "standardized_js": js_z,
    }
