from __future__ import annotations

from typing import Any, Mapping

import torch

from ..data.collator import shifted_token_views
from ..data.schema import TokenRegion


def zscore_within_sample(values: torch.Tensor, epsilon: float = 1.0e-6) -> torch.Tensor:
    """Z-score over the S reasoning steps of one sample (eq:zscore)."""
    values = values.float()
    if values.numel() <= 1:
        return torch.zeros_like(values)
    mean = values.mean()
    std = values.std(unbiased=False)
    if float(std.item()) < float(epsilon):
        return torch.zeros_like(values)
    return (values - mean) / std.clamp_min(epsilon)


def relative_disagreement(
    mean_uncertainty: torch.Tensor,
    disagreement: torch.Tensor,
    epsilon: float = 1.0e-6,
) -> torch.Tensor:
    """rho_i = V_i / max(U_i, eps_U) (eq:rho)."""
    uncertainty = mean_uncertainty.float().clamp_min(epsilon)
    return disagreement.float().clamp_min(0.0) / uncertainty


def step_weights_from_signals(
    mean_uncertainty: torch.Tensor,
    disagreement: torch.Tensor,
    lambda_uncertainty: float,
    lambda_disagreement: float,
    epsilon: float = 1.0e-6,
) -> torch.Tensor:
    """w_i = 1 + λ_U tanh(Û_i) + λ_D tanh(ρ̂_i) (eq:w)."""
    if float(lambda_uncertainty) < 0.0 or float(lambda_disagreement) < 0.0:
        raise ValueError(
            "lambda_uncertainty and lambda_disagreement must be non-negative"
        )
    rho = relative_disagreement(mean_uncertainty, disagreement, epsilon)
    uncertainty_term = torch.tanh(zscore_within_sample(mean_uncertainty, epsilon))
    disagreement_term = torch.tanh(zscore_within_sample(rho, epsilon))
    weights = (
        1.0
        + float(lambda_uncertainty) * uncertainty_term
        + float(lambda_disagreement) * disagreement_term
    )
    return weights.clamp_min(epsilon)


def phase2_token_weights(
    batch: dict[str, Any],
    signals: Mapping[str, Mapping[str, Any]],
    lambda_uncertainty: float,
    lambda_disagreement: float,
    epsilon: float = 1.0e-6,
) -> tuple[torch.Tensor, float]:
    """Length-normalized NLL weights for proposal Phase 2.

    Reasoning step s contributes w_s * L_s, where L_s is mean NLL over the
    step's tokens. Answer and format/control tokens are always-on blocks
    with weight 1, matching the notation that the final answer is compared
    on a shared token set and format tokens are not reasoning steps.

    Returns token weights in `response_targets` order and the denominator
    Σ_s w_s + 1_ans + 1_fixed over the micro-batch.
    """
    views = shifted_token_views(batch)
    valid = views["valid"]
    batch_ids = views["response_batch_indices"]
    regions = views["regions"][valid]
    step_ids = views["steps"][valid]
    weights = torch.zeros(
        batch_ids.numel(), device=batch_ids.device, dtype=torch.float32
    )
    denominator = 0.0
    reasoning = regions.eq(int(TokenRegion.REASONING))
    answer = regions.eq(int(TokenRegion.ANSWER))
    for row, sample_id in enumerate(batch["sample_ids"]):
        sample_mask = batch_ids.eq(row)
        record = signals[sample_id]
        uncertainty = torch.as_tensor(
            record["mean_uncertainty"], device=batch_ids.device, dtype=torch.float32
        )
        disagreement = torch.as_tensor(
            record["disagreement"], device=batch_ids.device, dtype=torch.float32
        )
        if uncertainty.numel() != disagreement.numel():
            raise ValueError(f"U/V step counts differ for {sample_id}")
        if (
            not torch.isfinite(uncertainty).all()
            or not torch.isfinite(disagreement).all()
        ):
            raise ValueError(f"Non-finite council signals for {sample_id}")
        if (disagreement < 0).any():
            raise ValueError(f"Disagreement must be non-negative for {sample_id}")
        step_w = step_weights_from_signals(
            uncertainty,
            disagreement,
            lambda_uncertainty,
            lambda_disagreement,
            epsilon,
        )
        sample_reasoning = sample_mask & reasoning
        if sample_reasoning.any():
            selected_steps = step_ids[sample_reasoning]
            if (
                int(selected_steps.min().item()) < 0
                or int(selected_steps.max().item()) >= step_w.numel()
            ):
                raise RuntimeError(f"Invalid reasoning step id in {sample_id}")
            unique_steps = torch.unique(selected_steps)
            for step in unique_steps.tolist():
                mask = sample_reasoning & step_ids.eq(int(step))
                count = int(mask.sum().item())
                weights[mask] = float(step_w[int(step)].item()) / float(count)
                denominator += float(step_w[int(step)].item())
        for region_mask in (answer, ~reasoning & ~answer):
            mask = sample_mask & region_mask
            if mask.any():
                weights[mask] = 1.0 / float(int(mask.sum().item()))
                denominator += 1.0
    return weights, denominator
