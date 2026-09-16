from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DPPMetrics:
    groups: int
    samples: int
    cholesky_fallbacks: int
    maximum_jitter: float


def normalized_support_features(
    support_log_probabilities: torch.Tensor,
    support_mask: torch.Tensor,
    epsilon: float = 1.0e-12,
) -> torch.Tensor:
    """L2-normalize full-vocabulary softmax probabilities restricted to C.

    `support_log_probabilities` is `[experts, tokens, support]` with
    log p_c = z_c - logsumexp(z_full): the candidate slice of a softmax
    over the whole vocabulary, not a softmax renormalized on C.
    `support_mask` is `[tokens, support]`. Padding coordinates are zero.
    """
    mask = support_mask.unsqueeze(0).to(device=support_log_probabilities.device)
    # Subtract the per-row max of log p_C for overflow safety. This is
    # equivalent to L2-normalizing p_full[C] because L2 is scale-invariant.
    masked = support_log_probabilities.float().masked_fill(~mask, -torch.inf)
    maximum = masked.max(dim=-1, keepdim=True).values
    maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
    values = torch.exp(masked - maximum).masked_fill(~mask, 0.0)
    norm = values.pow(2).sum(dim=-1, keepdim=True).sqrt().clamp_min(epsilon)
    return values / norm


def _batched_cholesky_logdet(
    grams: torch.Tensor, initial_jitter: float, maximum_jitter: float
) -> tuple[torch.Tensor, float, int]:
    identity = torch.eye(grams.shape[-1], device=grams.device, dtype=torch.float32)
    jitter = torch.full(
        (grams.shape[0],), initial_jitter, device=grams.device, dtype=torch.float32
    )
    fallbacks = 0
    while True:
        regularized = grams.float() + jitter[:, None, None] * identity
        factors, info = torch.linalg.cholesky_ex(regularized)
        failed = info.ne(0)
        if not failed.any():
            return (
                2.0 * torch.log(torch.diagonal(factors, dim1=-2, dim2=-1)).sum(dim=-1),
                float(jitter.max().item()),
                fallbacks,
            )
        fallbacks += int(failed.sum().item())
        if bool((jitter[failed] >= maximum_jitter).all()):
            sign, logabsdet = torch.linalg.slogdet(regularized)
            if (sign <= 0).any():
                raise FloatingPointError(
                    "DPP Gram matrix remains non-positive after maximum jitter"
                )
            return logabsdet, float(jitter.max().item()), fallbacks
        jitter = torch.where(failed, (jitter * 10.0).clamp(max=maximum_jitter), jitter)


def step_dpp_loss(
    features: torch.Tensor,
    sample_ids: torch.Tensor,
    step_ids: torch.Tensor,
    jitter: float = 1.0e-4,
    maximum_jitter: float = 1.0e-2,
    reduction: str = "mean",
) -> tuple[torch.Tensor, DPPMetrics]:
    """Token-wise negative logdet, averaged within steps and then samples."""
    if features.ndim != 3:
        raise ValueError("features must have shape [experts, tokens, dimensions]")
    if sample_ids.numel() != features.shape[1] or step_ids.numel() != features.shape[1]:
        raise ValueError("Token metadata does not match DPP features")
    if (step_ids < 0).any():
        raise ValueError("DPP received a non-reasoning token")
    pair_ids = torch.stack([sample_ids.long(), step_ids.long()], dim=-1)
    unique_pairs = torch.unique(pair_ids, dim=0)
    per_sample: dict[int, list[torch.Tensor]] = {}
    if not unique_pairs.numel():
        zero = features.sum() * 0.0
        return zero, DPPMetrics(0, 0, 0, jitter)
    grams = torch.einsum("mtk,ntk->tmn", features.float(), features.float())
    logdets, used_jitter, fallback_count = _batched_cholesky_logdet(
        grams, jitter, maximum_jitter
    )
    token_losses = -logdets
    for pair in unique_pairs:
        mask = (pair_ids == pair).all(dim=-1)
        per_sample.setdefault(int(pair[0].item()), []).append(
            token_losses[mask].mean()
        )
    sample_losses = [torch.stack(losses).mean() for losses in per_sample.values()]
    stacked = torch.stack(sample_losses)
    if reduction == "mean":
        result = stacked.mean()
    elif reduction == "sum":
        result = stacked.sum()
    elif reduction == "none":
        result = stacked
    else:
        raise ValueError(f"Unknown DPP reduction: {reduction}")
    return result, DPPMetrics(
        groups=len(unique_pairs),
        samples=len(sample_losses),
        cholesky_fallbacks=fallback_count,
        maximum_jitter=used_jitter,
    )


def marginal_log_uniqueness(gram: torch.Tensor, jitter: float = 1.0e-4) -> torch.Tensor:
    """Log Schur-complement contribution of each expert to `gram + eps I`."""
    gram = gram.float()
    count = gram.shape[0]
    values: list[torch.Tensor] = []
    for expert in range(count):
        others = torch.tensor(
            [i for i in range(count) if i != expert], device=gram.device
        )
        cross = gram[expert, others]
        sub = gram[others][:, others]
        identity = torch.eye(count - 1, device=gram.device, dtype=gram.dtype)
        solution = torch.linalg.solve(sub + jitter * identity, gram[others, expert])
        schur = gram[expert, expert] + jitter - cross @ solution
        values.append(torch.log(schur.clamp_min(1.0e-12)))
    return torch.stack(values)
