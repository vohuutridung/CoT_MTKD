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
    support_logits: torch.Tensor,
    support_mask: torch.Tensor,
    epsilon: float = 1.0e-12,
) -> torch.Tensor:
    """L2-normalized exponentiated-logit vectors on common support.

    `support_logits` is `[experts, tokens, support]`; `support_mask` is
    `[tokens, support]`. Padding coordinates are exactly zero.
    """
    mask = support_mask.unsqueeze(0).to(device=support_logits.device)
    masked = support_logits.float().masked_fill(~mask, -torch.inf)
    maximum = masked.max(dim=-1, keepdim=True).values
    values = torch.exp(masked - maximum).masked_fill(~mask, 0.0)
    norm = values.pow(2).sum(dim=-1, keepdim=True).sqrt().clamp_min(epsilon)
    return values / norm


def _cholesky_logdet(
    gram: torch.Tensor, initial_jitter: float, maximum_jitter: float
) -> tuple[torch.Tensor, float, int]:
    values, used_jitter, fallbacks = _batched_cholesky_logdet(
        gram.unsqueeze(0), initial_jitter, maximum_jitter
    )
    return values[0], used_jitter, fallbacks


def _batched_cholesky_logdet(
    grams: torch.Tensor, initial_jitter: float, maximum_jitter: float
) -> tuple[torch.Tensor, float, int]:
    """Factor all step matrices together; retry only matrices that failed.

    The common path needs one status check for the whole batch, rather than
    one device synchronization per reasoning step. Failed Cholesky factors
    never enter autograd: even a zero upstream gradient through a singular
    factor can produce non-finite gradients.
    """
    grams = grams.float()
    identity = torch.eye(grams.shape[-1], device=grams.device, dtype=grams.dtype)
    jitter = initial_jitter
    fallback_count = 0
    factor, info = torch.linalg.cholesky_ex(grams + jitter * identity)
    if not bool(info.ne(0).any()):
        return (
            2.0 * torch.log(torch.diagonal(factor, dim1=-2, dim2=-1)).sum(-1),
            jitter,
            fallback_count,
        )

    # Jitter retries are exceptional. Re-factor the successful subset so its
    # backward cannot touch the invalid factors produced above.
    values = grams.new_zeros(grams.shape[0])
    successful = torch.where(info.eq(0))[0]
    if successful.numel():
        factor = torch.linalg.cholesky(grams[successful] + jitter * identity)
        values = values.index_copy(
            0,
            successful,
            2.0 * torch.log(torch.diagonal(factor, dim1=-2, dim2=-1)).sum(-1),
        )
    pending = torch.where(info.ne(0))[0]
    while True:
        fallback_count += pending.numel()
        if jitter >= maximum_jitter:
            sign, logabsdet = torch.linalg.slogdet(
                grams[pending] + maximum_jitter * identity
            )
            if bool(sign.le(0).any()):
                raise FloatingPointError(
                    "DPP Gram matrix remains non-positive after maximum jitter"
                )
            return values.index_copy(0, pending, logabsdet), maximum_jitter, fallback_count
        jitter = min(maximum_jitter, jitter * 10.0)
        retried = grams[pending] + jitter * identity
        factor, info = torch.linalg.cholesky_ex(retried)
        if not bool(info.ne(0).any()):
            logdet = 2.0 * torch.log(torch.diagonal(factor, dim1=-2, dim2=-1)).sum(-1)
            return values.index_copy(0, pending, logdet), jitter, fallback_count
        successful = torch.where(info.eq(0))[0]
        if successful.numel():
            factor = torch.linalg.cholesky(retried[successful])
            logdet = 2.0 * torch.log(torch.diagonal(factor, dim1=-2, dim2=-1)).sum(-1)
            values = values.index_copy(0, pending[successful], logdet)
        pending = pending[info.ne(0)]


def step_dpp_loss(
    features: torch.Tensor,
    sample_ids: torch.Tensor,
    step_ids: torch.Tensor,
    jitter: float = 1.0e-4,
    maximum_jitter: float = 1.0e-2,
    reduction: str = "mean",
) -> tuple[torch.Tensor, DPPMetrics]:
    """Negative log-volume, mean over steps per sample then over samples."""
    if features.ndim != 3:
        raise ValueError("features must have shape [experts, tokens, dimensions]")
    if sample_ids.numel() != features.shape[1] or step_ids.numel() != features.shape[1]:
        raise ValueError("Token metadata does not match DPP features")
    if reduction not in {"mean", "sum", "none"}:
        raise ValueError(f"Unknown DPP reduction: {reduction}")
    if features.shape[1] == 0:
        zero = features.sum() * 0.0
        return zero, DPPMetrics(0, 0, 0, jitter)
    sample_ids = sample_ids.to(device=features.device, dtype=torch.long)
    step_ids = step_ids.to(device=features.device)
    if (step_ids < 0).any():
        raise ValueError("DPP received a non-reasoning token")
    expert_count = features.shape[0]
    pair_ids = torch.stack([sample_ids, step_ids.long()], dim=-1)
    unique_pairs, token_groups, token_counts = torch.unique(
        pair_ids, dim=0, sorted=True, return_inverse=True, return_counts=True
    )
    # Each token contributes an experts-by-experts matrix. Accumulating these
    # small matrices avoids a token mask and feature gather for every step.
    # Match GEMM's FP32 accumulation for reduced-precision inputs. Accumulating
    # BF16 per-token results in BF16 would introduce one rounding per token.
    accumulation_features = (
        features.float() if features.dtype in {torch.float16, torch.bfloat16} else features
    )
    token_grams = torch.einsum("mtk,ntk->tmn", accumulation_features, accumulation_features)
    # Long, nearly rank-one steps are sensitive to summation error at jitter
    # 1e-4. Accumulate only these tiny M-by-M contributions in FP64; the large
    # token/support features and their dot products remain in FP32.
    grams = torch.zeros(
        unique_pairs.shape[0], expert_count, expert_count,
        device=features.device, dtype=torch.float64,
    ).index_add(0, token_groups, token_grams.double())
    if features.dtype in {torch.float16, torch.bfloat16}:
        grams = grams.to(features.dtype) / token_counts[:, None, None]
    else:
        grams = (grams / token_counts[:, None, None]).to(features.dtype)
    logdets, used_jitter, fallback_count = _batched_cholesky_logdet(
        grams, jitter, maximum_jitter
    )
    step_losses = -logdets / expert_count
    unique_samples, sample_groups, step_counts = torch.unique(
        unique_pairs[:, 0], sorted=True, return_inverse=True, return_counts=True
    )
    stacked = step_losses.new_zeros(unique_samples.shape[0]).index_add(
        0, sample_groups, step_losses
    ) / step_counts
    if reduction == "mean":
        result = stacked.mean()
    elif reduction == "sum":
        result = stacked.sum()
    elif reduction == "none":
        result = stacked
    return result, DPPMetrics(
        groups=unique_pairs.shape[0],
        samples=unique_samples.shape[0],
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
