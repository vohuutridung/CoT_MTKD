from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .geometry_losses import _chunk_gradient, _validate_inputs


def _teacher_log_probabilities(values: torch.Tensor) -> torch.Tensor:
    if values.ndim != 3 or values.shape[0] < 2 or min(values.shape[1:]) == 0:
        raise ValueError("Teacher log probabilities must have shape [M >= 2, tokens, vocabulary]")
    if not values.is_floating_point() or not bool(torch.isfinite(values).all()):
        raise ValueError("Teacher log probabilities must be finite floating-point values")
    # Log-softmax of finite teacher logits has finite values even when exp(log p)
    # underflows. No probability floor is needed, including for geometric means.
    return values.detach().to(dtype=torch.float64)


def _expm1_minus_x(values: torch.Tensor) -> torch.Tensor:
    # The Taylor remainder is below double roundoff on this interval. This
    # evaluates exp(x) - 1 - x without losing its leading x^2/2 term.
    small = values.abs() < 1.0e-3
    series = values.square() * (
        0.5
        + values
        * (
            1.0 / 6.0
            + values
            * (
                1.0 / 24.0
                + values
                * (
                    1.0 / 120.0
                    + values * (1.0 / 720.0 + values * (1.0 / 5040.0 + values / 40320.0))
                )
            )
        )
    )
    return torch.where(small, series, torch.expm1(values) - values)


def _log_mean_exp_centered(centered: torch.Tensor) -> torch.Tensor:
    # For small arguments logsumexp(x) - log(M) suffers cancellation. Centered
    # expm1/log1p keeps the positive-rho correction even very close to rho=0.
    # For large arguments logsumexp avoids exp overflow. The branch changes
    # only the numerical evaluation, never the aggregation parameter.
    if bool((centered.abs() <= 0.5).all()):
        mean_expm1 = centered.mean(dim=0) + _expm1_minus_x(centered).mean(dim=0)
        return torch.log1p(mean_expm1)
    return torch.logsumexp(centered, dim=0) - math.log(centered.shape[0])


@torch.no_grad()
def normalized_js_disagreement(log_teacher_probabilities: torch.Tensor) -> torch.Tensor:
    """Return each token's uniform-teacher JSD / log(M), in float64.

    Input axes are [teachers, tokens, vocabulary], with finite log-softmax
    values. The nonnegative f-divergence form avoids subtracting almost equal
    entropies when teachers are close to agreement.
    """
    log_probabilities = _teacher_log_probabilities(log_teacher_probabilities)
    mean_log = log_probabilities.mean(dim=0)
    log_mixture = mean_log + _log_mean_exp_centered(log_probabilities - mean_log)
    delta = log_probabilities - log_mixture
    # KL(p || q) = sum q [exp(delta) * delta - expm1(delta)]. The linear
    # probability terms integrate to zero. In this application delta <= log M.
    small = delta.abs() < 1.0e-3
    series = delta.square() * (
        0.5
        + delta
        * (
            1.0 / 3.0
            + delta
            * (
                1.0 / 8.0
                + delta
                * (1.0 / 30.0 + delta * (1.0 / 144.0 + delta * (1.0 / 840.0 + delta / 5760.0)))
            )
        )
    )
    divergence_kernel = torch.where(small, series, delta.exp() * delta - torch.expm1(delta))
    disagreement = (log_mixture.exp().unsqueeze(0) * divergence_kernel).sum(dim=-1).mean(
        dim=0
    ) / math.log(log_probabilities.shape[0])
    if not bool(torch.isfinite(disagreement).all()):
        raise ValueError("Teacher Jensen-Shannon disagreement is nonfinite")
    # The exact quantity is in [0, 1]. Clamping only removes roundoff at these
    # mathematical bounds; positive values near zero remain positive.
    return disagreement.clamp(0.0, 1.0).detach()


@torch.no_grad()
def power_mean_log_target(log_teacher_probabilities: torch.Tensor, rho: float) -> torch.Tensor:
    """Return the normalized uniform-teacher power mean, in log space.

    rho=0 is normalized geometric consensus and rho=1 is arithmetic coverage.
    Every positive rho uses the positive-power formula, including near zero.
    Probability algebra uses float64 without floors or vocabulary truncation.
    """
    if not math.isfinite(rho) or not 0.0 <= rho <= 1.0:
        raise ValueError("The power-mean parameter rho must be finite and in [0, 1]")
    log_probabilities = _teacher_log_probabilities(log_teacher_probabilities)
    mean_log = log_probabilities.mean(dim=0)
    if rho == 0.0:
        unnormalized_log_target = mean_log
    else:
        centered = rho * (log_probabilities - mean_log)
        unnormalized_log_target = mean_log + _log_mean_exp_centered(centered) / rho
    target = F.log_softmax(unnormalized_log_target, dim=-1)
    if not bool(torch.isfinite(target).all()):
        raise ValueError("The teacher power-mean target is nonfinite")
    return target.detach()


@torch.no_grad()
def _head_log_probabilities(
    teacher_hidden_by_expert: list[torch.Tensor],
    head: torch.nn.Module,
    start: int,
    end: int,
    temperature: float,
    head_device: torch.device,
    head_dtype: torch.dtype,
) -> torch.Tensor:
    values = []
    for hidden in teacher_hidden_by_expert:
        logits = head(hidden[start:end].detach().to(device=head_device, dtype=head_dtype))
        values.append(F.log_softmax(logits.to(dtype=torch.float64) / temperature, dim=-1))
    return torch.stack(values, dim=0)


def adaptive_kd_hidden_gradient(
    student_hidden: torch.Tensor,
    teacher_hidden_by_expert: list[torch.Tensor],
    head: torch.nn.Module,
    temperature: float,
    chunk_tokens: int,
    *,
    teacher_probability_cache_bytes: int = 0,
) -> tuple[torch.Tensor, float, float]:
    """Return the step KD hidden cotangent, mean loss, and disagreement ds.

    Implements T^2/n sum_t KL(sg(q_t(rho)) || pS_t), with one rho=ds for
    this entire reasoning step. Cache the step's teacher log probabilities when
    they fit the byte budget, avoiding a second teacher-head sweep. Larger steps
    use two sweeps with bounded chunk storage. Both paths retain full vocabulary
    and the same step rho. No teacher graph is constructed; the cotangent is
    detached and already normalized by n.
    """
    head_device, head_dtype = _validate_inputs(
        student_hidden, teacher_hidden_by_expert, head, temperature, chunk_tokens
    )
    if len(teacher_hidden_by_expert) < 2:
        raise ValueError("Disagreement-adaptive output-space KD requires at least two teachers")
    if teacher_probability_cache_bytes < 0:
        raise ValueError("Teacher probability cache byte budget must be nonnegative")
    token_count = student_hidden.shape[0]
    disagreement_total = torch.zeros((), device=head_device, dtype=torch.float64)
    cached_probabilities: list[torch.Tensor | None] = []
    cache_step = False
    for start in range(0, token_count, chunk_tokens):
        end = min(token_count, start + chunk_tokens)
        log_probabilities = _head_log_probabilities(
            teacher_hidden_by_expert, head, start, end, temperature, head_device, head_dtype
        )
        if start == 0:
            required_bytes = (
                len(teacher_hidden_by_expert)
                * token_count
                * log_probabilities.shape[-1]
                * log_probabilities.element_size()
            )
            cache_step = required_bytes <= teacher_probability_cache_bytes
        disagreement_total += normalized_js_disagreement(log_probabilities).sum()
        if cache_step:
            cached_probabilities.append(log_probabilities)
        del log_probabilities
    disagreement = min(1.0, max(0.0, float(disagreement_total) / token_count))

    gradient = torch.zeros_like(student_hidden)
    loss_total = 0.0
    scale = temperature**2 / token_count
    for index, start in enumerate(range(0, token_count, chunk_tokens)):
        end = min(token_count, start + chunk_tokens)
        if cache_step:
            log_probabilities = cached_probabilities[index]
            assert log_probabilities is not None
        else:
            log_probabilities = _head_log_probabilities(
                teacher_hidden_by_expert, head, start, end, temperature, head_device, head_dtype
            )
        target = power_mean_log_target(log_probabilities, disagreement)
        if cache_step:
            # Release each cached chunk once its target has been constructed.
            cached_probabilities[index] = None
        del log_probabilities
        current_gradient, current_loss = _chunk_gradient(
            student_hidden[start:end],
            head,
            target,
            temperature,
            scale,
            head_device,
            head_dtype,
        )
        gradient[start:end] = current_gradient
        loss_total += current_loss
        del target
    return gradient.detach(), loss_total, disagreement
