from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from ..stage1.kneedle import build_union_support, local_k_from_probe


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
    Probability algebra uses float64 on the reduced support plus tail categories.
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
def support_from_probes(
    top_values: torch.Tensor,
    top_ids: torch.Tensor,
    targets: torch.Tensor,
    k_min: int = 8,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Use Phase-1 local Kneedle and canonical union, then explicitly add gold.

    Probes MUST exclude gold (the shared full_vocab_probe does this). This
    function also handles duplicates defensively; all selection is detached.
    """
    choices = [local_k_from_probe(values, k_min=k_min) for values in top_values]
    raw_k = torch.stack([raw for raw, _ in choices])
    selected_k = torch.stack([k for _, k in choices])
    union, mask = build_union_support(top_ids, selected_k)
    gold_present = ((union == targets[:, None]) & mask).any(-1)
    # Reuse the Phase-1 canonicalizer for the augmented union too. Its validity
    # convention is prefix lengths; adding gold as another one-element set
    # ensures duplicates disappear and valid ids remain ascending.
    width = max(union.shape[-1], 1)
    candidates = top_ids.new_zeros((2, len(targets), width))
    candidates[0, :, : union.shape[-1]] = union
    candidates[1, :, 0] = targets
    support, valid = build_union_support(
        candidates, torch.stack([mask.sum(-1), torch.ones_like(targets)])
    )
    return (
        support,
        valid,
        {
            "raw_k": raw_k,
            "selected_k": selected_k,
            "union_sizes": mask.sum(-1),
            "support_sizes": valid.sum(-1),
            "gold_present_before_add": gold_present,
        },
    )


def reduced_log_distribution(
    logits: torch.Tensor,
    support_ids: torch.Tensor,
    support_mask: torch.Tensor,
    temperature: float,
    anomalies: dict[str, int] | None = None,
) -> torch.Tensor:
    """Full-softmax mass on support and ONE complement bucket, without full p.

    logZ and gathered support use FP32 (FP64 for reference inputs). The tail
    uses -expm1(log support mass). Near saturation, an outside logsumexp avoids
    cancellation and preserves tiny tails and their gradients. Empty complement
    gets a 1e-30 numerical floor. Every floor/roundoff correction is counted.
    Padding is log(0) represented by a finite sentinel, with exactly zero exp.
    """
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if logits.ndim != 2 or support_ids.shape != support_mask.shape:
        raise ValueError("Expected [tokens, vocabulary] logits and matching support tensors")
    if support_ids.shape[0] != logits.shape[0] or support_mask.dtype != torch.bool:
        raise ValueError("Invalid support mask")
    if not bool(torch.isfinite(logits).all()):
        raise FloatingPointError("Nonfinite logits in reduced distribution")
    scaled = (
        logits.to(torch.float64 if logits.dtype == torch.float64 else torch.float32) / temperature
    )
    log_z = torch.logsumexp(scaled, -1, keepdim=True)
    selected = scaled.gather(-1, support_ids.long()).masked_fill(~support_mask, -torch.inf)
    log_mass = torch.logsumexp(selected, -1, keepdim=True) - log_z
    remainder = -torch.expm1(log_mass.clamp_max(0))
    tail = remainder.clamp_min(1e-30).log()
    near = log_mass.squeeze(-1) > -1e-4
    empty_complement = torch.zeros_like(near)
    if bool(near.any()):
        outside = scaled[near].clone()
        chosen = support_ids[near].long()
        # Padding repeats a valid id, so it does not mask an extra category.
        outside.scatter_(1, chosen, -torch.inf)
        empty = support_mask[near].sum(-1) == logits.shape[-1]
        empty_complement[near] = empty
        # Avoid differentiating logsumexp of an all -inf row.
        outside = torch.where(empty[:, None], torch.zeros_like(outside), outside)
        outside_log = torch.logsumexp(outside, -1, keepdim=True) - log_z[near]
        tail = tail.clone()
        tail[near] = torch.where(
            empty[:, None], torch.full_like(outside_log, math.log(1e-30)), outside_log
        )
    if anomalies is not None:
        for key, value in {
            "tail_roundoff_corrections": int((log_mass > 0).sum()),
            "tail_probability_clamps": int(empty_complement.sum()),
            "tail_complement_fallbacks": int(near.sum()),
        }.items():
            anomalies[key] = anomalies.get(key, 0) + value
    reduced = torch.cat([selected - log_z, tail], -1)
    # Correct only summation roundoff (or the empty-complement floor), never
    # normalize the selected region before calculating its complement mass.
    reduced = reduced - torch.logsumexp(reduced, -1, keepdim=True)
    return reduced.masked_fill(
        ~torch.cat([support_mask, torch.ones_like(near[:, None])], -1), -1e30
    )


@torch.no_grad()
def local_js_log_distribution(
    logits: torch.Tensor,
    support_ids: torch.Tensor,
    mask: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Restriction+renormalization of full softmax equals support-only softmax.

    The full partition function cancels; JS has no tail bucket.
    """
    selected = logits.gather(-1, support_ids.long()).double() / temperature
    selected = selected.masked_fill(~mask, -torch.inf)
    return F.log_softmax(selected, -1).masked_fill(~mask, -1e30).detach()


def reduced_kd_sft_tokens(
    logits: torch.Tensor,
    targets: torch.Tensor,
    support_ids: torch.Tensor,
    mask: torch.Tensor,
    log_target: torch.Tensor,
    kd_temperature: float,
    anomalies: dict[str, int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Unreduced token losses, with sg(target), KL(q||rS), and ordinary CE."""
    student = reduced_log_distribution(logits, support_ids, mask, kd_temperature, anomalies)
    target = log_target.detach().to(student.dtype)
    target = target - torch.logsumexp(target, -1, keepdim=True)
    probabilities = target.exp()
    kd = kd_temperature**2 * (probabilities * (target - student)).sum(-1)
    sft = F.cross_entropy(logits.float(), targets.long(), reduction="none")
    entropy = -(probabilities * target).sum(-1)
    return kd, sft, student[:, -1].exp(), entropy


def cached_kd_sft_hidden_gradient(
    student_hidden: torch.Tensor,
    head: torch.nn.Module,
    targets: torch.Tensor,
    support_ids: torch.Tensor,
    mask: torch.Tensor,
    log_target: torch.Tensor,
    token_weights: torch.Tensor,
    kd_temperature: float,
    sft_weight: float,
    chunk_tokens: int,
    anomalies: dict[str, int] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Bounded head chunks, one combined KD+SFT hidden cotangent per sample.

    token_weights already encode token->step->sample means. Returned token
    diagnostics permit step logging with exactly the same supervision mask.
    No teacher module or teacher tensor is accepted by this training path.
    """
    if chunk_tokens < 1 or not math.isfinite(sft_weight) or sft_weight < 0:
        raise ValueError("Invalid chunk_tokens or sft_weight")
    parameter = next(head.parameters())
    gradient = torch.zeros_like(student_hidden)
    diagnostics: dict[str, list[torch.Tensor]] = {
        key: [] for key in ("kd", "sft", "student_tail", "entropy")
    }
    for start in range(0, len(student_hidden), chunk_tokens):
        end = min(len(student_hidden), start + chunk_tokens)
        hidden = (
            student_hidden[start:end]
            .detach()
            .to(parameter.device, parameter.dtype)
            .requires_grad_(True)
        )
        logits = head(hidden)
        kd, sft, tail, entropy = reduced_kd_sft_tokens(
            logits,
            targets[start:end].to(parameter.device),
            support_ids[start:end].to(parameter.device),
            mask[start:end].to(parameter.device),
            log_target[start:end].to(parameter.device),
            kd_temperature,
            anomalies,
        )
        objective = ((kd + sft_weight * sft) * token_weights[start:end].to(parameter.device)).sum()
        gradient[start:end] = torch.autograd.grad(objective, hidden)[0].to(gradient.dtype)
        for key, value in zip(diagnostics, (kd, sft, tail, entropy), strict=True):
            diagnostics[key].append(value.detach().cpu())
    return gradient.detach(), {key: torch.cat(values) for key, values in diagnostics.items()}
