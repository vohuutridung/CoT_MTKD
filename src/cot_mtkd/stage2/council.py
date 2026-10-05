"""Council Top-k tools and Phase-2 token objectives.

Implements the shared Council Top-k tool (Kneedle on the consensus distribution
``p_bar``), the council signals used by Phase 2 (restricted distributions on
``V_k``, Jensen-Shannon disagreement, per-token variance mask and ``V_k`` mass),
the dynamic/effective temperatures and the student objectives: SFT, the
token-wise masked self-tempered KL on ``V_k`` and the binary mass loss.

Every council quantity is computed under ``no_grad``; only the student terms
carry gradients, and only through the student logits.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

TEMPERATURE_SCHEDULES = ("linear", "sigmoid", "step")
NEGATIVE_INFINITY = -torch.inf


# --------------------------------------------------------------------------
# Section 3: Council Top-k
# --------------------------------------------------------------------------
@torch.no_grad()
def kneedle_support(
    mean_probabilities: torch.Tensor, k_max: int | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select ``V_k`` per token with Kneedle on the sorted consensus ``p_bar``.

    ``mean_probabilities`` has shape ``[tokens, vocabulary]``. Sorting is
    descending; with ``x_j = j / N'`` and ``y_j`` the min-max normalized
    probability, the knee is ``argmax_j ((1 - x_j) - y_j)`` (first maximum).
    ``N' = N`` by default; ``k_max`` restricts the window to the top ``K_max``
    entries with ``p_min = p_bar_(K_max)``. A flat window (``p_max == p_min``)
    keeps ``k = N'``. Returns padded ids ``[tokens, width]`` (descending
    ``p_bar``), a validity mask and the per-token ``k``; nothing is
    differentiable.
    """
    if mean_probabilities.ndim != 2:
        raise ValueError("Consensus probabilities must have shape [tokens, vocabulary]")
    tokens, vocabulary = mean_probabilities.shape
    if vocabulary < 1:
        raise ValueError("Consensus probabilities need a nonempty vocabulary")
    window = vocabulary if k_max is None else min(int(k_max), vocabulary)
    if window < 1:
        raise ValueError("k_max must be a positive integer")
    device = mean_probabilities.device
    if tokens == 0:
        return (
            torch.empty((0, 0), dtype=torch.long, device=device),
            torch.empty((0, 0), dtype=torch.bool, device=device),
            torch.empty((0,), dtype=torch.long, device=device),
        )
    values = mean_probabilities.detach().float()
    if window == vocabulary:
        sorted_p, order = torch.sort(values, dim=-1, descending=True, stable=True)
    else:
        sorted_p, order = torch.topk(values, window, dim=-1, largest=True, sorted=True)
    p_max, p_min = sorted_p[:, :1], sorted_p[:, -1:]
    span = p_max - p_min
    ranks = torch.arange(1, window + 1, device=device, dtype=sorted_p.dtype)
    x = ranks / float(window)
    y = (sorted_p - p_min) / torch.where(span > 0, span, torch.ones_like(span))
    distance = (1.0 - x).unsqueeze(0) - y
    k = distance.argmax(dim=-1) + 1
    k = torch.where(span.squeeze(-1) > 0, k, torch.full_like(k, window))
    width = int(k.max().item())
    ids = order[:, :width].to(torch.long)
    mask = torch.arange(width, device=device)[None, :] < k[:, None]
    return ids, mask, k


def restricted_log_softmax(
    logits: torch.Tensor, support_ids: torch.Tensor, support_mask: torch.Tensor
) -> torch.Tensor:
    """``log pi_S``: full-vocabulary softmax restricted and renormalized on ``S``.

    Equals the softmax of the logits cut to ``S`` (the partition function
    cancels). ``logits`` is ``[..., tokens, vocabulary]``; ``support_ids`` and
    ``support_mask`` are ``[tokens, width]``. Padding receives ``-inf``.
    """
    if support_ids.shape != support_mask.shape or support_ids.ndim != 2:
        raise ValueError("Support ids and mask must share a [tokens, width] shape")
    if logits.shape[-2] != support_ids.shape[0]:
        raise ValueError("Logits and support disagree on the number of tokens")
    if not bool(support_mask.any(dim=-1).all()):
        raise ValueError("Every token needs at least one support category")
    index = support_ids.to(torch.long).expand(*logits.shape[:-2], *support_ids.shape)
    selected = logits.gather(-1, index).masked_fill(~support_mask, NEGATIVE_INFINITY)
    return F.log_softmax(selected, dim=-1).masked_fill(~support_mask, NEGATIVE_INFINITY)


@dataclass(frozen=True)
class CouncilSignals:
    """Per-token disagreement and per-category variance mask on ``V_k``."""

    js: torch.Tensor  # [tokens] nats in [0, ln M]
    variance: torch.Tensor  # [tokens, width] sigma^2(v), zero on padding
    variance_mask: torch.Tensor  # [tokens, width] M(v) in [0, 1], zero on padding


@torch.no_grad()
def council_signals(
    expert_log_pi: torch.Tensor, support_mask: torch.Tensor, mask_epsilon: float = 1.0e-12
) -> CouncilSignals:
    """JSD and variance mask from ``M >= 2`` restricted expert distributions.

    ``expert_log_pi`` is ``[experts, tokens, width]`` (``-inf`` on padding).
    ``JS = H(pi_bar) - mean_m H(pi^(m))`` and ``sigma^2(v) = mean_m (pi^(m)(v)
    - pi_bar(v))^2``, ``M(v) = sigma^2(v) / (max_u sigma^2(u) + eps)``. All
    algebra runs in float64.
    """
    if expert_log_pi.ndim != 3 or expert_log_pi.shape[0] < 2:
        raise ValueError("Council signals need [experts >= 2, tokens, width] log-probabilities")
    if not math.isfinite(mask_epsilon) or mask_epsilon <= 0:
        raise ValueError("mask_epsilon must be finite and positive")
    experts = expert_log_pi.shape[0]
    valid = support_mask.unsqueeze(0)
    pi = expert_log_pi.double().exp().masked_fill(~valid, 0.0)
    pi_bar = pi.mean(dim=0)
    entropy_bar = -torch.special.xlogy(pi_bar, pi_bar).sum(dim=-1)
    entropy_experts = -torch.special.xlogy(pi, pi).sum(dim=-1).mean(dim=0)
    js = (entropy_bar - entropy_experts).clamp(0.0, math.log(experts))
    variance = (pi - pi_bar.unsqueeze(0)).square().mean(dim=0).masked_fill(~support_mask, 0.0)
    scale = variance.max(dim=-1, keepdim=True).values + mask_epsilon
    variance_mask = (variance / scale).clamp(0.0, 1.0).masked_fill(~support_mask, 0.0)
    if not bool(torch.isfinite(js).all()) or not bool(torch.isfinite(variance_mask).all()):
        raise FloatingPointError("Council disagreement produced nonfinite values")
    return CouncilSignals(js, variance, variance_mask)


@torch.no_grad()
def support_mass(
    mean_probabilities: torch.Tensor, support_ids: torch.Tensor, support_mask: torch.Tensor
) -> torch.Tensor:
    """``q = p_bar(V_k)`` from the full-vocabulary consensus at temperature 1."""
    gathered = mean_probabilities.double().gather(-1, support_ids.to(torch.long))
    return gathered.masked_fill(~support_mask, 0.0).sum(dim=-1).clamp(0.0, 1.0)


# --------------------------------------------------------------------------
# Section 5.3: dynamic temperature
# --------------------------------------------------------------------------
def _linear_js_max(council: dict[str, Any], expert_count: int, js_p95: float | None) -> float:
    """``null`` keeps ``ln M``; ``p95`` is the cached reasoning-token JS percentile."""
    js_max = council.get("js_max")
    if js_max is None:
        return math.log(expert_count)
    if js_max == "p95":
        if js_p95 is None or not math.isfinite(float(js_p95)) or float(js_p95) <= 0.0:
            raise ValueError("js_max is p95 but the council JS 95th percentile is missing or nonpositive")
        return float(js_p95)
    return float(js_max)


def validate_temperature_config(council: dict[str, Any]) -> None:
    tau_min, tau_max = float(council["tau_min"]), float(council["tau_max"])
    if not (math.isfinite(tau_min) and math.isfinite(tau_max) and 0.0 < tau_min < 1.0 < tau_max):
        raise ValueError("Dynamic temperature requires 0 < tau_min < 1 < tau_max")
    schedule = council.get("temperature_schedule", "linear")
    if schedule not in TEMPERATURE_SCHEDULES:
        raise ValueError(f"temperature_schedule must be one of {TEMPERATURE_SCHEDULES}")
    js_max = council.get("js_max")
    if isinstance(js_max, str):
        if js_max != "p95":
            raise ValueError("js_max must be null (ln M), 'p95', or a positive number")
    elif js_max is not None and (isinstance(js_max, bool) or not math.isfinite(float(js_max)) or float(js_max) <= 0.0):
        raise ValueError("js_max must be null (ln M), 'p95', or a positive number")
    if schedule == "sigmoid":
        gamma = float(council.get("sigmoid_gamma", 1.0))
        if not math.isfinite(gamma) or gamma <= 0.0:
            raise ValueError("sigmoid_gamma must be finite and positive")
        center = council.get("sigmoid_center")
        if center is not None and (not math.isfinite(float(center)) or float(center) < 0.0):
            raise ValueError("sigmoid_center must be null (median JS) or nonnegative")
    if schedule == "step":
        low, high = float(council["step_theta_1"]), float(council["step_theta_2"])
        if not (math.isfinite(low) and math.isfinite(high) and 0.0 <= low < high):
            raise ValueError("Step temperature requires 0 <= step_theta_1 < step_theta_2")


@torch.no_grad()
def dynamic_temperature(
    js: torch.Tensor,
    council: dict[str, Any],
    expert_count: int,
    js_median: float | None = None,
    js_p95: float | None = None,
) -> torch.Tensor:
    """Map per-token JSD (nats) to ``tau_{i,t}``; linear is the default."""
    validate_temperature_config(council)
    if expert_count < 2:
        raise ValueError("Dynamic temperature needs at least two experts")
    js = js.detach().double()
    tau_min, tau_max = float(council["tau_min"]), float(council["tau_max"])
    schedule = council.get("temperature_schedule", "linear")
    if schedule == "linear":
        js_max = _linear_js_max(council, expert_count, js_p95)
        return tau_min + (tau_max - tau_min) * (js / js_max).clamp(max=1.0)
    if schedule == "sigmoid":
        center = council.get("sigmoid_center")
        if center is None:
            if js_median is None:
                raise ValueError("sigmoid_center is null and no JS median is available")
            center = float(js_median)
        gamma = float(council.get("sigmoid_gamma", 1.0))
        return tau_min + (tau_max - tau_min) * torch.sigmoid(gamma * (js - float(center)))
    low, high = float(council["step_theta_1"]), float(council["step_theta_2"])
    tau = torch.ones_like(js)
    tau = torch.where(js <= low, torch.full_like(js, tau_min), tau)
    return torch.where(js >= high, torch.full_like(js, tau_max), tau)


@torch.no_grad()
def effective_temperature(variance_mask: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    """``tau_eff(v)``: mask modulates flattening, inverted mask modulates sharpening."""
    if variance_mask.ndim != 2 or tau.shape != variance_mask.shape[:1]:
        raise ValueError("Expected variance_mask [tokens, width] and tau [tokens]")
    mask = variance_mask.to(torch.float64)
    delta = (tau.double() - 1.0).unsqueeze(-1)
    flatten = 1.0 + mask * delta
    sharpen = 1.0 + (1.0 - mask) * delta
    value = torch.where((tau >= 1.0).unsqueeze(-1), flatten, sharpen)
    if not bool((value > 0).all()):
        raise FloatingPointError("Effective temperature must be positive")
    return value


# --------------------------------------------------------------------------
# Section 5.4: student objectives
# --------------------------------------------------------------------------
def reasoning_token_terms(
    logits: torch.Tensor,
    targets: torch.Tensor,
    support_ids: torch.Tensor,
    support_mask: torch.Tensor,
    tau_eff: torch.Tensor,
    q: torch.Tensor,
    epsilon_m: float,
) -> dict[str, torch.Tensor]:
    """Unreduced per-token SFT, masked self-tempered KL, mass loss and diagnostics.

    ``logits`` are full-vocabulary student logits ``[tokens, vocabulary]``.
    ``P = Softmax(z|V_k)``, ``Q = sg[Softmax(z|V_k / tau_eff)]``,
    ``KL = sum_v Q log(Q / P)``; ``m = exp(lse(z|V_k) - lse(z))`` clipped to
    ``[eps_m, 1 - eps_m]`` and compared with ``q`` by a binary KL.
    """
    if not (0.0 < epsilon_m < 0.5):
        raise ValueError("epsilon_m must lie in (0, 0.5)")
    if logits.ndim != 2 or support_ids.shape != support_mask.shape:
        raise ValueError("Expected [tokens, vocabulary] logits and matching support tensors")
    if support_ids.shape[0] != logits.shape[0] or tau_eff.shape != support_ids.shape:
        raise ValueError("Support, temperature and logits disagree on token count")
    if not bool(torch.isfinite(logits).all()):
        raise FloatingPointError("Nonfinite student logits")
    logits = logits.float()
    sft = F.cross_entropy(logits, targets.to(torch.long), reduction="none")
    selected = logits.gather(-1, support_ids.to(torch.long))
    selected = selected.masked_fill(~support_mask, NEGATIVE_INFINITY)
    log_p = F.log_softmax(selected, dim=-1)
    tempered = selected.detach() / tau_eff.to(selected.dtype)
    log_q = F.log_softmax(tempered.masked_fill(~support_mask, NEGATIVE_INFINITY), dim=-1)
    q_dist = log_q.exp().masked_fill(~support_mask, 0.0)
    kl = torch.where(support_mask, q_dist * (log_q - log_p), torch.zeros_like(log_p)).sum(-1)
    p_dist = log_p.exp()
    entropy = -torch.where(support_mask, p_dist * log_p, torch.zeros_like(log_p)).sum(-1)
    log_mass = torch.logsumexp(selected, dim=-1) - torch.logsumexp(logits, dim=-1)
    log_mass = log_mass.clamp(min=math.log(epsilon_m), max=math.log1p(-epsilon_m))
    log_tail = torch.log(-torch.expm1(log_mass))
    q = q.to(log_mass.dtype)
    mass = (
        torch.special.xlogy(q, q)
        - q * log_mass
        + torch.special.xlogy(1.0 - q, 1.0 - q)
        - (1.0 - q) * log_tail
    )
    return {
        "sft": sft,
        "kl": kl,
        "mass": mass,
        "entropy": entropy,
        "student_mass": log_mass.detach().exp(),
    }


@dataclass
class ReasoningSupervision:
    """Padded per-token council signals for every retained reasoning token."""

    targets: torch.Tensor  # [tokens] long
    support_ids: torch.Tensor  # [tokens, width] long
    support_mask: torch.Tensor  # [tokens, width] bool
    tau_eff: torch.Tensor  # [tokens, width] float64
    q: torch.Tensor  # [tokens] float64
    sft_weight: torch.Tensor  # [tokens] float32
    kl_weight: torch.Tensor  # [tokens] float32 (alpha / |T|, zero outside T)
    mass_weight: torch.Tensor  # [tokens] float32 (beta / |T'|, zero outside T')


@dataclass
class PlainSupervision:
    """Always-on blocks: SFT-only tokens (answer block and format block)."""

    targets: torch.Tensor  # [tokens] long
    sft_weight: torch.Tensor  # [tokens] float32


def student_hidden_gradient(
    hidden: torch.Tensor,
    head: torch.nn.Module,
    reasoning: ReasoningSupervision | None,
    plain: PlainSupervision | None,
    chunk_tokens: int,
    epsilon_m: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Bounded head chunks; one combined hidden cotangent for the whole sample.

    Rows of ``hidden`` are the reasoning tokens first, then the plain tokens.
    Weights already encode the per-sample normalizations and the loss
    coefficients. Returned diagnostics are per reasoning token (CPU).
    """
    if chunk_tokens < 1:
        raise ValueError("chunk_tokens must be positive")
    parameter = next(head.parameters())
    gradient = torch.zeros_like(hidden)
    reasoning_count = 0 if reasoning is None else int(reasoning.targets.shape[0])
    plain_count = 0 if plain is None else int(plain.targets.shape[0])
    if hidden.shape[0] != reasoning_count + plain_count:
        raise ValueError("Hidden rows must match the supervised reasoning plus plain tokens")
    diagnostics: dict[str, list[torch.Tensor]] = {
        key: [] for key in ("sft", "kl", "mass", "entropy", "student_mass")
    }
    plain_sft: list[torch.Tensor] = []

    def leaf(start: int, end: int) -> torch.Tensor:
        return hidden[start:end].detach().to(parameter.device, parameter.dtype).requires_grad_(True)

    for start in range(0, reasoning_count, chunk_tokens):
        end = min(reasoning_count, start + chunk_tokens)
        current = leaf(start, end)
        logits = head(current).float()
        device = logits.device
        terms = reasoning_token_terms(
            logits,
            reasoning.targets[start:end].to(device),
            reasoning.support_ids[start:end].to(device),
            reasoning.support_mask[start:end].to(device),
            reasoning.tau_eff[start:end].to(device),
            reasoning.q[start:end].to(device),
            epsilon_m,
        )
        objective = (
            terms["sft"] * reasoning.sft_weight[start:end].to(device)
            + terms["kl"] * reasoning.kl_weight[start:end].to(device)
            + terms["mass"] * reasoning.mass_weight[start:end].to(device)
        ).sum()
        gradient[start:end] = torch.autograd.grad(objective, current)[0].to(gradient.dtype)
        for key, values in diagnostics.items():
            values.append(terms[key].detach().cpu())
        del logits, terms, objective, current
    for start in range(0, plain_count, chunk_tokens):
        end = min(plain_count, start + chunk_tokens)
        current = leaf(reasoning_count + start, reasoning_count + end)
        logits = head(current).float()
        sft = F.cross_entropy(logits, plain.targets[start:end].to(logits.device), reduction="none")
        objective = (sft * plain.sft_weight[start:end].to(logits.device)).sum()
        gradient[reasoning_count + start : reasoning_count + end] = torch.autograd.grad(
            objective, current
        )[0].to(gradient.dtype)
        plain_sft.append(sft.detach().cpu())
        del logits, sft, objective, current
    empty = torch.empty(0)
    summary = {key: torch.cat(values) if values else empty for key, values in diagnostics.items()}
    summary["plain_sft"] = torch.cat(plain_sft) if plain_sft else empty
    return gradient.detach(), summary
