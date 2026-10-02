from __future__ import annotations

import torch


def council_kneedle_candidates(
    council_probabilities: torch.Tensor,
    targets: torch.Tensor,
    epsilon: float = 1.0e-12,
    k_max: int | None = None,
) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    """Kneedle on the mean expert distribution at each token.

    The rank axis is ``x_j = j / N'``, where ``N'`` is the number of kept
    ranks (the vocabulary, or ``dpp_topk_cap`` when it is smaller). A flat
    row (``p_max == p_min``) selects every kept rank. The ground-truth token
    is removed only after the elbow is chosen. Returned tail mass is
    ``1 - sum_{v in V_k} p(v)`` and the outside flag is whether ``y*`` is
    missing from that same ``V_k``, both measured before the removal.
    """
    if council_probabilities.ndim != 2 or targets.shape != council_probabilities.shape[:1]:
        raise ValueError("Council probabilities and targets have incompatible shapes")
    vocab_size = council_probabilities.shape[-1]
    if k_max is None or int(k_max) >= vocab_size:
        probabilities, ids = torch.sort(
            council_probabilities.float(), dim=-1, descending=True
        )
        kept = vocab_size
        minimum = probabilities[:, -1:]
    else:
        kept = max(1, int(k_max))
        probabilities, ids = torch.topk(
            council_probabilities.float(), k=kept, dim=-1, largest=True, sorted=True
        )
        minimum = probabilities[:, -1:]
    ranks = torch.arange(1, kept + 1, device=probabilities.device, dtype=torch.float32)
    x = ranks / float(kept)
    maximum = probabilities[:, :1]
    flat = maximum.squeeze(-1) == minimum.squeeze(-1)
    y = (probabilities - minimum) / (maximum - minimum).clamp_min(epsilon)
    elbow = ((1.0 - x) - y).argmax(dim=-1) + 1
    elbow = torch.where(flat, torch.full_like(elbow, kept), elbow)
    in_support = torch.arange(kept, device=probabilities.device).unsqueeze(0) < elbow.unsqueeze(1)
    mass = (probabilities * in_support.to(probabilities.dtype)).sum(dim=-1)
    tail_mass = (1.0 - mass).detach().to("cpu")
    target_ids = targets.to(device=ids.device)
    ystar_outside = ~(ids.eq(target_ids.unsqueeze(1)) & in_support).any(dim=-1)
    ystar_outside = ystar_outside.detach().to("cpu")
    candidates = []
    for row, k in enumerate(elbow.tolist()):
        top_ids = ids[row, :k]
        candidates.append(top_ids[top_ids.ne(targets[row])].detach().to("cpu", torch.long))
    return candidates, elbow.detach().to("cpu", torch.long), tail_mass, ystar_outside


def dpp_support_rates(
    tail_mass_sum: float, ystar_outside_count: float, token_count: float
) -> tuple[float, float]:
    """Mean tail mass and ``y*`` miss rate over DPP tokens.

    Callers sum the per-token values across ranks first, then divide here.
    """
    denominator = max(float(token_count), 1.0)
    return float(tail_mass_sum) / denominator, float(ystar_outside_count) / denominator


def pad_candidate_support(
    candidates: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad per-token candidate sets for chunked logits and VJPs."""
    width = max((ids.numel() for ids in candidates), default=0)
    if width == 0:
        return (
            torch.empty((len(candidates), 0), dtype=torch.long),
            torch.empty((len(candidates), 0), dtype=torch.bool),
        )
    support = torch.zeros((len(candidates), width), dtype=torch.long)
    mask = torch.zeros((len(candidates), width), dtype=torch.bool)
    for row, ids in enumerate(candidates):
        count = ids.numel()
        if count:
            support[row, :count] = ids
            mask[row, :count] = True
    return support, mask


def capped_k_from_probe(
    sorted_probe_logits: torch.Tensor,
    non_target_min: torch.Tensor,
    non_target_max: torch.Tensor,
    vocab_size: int,
    min_k: int = 5,
    max_k: int = 128,
    epsilon: float = 1.0e-12,
) -> torch.Tensor:
    """Capped/probed Kneedle on descending non-target logits.

    `sorted_probe_logits` has shape `[tokens, probe_k]`. Global non-target
    min/max are used even though only the leading probe ranks are sorted.
    """
    if sorted_probe_logits.ndim != 2:
        raise ValueError("Probe logits must have shape [tokens, probe_k]")
    ranks = torch.arange(
        1,
        sorted_probe_logits.shape[1] + 1,
        device=sorted_probe_logits.device,
        dtype=torch.float32,
    )
    x = ranks / float(max(vocab_size - 1, 1))
    denominator = (non_target_max - non_target_min).unsqueeze(-1).clamp_min(epsilon)
    y = (
        sorted_probe_logits.float() - non_target_min.unsqueeze(-1).float()
    ) / denominator
    distance = (1.0 - x.unsqueeze(0)) - y
    elbow = distance.argmax(dim=-1) + 1
    return elbow.clamp(min=min_k, max=max_k).to(torch.int64)


def build_union_support(
    top_ids: torch.Tensor, selected_k: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the detached common union support for every reasoning token.

    Args:
        top_ids: int tensor `[experts, tokens, probe_k]` on CPU or GPU.
        selected_k: int tensor `[experts, tokens]`.
    Returns:
        padded support ids `[tokens, max_union]` and bool validity mask.
    """
    if top_ids.ndim != 3 or selected_k.shape != top_ids.shape[:2]:
        raise ValueError("Incompatible top-id and K tensors")
    ids_cpu = top_ids.detach().to("cpu", dtype=torch.int64)
    k_cpu = selected_k.detach().to("cpu", dtype=torch.int64)
    per_token: list[torch.Tensor] = []
    maximum = 0
    for token in range(ids_cpu.shape[1]):
        pieces = [
            ids_cpu[expert, token, : int(k_cpu[expert, token])]
            for expert in range(ids_cpu.shape[0])
        ]
        union = torch.unique(torch.cat(pieces), sorted=True)
        per_token.append(union)
        maximum = max(maximum, union.numel())
    support = torch.zeros((ids_cpu.shape[1], maximum), dtype=torch.int64)
    mask = torch.zeros((ids_cpu.shape[1], maximum), dtype=torch.bool)
    for token, union in enumerate(per_token):
        support[token, : union.numel()] = union
        mask[token, : union.numel()] = True
        if union.numel() < maximum:
            support[token, union.numel() :] = union[0]
    return support, mask
