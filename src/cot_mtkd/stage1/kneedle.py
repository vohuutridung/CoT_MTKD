from __future__ import annotations

import torch


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
        padded support ids `[tokens, max_union]` and bool validity mask on
        `top_ids.device`. Valid ids are ascending and unique. Padding repeats
        the first valid id, or zero for an empty support.
    """
    if top_ids.ndim != 3 or selected_k.shape != top_ids.shape[:2]:
        raise ValueError("Incompatible top-id and K tensors")
    ids = top_ids.detach().to(dtype=torch.int64)
    k = selected_k.detach().to(device=ids.device, dtype=torch.int64)
    experts, tokens, probe_k = ids.shape
    if not experts or not tokens or not probe_k:
        return (
            ids.new_empty((tokens, 0)),
            torch.empty((tokens, 0), device=ids.device, dtype=torch.bool),
        )
    # Match the prefix slicing used by the reference implementation, including
    # K values outside the probe width. This synchronizes only a scalar on CUDA.
    k = torch.where(k < 0, (k + probe_k).clamp_min(0), k.clamp_max(probe_k))
    candidate_k = int(k.max().item())
    if candidate_k == 0:
        return (
            ids.new_empty((tokens, 0)),
            torch.empty((tokens, 0), device=ids.device, dtype=torch.bool),
        )
    ranks = torch.arange(candidate_k, device=ids.device)
    valid = ranks[None, None, :] < k[:, :, None]
    # Vocabulary ids cannot equal this sentinel. Move padding to the end before
    # finding adjacent duplicates, then compact unique ids with a second sort.
    sentinel = torch.iinfo(torch.int64).max
    candidates = ids[:, :, :candidate_k].masked_fill(~valid, sentinel)
    ordered = candidates.permute(1, 0, 2).reshape(tokens, -1).sort(dim=1).values
    starts = ordered != sentinel
    starts[:, 1:] &= ordered[:, 1:] != ordered[:, :-1]
    counts = starts.sum(dim=1)
    maximum = int(counts.max().item())
    compact = ordered.masked_fill(~starts, sentinel).sort(dim=1).values[:, :maximum]
    mask = torch.arange(maximum, device=ids.device)[None, :] < counts[:, None]
    first = ordered[:, :1].masked_fill(counts[:, None] == 0, 0)
    support = torch.where(mask, compact, first)
    return support, mask
