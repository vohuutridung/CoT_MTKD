from __future__ import annotations

import torch


@torch.no_grad()
def local_k_from_probe(
    sorted_probe_logits: torch.Tensor,
    k_min: int = 8,
    epsilon: float = 1.0e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Local Kneedle over the top-512 non-target logits (or fewer candidates).

    Both axes are normalized inside the supplied descending search window:
    x[j] = j / (K - 1), u[j] = (z[j] - z[-1]) / (z[0] - z[-1] + epsilon),
    using zero-based j. Return raw knee ranks and final support sizes, with
    k = min(K, max(raw_k, k_min)). Ties choose the first maximum. For K=1,
    both sizes are 1; for K=0 they are 0. Selection is detached from autograd.
    """
    if sorted_probe_logits.ndim != 2:
        raise ValueError("Probe logits must have shape [tokens, K]")
    if k_min < 1 or epsilon <= 0:
        raise ValueError("k_min and epsilon must be positive")
    tokens, window = sorted_probe_logits.shape
    if window > 512:
        raise ValueError("Local Kneedle search window cannot exceed 512 candidates")
    if window <= 1:
        raw_k = torch.full(
            (tokens,), window, device=sorted_probe_logits.device, dtype=torch.int64
        )
        return raw_k, raw_k.clone()
    ranks = torch.arange(
        window,
        device=sorted_probe_logits.device,
        dtype=torch.float64 if sorted_probe_logits.dtype == torch.float64 else torch.float32,
    )
    x = ranks / float(window - 1)
    # Preserve FP64 inputs for reference checks; promote FP16/BF16 to FP32.
    logits = sorted_probe_logits.to(
        torch.float64 if sorted_probe_logits.dtype == torch.float64 else torch.float32
    )
    tail = logits[:, -1:]
    u = (logits - tail) / (logits[:, :1] - tail + epsilon)
    distance = (1.0 - x.unsqueeze(0)) - u
    raw_k = (distance.argmax(dim=-1) + 1).to(torch.int64)
    return raw_k, raw_k.clamp(min=k_min).clamp(max=window)


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
