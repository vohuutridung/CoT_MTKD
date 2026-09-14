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
