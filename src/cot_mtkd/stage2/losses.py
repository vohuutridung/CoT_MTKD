from __future__ import annotations

import torch
import torch.nn.functional as F

from .tail_kl import tail_bucket_kl


def dual_source_hidden_gradients(
    hidden: torch.Tensor,
    head: torch.nn.Module,
    targets: torch.Tensor,
    hard_weights: torch.Tensor,
    support_ids: torch.Tensor,
    teacher_top_probabilities: torch.Tensor,
    teacher_tail_mass: torch.Tensor,
    temperature: float,
    epsilon: float,
    chunk_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor, float, float, float, int]:
    """Loss sums and hidden cotangents without retaining full-vocabulary logits."""
    if not (
        hidden.shape[0]
        == targets.numel()
        == hard_weights.numel()
        == support_ids.shape[0]
        == teacher_top_probabilities.shape[0]
        == teacher_tail_mass.numel()
    ):
        raise ValueError("Dual-source token tensors are misaligned")
    hard_gradient = torch.zeros_like(hidden)
    kd_gradient = torch.zeros_like(hidden)
    hard_sum = kd_sum = 0.0
    hard_weight_sum = float(hard_weights.float().sum().item())
    head_dtype = next(head.parameters()).dtype
    for start in range(0, hidden.shape[0], chunk_tokens):
        end = min(hidden.shape[0], start + chunk_tokens)
        leaf = hidden[start:end].detach().to(dtype=head_dtype).requires_grad_(True)
        logits = head(leaf).float()
        ce = F.cross_entropy(
            logits, targets[start:end].to(logits.device), reduction="none"
        )
        current_hard = (ce * hard_weights[start:end].to(logits.device).float()).sum()
        current_kd = tail_bucket_kl(
            logits,
            support_ids[start:end].to(logits.device),
            teacher_top_probabilities[start:end].to(logits.device),
            teacher_tail_mass[start:end].to(logits.device),
            temperature,
            epsilon,
        ).sum()
        hard_grad = torch.autograd.grad(current_hard, leaf, retain_graph=True)[0]
        kd_grad = torch.autograd.grad(current_kd, leaf)[0]
        hard_gradient[start:end] = hard_grad.to(hard_gradient.dtype)
        kd_gradient[start:end] = kd_grad.to(kd_gradient.dtype)
        hard_sum += float(current_hard.detach().item())
        kd_sum += float(current_kd.detach().item())
    return (
        hard_gradient,
        kd_gradient,
        hard_sum,
        kd_sum,
        hard_weight_sum,
        int(hidden.shape[0]),
    )
