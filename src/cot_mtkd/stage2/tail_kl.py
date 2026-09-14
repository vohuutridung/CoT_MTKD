from __future__ import annotations

import torch
import torch.nn.functional as F


def sparse_topk_with_tail(
    probabilities: torch.Tensor, top_k: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if probabilities.ndim != 2:
        raise ValueError("Probabilities must have shape [tokens, vocabulary]")
    values, ids = torch.topk(
        probabilities.float(), k=min(top_k, probabilities.shape[-1]), dim=-1
    )
    tail = (1.0 - values.sum(dim=-1)).clamp_min(0.0)
    return ids.to(torch.int32), values, tail


def tail_bucket_kl(
    student_logits: torch.Tensor,
    support_ids: torch.Tensor,
    teacher_top_probabilities: torch.Tensor,
    teacher_tail_mass: torch.Tensor,
    temperature: float = 2.0,
    epsilon: float = 1.0e-8,
) -> torch.Tensor:
    """Per-token KL after mapping vocabulary to Top-K plus one tail category."""
    if student_logits.shape[0] != support_ids.shape[0]:
        raise ValueError("Student/cache token counts differ")
    log_probabilities = F.log_softmax(student_logits.float() / temperature, dim=-1)
    student_top_log = log_probabilities.gather(-1, support_ids.long())
    student_top = student_top_log.exp()
    student_tail = (1.0 - student_top.sum(dim=-1)).clamp_min(epsilon)
    teacher_categories = torch.cat(
        [teacher_top_probabilities.float(), teacher_tail_mass.float().unsqueeze(-1)],
        dim=-1,
    ).clamp_min(epsilon)
    teacher_categories = teacher_categories / teacher_categories.sum(
        dim=-1, keepdim=True
    )
    student_categories = torch.cat(
        [student_top, student_tail.unsqueeze(-1)], dim=-1
    ).clamp_min(epsilon)
    student_categories = student_categories / student_categories.sum(
        dim=-1, keepdim=True
    )
    kl = (
        teacher_categories * (teacher_categories.log() - student_categories.log())
    ).sum(dim=-1)
    return (float(temperature) ** 2) * kl.clamp_min(0.0)
