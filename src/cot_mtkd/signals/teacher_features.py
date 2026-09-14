from __future__ import annotations

import torch


def zscore_across_experts(
    values: torch.Tensor, epsilon: float = 1.0e-6
) -> torch.Tensor:
    values = values.float()
    mean = values.mean(dim=-1, keepdim=True)
    std = values.std(dim=-1, unbiased=False, keepdim=True)
    return torch.where(
        std >= epsilon,
        (values - mean) / std.clamp_min(epsilon),
        torch.zeros_like(values),
    )


def reasoning_teacher_weights(
    competence: torch.Tensor,
    agreement: torch.Tensor,
    uniqueness: torch.Tensor,
    uniform_mass: float = 0.10,
    epsilon: float = 1.0e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not (competence.shape == agreement.shape == uniqueness.shape):
        raise ValueError("Teacher features must have matching [steps, experts] shapes")
    c = zscore_across_experts(competence, epsilon)
    a = zscore_across_experts(agreement, epsilon)
    u = zscore_across_experts(uniqueness, epsilon)
    score = c + a + torch.sigmoid(c) * u
    adaptive = torch.softmax(score, dim=-1)
    expert_count = score.shape[-1]
    weights = (1.0 - uniform_mass) * adaptive + uniform_mass / expert_count
    return weights, score


def answer_teacher_weights(
    competence: torch.Tensor,
    agreement: torch.Tensor,
    uniform_mass: float = 0.10,
    epsilon: float = 1.0e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    if competence.shape != agreement.shape:
        raise ValueError("Answer competence and agreement must have equal shapes")
    c = zscore_across_experts(competence.unsqueeze(0), epsilon).squeeze(0)
    a = zscore_across_experts(agreement.unsqueeze(0), epsilon).squeeze(0)
    score = c + a
    adaptive = torch.softmax(score, dim=-1)
    expert_count = score.shape[-1]
    weights = (1.0 - uniform_mass) * adaptive + uniform_mass / expert_count
    return weights, score
