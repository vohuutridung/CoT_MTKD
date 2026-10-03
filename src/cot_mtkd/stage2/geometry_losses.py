from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _validate_inputs(
    student_hidden: torch.Tensor,
    teacher_hidden_by_expert: list[torch.Tensor],
    head: torch.nn.Module,
    temperature: float,
    chunk_tokens: int,
) -> tuple[torch.device, torch.dtype]:
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("KD temperature must be finite and positive")
    if isinstance(chunk_tokens, bool) or not isinstance(chunk_tokens, int) or chunk_tokens <= 0:
        raise ValueError("chunk_tokens must be a positive integer")
    if student_hidden.ndim != 2 or student_hidden.shape[0] == 0:
        raise ValueError("KD requires nonempty student hidden states with shape [tokens, hidden]")
    if not student_hidden.is_floating_point() or not bool(torch.isfinite(student_hidden).all()):
        raise ValueError("Student hidden states must be finite floating-point values")
    if not teacher_hidden_by_expert:
        raise ValueError("KD requires at least one teacher")
    for hidden in teacher_hidden_by_expert:
        if hidden.shape != student_hidden.shape:
            raise ValueError("Teacher and student hidden states are misaligned")
        if not hidden.is_floating_point() or not bool(torch.isfinite(hidden).all()):
            raise ValueError("Teacher hidden states must be finite floating-point values")
    parameter = next(head.parameters(), None)
    if parameter is None:
        raise ValueError("The frozen output head must have parameters")
    return parameter.device, parameter.dtype


def _chunk_gradient(
    student_hidden: torch.Tensor,
    head: torch.nn.Module,
    target_log_probabilities: torch.Tensor,
    temperature: float,
    scale: float,
    head_device: torch.device,
    head_dtype: torch.dtype,
) -> tuple[torch.Tensor, float]:
    with torch.enable_grad():
        leaf = student_hidden.detach().to(device=head_device, dtype=head_dtype).requires_grad_(True)
        log_probabilities = F.log_softmax(head(leaf).float() / temperature, dim=-1)
        loss = F.kl_div(
            log_probabilities,
            target_log_probabilities.detach(),
            reduction="sum",
            log_target=True,
        ) * scale
        gradient = torch.autograd.grad(loss, leaf)[0]
    if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(gradient).all()):
        raise ValueError("Full-vocabulary KD produced a nonfinite loss or hidden gradient")
    return gradient.detach().to(student_hidden.device, dtype=student_hidden.dtype), float(loss.detach())


def teacher_kd_hidden_gradient(
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
    head: torch.nn.Module,
    temperature: float,
    chunk_tokens: int,
) -> tuple[torch.Tensor, float]:
    """Return d(T²/n sum KL(pm || pS))/d(hidden) and the same mean loss.

    Teacher states may remain on CPU. Only one token chunk's full-vocabulary
    logits is materialized at a time; all vocabulary probabilities are used.
    The returned cotangent is detached and already normalized by n.
    """
    head_device, head_dtype = _validate_inputs(
        student_hidden, [teacher_hidden], head, temperature, chunk_tokens
    )
    gradient = torch.zeros_like(student_hidden)
    loss_total = 0.0
    scale = temperature**2 / student_hidden.shape[0]
    for start in range(0, student_hidden.shape[0], chunk_tokens):
        end = min(student_hidden.shape[0], start + chunk_tokens)
        with torch.no_grad():
            teacher_logits = head(
                teacher_hidden[start:end].detach().to(device=head_device, dtype=head_dtype)
            ).float()
            target_log_probabilities = F.log_softmax(teacher_logits / temperature, dim=-1)
        del teacher_logits
        current_gradient, current_loss = _chunk_gradient(
            student_hidden[start:end], head, target_log_probabilities,
            temperature, scale, head_device, head_dtype,
        )
        del target_log_probabilities
        gradient[start:end] = current_gradient
        loss_total += current_loss
    return gradient.detach(), loss_total


def blended_kd_hidden_gradient(
    student_hidden: torch.Tensor,
    teacher_hidden_by_expert: list[torch.Tensor],
    head: torch.nn.Module,
    teacher_weights: torch.Tensor,
    consensus_weight: float,
    temperature: float,
    chunk_tokens: int,
) -> tuple[torch.Tensor, float]:
    """Full-vocabulary coverage/consensus KD with a detached mixed target.

    Coverage uses the weighted arithmetic mixture. Consensus is softmax of
    weighted teacher logits at the common temperature, equivalently the
    normalized geometric mixture. Mixtures are formed in log space without
    probability floors or vocabulary truncation.
    """
    head_device, head_dtype = _validate_inputs(
        student_hidden, teacher_hidden_by_expert, head, temperature, chunk_tokens
    )
    if not math.isfinite(consensus_weight) or not 0 <= consensus_weight <= 1:
        raise ValueError("consensus_weight must be finite and between zero and one")
    weights = teacher_weights.detach().to(device="cpu", dtype=torch.float64)
    if weights.ndim != 1 or weights.numel() != len(teacher_hidden_by_expert):
        raise ValueError("Teacher weights and expert hidden states are misaligned")
    if not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()):
        raise ValueError("Teacher weights must be finite and nonnegative")
    if not math.isclose(float(weights.sum()), 1.0, rel_tol=1.0e-6, abs_tol=1.0e-7):
        raise ValueError("Positive-utility teacher weights must sum to one")
    selected = [(index, value) for index, value in enumerate(weights.tolist()) if value > 0]
    gradient = torch.zeros_like(student_hidden)
    loss_total = 0.0
    scale = temperature**2 / student_hidden.shape[0]
    for start in range(0, student_hidden.shape[0], chunk_tokens):
        end = min(student_hidden.shape[0], start + chunk_tokens)
        with torch.no_grad():
            coverage_log_probabilities = None
            consensus_logits = None
            for expert_index, weight in selected:
                logits = head(
                    teacher_hidden_by_expert[expert_index][start:end].detach().to(
                        device=head_device, dtype=head_dtype
                    )
                ).float() / temperature
                if consensus_weight < 1:
                    weighted_log_probabilities = F.log_softmax(logits, dim=-1) + math.log(weight)
                    coverage_log_probabilities = (
                        weighted_log_probabilities
                        if coverage_log_probabilities is None
                        else torch.logaddexp(coverage_log_probabilities, weighted_log_probabilities)
                    )
                    del weighted_log_probabilities
                if consensus_weight > 0:
                    weighted_logits = logits * weight
                    consensus_logits = (
                        weighted_logits
                        if consensus_logits is None
                        else consensus_logits + weighted_logits
                    )
                    del weighted_logits
                del logits
            if consensus_weight == 0:
                target_log_probabilities = coverage_log_probabilities
            elif consensus_weight == 1:
                target_log_probabilities = F.log_softmax(consensus_logits, dim=-1)
            else:
                target_log_probabilities = torch.logaddexp(
                    F.log_softmax(consensus_logits, dim=-1) + math.log(consensus_weight),
                    coverage_log_probabilities + math.log1p(-consensus_weight),
                )
            del coverage_log_probabilities, consensus_logits
        current_gradient, current_loss = _chunk_gradient(
            student_hidden[start:end], head, target_log_probabilities,
            temperature, scale, head_device, head_dtype,
        )
        del target_log_probabilities
        gradient[start:end] = current_gradient
        loss_total += current_loss
    return gradient.detach(), loss_total
