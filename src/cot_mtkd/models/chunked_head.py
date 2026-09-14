from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F


def decoder_and_lm_head(
    model: torch.nn.Module,
) -> tuple[torch.nn.Module, torch.nn.Module]:
    causal_lm = model.get_base_model() if hasattr(model, "get_base_model") else model
    decoder = getattr(causal_lm, "model", None)
    if decoder is None:
        raise TypeError(
            f"Cannot locate decoder module on {causal_lm.__class__.__name__}"
        )
    head = causal_lm.get_output_embeddings()
    if head is None:
        raise TypeError("Causal LM has no output embedding / LM head")
    return decoder, head


def forward_hidden(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    use_cache: bool = False,
    past_key_values: Any | None = None,
):
    decoder, _ = decoder_and_lm_head(model)
    return decoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=use_cache,
        past_key_values=past_key_values,
        return_dict=True,
    )


def shifted_hidden(hidden: torch.Tensor) -> torch.Tensor:
    return hidden[:, :-1, :]


def gather_hidden_positions(
    hidden: torch.Tensor, batch_indices: torch.Tensor, hidden_indices: torch.Tensor
) -> torch.Tensor:
    return shifted_hidden(hidden)[batch_indices, hidden_indices]


def full_vocab_probe(
    hidden: torch.Tensor,
    head: torch.nn.Module,
    targets: torch.Tensor,
    probe_k: int,
    chunk_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return descending non-target top-k values/ids and full non-target min/max on CPU."""
    count = hidden.shape[0]
    all_values: list[torch.Tensor] = []
    all_ids: list[torch.Tensor] = []
    all_min: list[torch.Tensor] = []
    all_max: list[torch.Tensor] = []
    head_device = next(head.parameters()).device
    head_dtype = next(head.parameters()).dtype
    with torch.no_grad():
        for start in range(0, count, chunk_tokens):
            end = min(count, start + chunk_tokens)
            current = hidden[start:end].to(device=head_device, dtype=head_dtype)
            logits = head(current).float()
            target = targets[start:end].to(logits.device)
            row = torch.arange(end - start, device=logits.device)
            target_values = logits[row, target].clone()
            logits[row, target] = -torch.inf
            values, ids = torch.topk(
                logits, k=min(probe_k, logits.shape[-1] - 1), dim=-1
            )
            maximum = values[:, 0]
            logits[row, target] = torch.inf
            minimum = logits.min(dim=-1).values
            logits[row, target] = target_values
            all_values.append(values.cpu())
            all_ids.append(ids.to(torch.int32).cpu())
            all_min.append(minimum.cpu())
            all_max.append(maximum.cpu())
    return (
        torch.cat(all_values),
        torch.cat(all_ids),
        torch.cat(all_min),
        torch.cat(all_max),
    )


def gather_support_logits(
    hidden: torch.Tensor,
    head: torch.nn.Module,
    support_ids: torch.Tensor,
    chunk_tokens: int,
    output_device: torch.device | None = None,
) -> torch.Tensor:
    count = hidden.shape[0]
    values: list[torch.Tensor] = []
    head_device = next(head.parameters()).device
    head_dtype = next(head.parameters()).dtype
    with torch.no_grad():
        for start in range(0, count, chunk_tokens):
            end = min(count, start + chunk_tokens)
            current = hidden[start:end].to(device=head_device, dtype=head_dtype)
            logits = head(current)
            ids = support_ids[start:end].to(logits.device, dtype=torch.long)
            selected = logits.gather(-1, ids).float()
            values.append(selected.to(output_device or head_device))
    return torch.cat(values, dim=0)


def cross_entropy_hidden_gradient(
    hidden: torch.Tensor,
    head: torch.nn.Module,
    targets: torch.Tensor,
    chunk_tokens: int,
    token_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Compute CE sum and d(sum CE)/d(hidden) without retaining dense logits."""
    gradient = torch.zeros_like(hidden)
    loss_total = torch.zeros((), device=hidden.device, dtype=torch.float64)
    weight_total = 0.0
    head_dtype = next(head.parameters()).dtype
    for start in range(0, hidden.shape[0], chunk_tokens):
        end = min(hidden.shape[0], start + chunk_tokens)
        leaf = hidden[start:end].detach().to(dtype=head_dtype).requires_grad_(True)
        logits = head(leaf).float()
        losses = F.cross_entropy(
            logits, targets[start:end].to(logits.device), reduction="none"
        )
        if token_weights is not None:
            weights = token_weights[start:end].to(losses.device, dtype=torch.float32)
            loss = (losses * weights).sum()
            weight_total += float(weights.sum().item())
        else:
            loss = losses.sum()
            weight_total += float(end - start)
        grad = torch.autograd.grad(loss, leaf)[0]
        gradient[start:end] = grad.to(gradient.dtype)
        loss_total += loss.detach().double()
    return gradient, loss_total.float(), weight_total


def cross_entropy_from_hidden_no_grad(
    hidden: torch.Tensor,
    head: torch.nn.Module,
    targets: torch.Tensor,
    chunk_tokens: int,
) -> tuple[float, int]:
    loss_total = 0.0
    head_device = next(head.parameters()).device
    head_dtype = next(head.parameters()).dtype
    with torch.no_grad():
        for start in range(0, hidden.shape[0], chunk_tokens):
            end = min(hidden.shape[0], start + chunk_tokens)
            logits = head(hidden[start:end].to(head_device, dtype=head_dtype)).float()
            loss_total += float(
                F.cross_entropy(
                    logits, targets[start:end].to(head_device), reduction="sum"
                ).item()
            )
    return loss_total, int(hidden.shape[0])


def support_vjp_hidden_gradient(
    hidden: torch.Tensor,
    head: torch.nn.Module,
    support_ids: torch.Tensor,
    support_logit_gradient: torch.Tensor,
    chunk_tokens: int,
) -> torch.Tensor:
    gradient = torch.zeros_like(hidden)
    head_dtype = next(head.parameters()).dtype
    for start in range(0, hidden.shape[0], chunk_tokens):
        end = min(hidden.shape[0], start + chunk_tokens)
        leaf = hidden[start:end].detach().to(dtype=head_dtype).requires_grad_(True)
        logits = head(leaf)
        ids = support_ids[start:end].to(logits.device, dtype=torch.long)
        selected = logits.gather(-1, ids).float()
        cotangent = support_logit_gradient[start:end].to(
            selected.device, dtype=torch.float32
        )
        surrogate = (selected * cotangent).sum()
        grad = torch.autograd.grad(surrogate, leaf)[0]
        gradient[start:end] = grad.to(gradient.dtype)
    return gradient


def chunked_probability_statistics(
    hidden_by_expert: list[torch.Tensor],
    head: torch.nn.Module,
    targets: torch.Tensor,
    chunk_tokens: int,
    temperature: float,
    callback: Callable[[int, int, list[torch.Tensor], list[torch.Tensor]], None],
) -> None:
    """Stream per-expert log-probabilities/probabilities through a callback."""
    count = hidden_by_expert[0].shape[0]
    head_device = next(head.parameters()).device
    head_dtype = next(head.parameters()).dtype
    with torch.no_grad():
        for start in range(0, count, chunk_tokens):
            end = min(count, start + chunk_tokens)
            probabilities: list[torch.Tensor] = []
            log_probabilities: list[torch.Tensor] = []
            for hidden in hidden_by_expert:
                logits = head(
                    hidden[start:end].to(head_device, dtype=head_dtype)
                ).float()
                log_probs = F.log_softmax(logits / temperature, dim=-1)
                log_probabilities.append(log_probs)
                probabilities.append(log_probs.exp())
            callback(start, end, probabilities, log_probabilities)
