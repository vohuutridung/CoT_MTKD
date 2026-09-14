from __future__ import annotations

from typing import Any

import torch

from ..data.schema import PreparedRecord
from ..data.serialize import pag_answer_prefix
from ..models.chunked_head import (
    cross_entropy_from_hidden_no_grad,
    decoder_and_lm_head,
    forward_hidden,
)


def _crop_legacy_cache(past_key_values: Any, length: int) -> Any:
    cropped_layers = []
    for layer in past_key_values:
        cropped = []
        for value in layer:
            if torch.is_tensor(value) and value.ndim >= 3:
                cropped.append(value[..., :length, :])
            else:
                cropped.append(value)
        cropped_layers.append(tuple(cropped))
    return tuple(cropped_layers)


def crop_cache(past_key_values: Any, length: int) -> Any:
    if hasattr(past_key_values, "crop"):
        past_key_values.crop(length)
        return past_key_values
    return _crop_legacy_cache(past_key_values, length)


def record_pag_parts(
    record: PreparedRecord, tokenizer: Any
) -> tuple[list[int], list[list[int]], list[int], list[int]]:
    positions_by_step: list[list[int]] = []
    for step_id in sorted({value for value in record.step_ids if value >= 0}):
        positions_by_step.append(
            [index for index, value in enumerate(record.step_ids) if value == step_id]
        )
    first_reasoning = min(
        (index for positions in positions_by_step for index in positions),
        default=record.answer_start,
    )
    prefix = record.input_ids[:first_reasoning]
    steps = [
        [record.input_ids[index] for index in positions]
        for positions in positions_by_step
    ]
    # Tokenize the literal answer prefix and raw solution jointly: BPE at this
    # boundary is context-sensitive. A token crossing the character boundary is
    # counted as the first solution target, preserving the exact continuation.
    prefix_text = pag_answer_prefix()
    continuation = tokenizer(
        prefix_text + record.solution,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    continuation_ids = list(continuation["input_ids"])
    offsets = [tuple(pair) for pair in continuation["offset_mapping"]]
    solution_start = next(
        (index for index, (_, end) in enumerate(offsets) if end > len(prefix_text)),
        len(continuation_ids),
    )
    answer_prefix = continuation_ids[:solution_start]
    solution = continuation_ids[solution_start:]
    if not answer_prefix:
        raise ValueError("PAG literal answer prefix tokenized to an empty sequence")
    if not solution:
        raise ValueError(
            f"Gold solution tokenized to an empty sequence for {record.sample_id}"
        )
    return prefix, steps, list(answer_prefix), list(solution)


def _score_solution_from_prefix_cache(
    model: torch.nn.Module,
    cache: Any,
    prefix_length: int,
    answer_prefix: list[int],
    solution: list[int],
    device: torch.device,
    chunk_tokens: int,
) -> tuple[float, Any]:
    continuation = torch.tensor(
        [answer_prefix + solution], device=device, dtype=torch.long
    )
    cache = crop_cache(cache, prefix_length)
    outputs = forward_hidden(
        model,
        continuation,
        attention_mask=None,
        use_cache=True,
        past_key_values=cache,
    )
    answer_length = len(answer_prefix)
    if answer_length < 1:
        raise ValueError("PAG answer prefix must contain at least one token")
    solution_hidden = outputs.last_hidden_state[
        0, answer_length - 1 : answer_length + len(solution) - 1
    ]
    _, head = decoder_and_lm_head(model)
    loss_sum, count = cross_entropy_from_hidden_no_grad(
        solution_hidden,
        head,
        torch.tensor(solution, device=device, dtype=torch.long),
        chunk_tokens,
    )
    cache = crop_cache(outputs.past_key_values, prefix_length)
    return loss_sum / max(count, 1), cache


def score_reference_solution_pag(
    model: torch.nn.Module,
    record: PreparedRecord,
    tokenizer: Any,
    device: torch.device,
    chunk_tokens: int,
) -> torch.Tensor:
    prefix, steps, answer_prefix, solution = record_pag_parts(record, tokenizer)
    if not steps:
        return torch.empty(0, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        prefix_tensor = torch.tensor([prefix], device=device, dtype=torch.long)
        prefix_output = forward_hidden(model, prefix_tensor, None, use_cache=True)
        cache = prefix_output.past_key_values
        prefix_length = len(prefix)
        nll_values: list[float] = []
        nll, cache = _score_solution_from_prefix_cache(
            model,
            cache,
            prefix_length,
            answer_prefix,
            solution,
            device,
            chunk_tokens,
        )
        nll_values.append(nll)
        for step in steps:
            cache = crop_cache(cache, prefix_length)
            step_tensor = torch.tensor([step], device=device, dtype=torch.long)
            step_output = forward_hidden(
                model,
                step_tensor,
                attention_mask=None,
                use_cache=True,
                past_key_values=cache,
            )
            prefix_length += len(step)
            cache = step_output.past_key_values
            nll, cache = _score_solution_from_prefix_cache(
                model,
                cache,
                prefix_length,
                answer_prefix,
                solution,
                device,
                chunk_tokens,
            )
            nll_values.append(nll)
    values = torch.tensor(nll_values, dtype=torch.float32)
    return values[:-1] - values[1:]
