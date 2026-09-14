from __future__ import annotations

from dataclasses import asdict
from typing import Any, Sequence

import torch

from .schema import PreparedRecord, TokenRegion


class LongCoTCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = int(pad_token_id)

    def __call__(self, records: Sequence[PreparedRecord]) -> dict[str, Any]:
        max_length = max(len(record.input_ids) for record in records)

        def padded(values: list[int], fill: int) -> list[int]:
            return values + [fill] * (max_length - len(values))

        return {
            "sample_ids": [record.sample_id for record in records],
            "input_ids": torch.tensor(
                [padded(record.input_ids, self.pad_token_id) for record in records],
                dtype=torch.long,
            ),
            "labels": torch.tensor(
                [padded(record.labels, -100) for record in records], dtype=torch.long
            ),
            "attention_mask": torch.tensor(
                [padded(record.attention_mask, 0) for record in records],
                dtype=torch.long,
            ),
            "region_ids": torch.tensor(
                [
                    padded(record.region_ids, int(TokenRegion.PADDING))
                    for record in records
                ],
                dtype=torch.int16,
            ),
            "step_ids": torch.tensor(
                [padded(record.step_ids, -1) for record in records], dtype=torch.int32
            ),
            "records": [asdict(record) for record in records],
        }


def shifted_token_views(batch: dict[str, Any]) -> dict[str, torch.Tensor]:
    labels = batch["labels"][:, 1:]
    regions = batch["region_ids"][:, 1:]
    steps = batch["step_ids"][:, 1:]
    valid = labels.ne(-100)
    reasoning = regions.eq(int(TokenRegion.REASONING)) & valid
    batch_indices, token_indices = valid.nonzero(as_tuple=True)
    reasoning_batch, reasoning_token = reasoning.nonzero(as_tuple=True)
    return {
        "labels": labels,
        "regions": regions,
        "steps": steps,
        "valid": valid,
        "reasoning": reasoning,
        "response_batch_indices": batch_indices,
        "response_hidden_indices": token_indices,
        "reasoning_batch_indices": reasoning_batch,
        "reasoning_hidden_indices": reasoning_token,
        "response_targets": labels[valid],
        "reasoning_targets": labels[reasoning],
        "reasoning_step_ids": steps[reasoning],
    }
