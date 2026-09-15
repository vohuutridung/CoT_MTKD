from __future__ import annotations

import torch

from ..data.collator import shifted_token_views
from ..data.schema import TokenRegion
from ..utils.seed import derived_seed


def step_level_token_weights(
    batch: dict[str, torch.Tensor],
    expert_index: int,
    base_seed: int,
    global_step: int,
    rng_stream: int,
    drop_probability: float,
) -> tuple[torch.Tensor, int, int]:
    """Length-normalized step NLL with one Bernoulli mask per step and expert.

    Answer tokens form their own fixed-weight block. Format/control tokens
    form a separate always-on block, preserving the hard-loss token contract
    without diluting final-answer NLL with delimiters or markers.
    Returns weights in `response_targets` order, the fixed segment count for
    batch normalization, and the number of dropped reasoning steps.
    """
    if not 0.0 <= drop_probability < 1.0:
        raise ValueError("step dropout probability must be in [0, 1)")
    views = shifted_token_views(batch)
    valid = views["valid"]
    batch_ids = views["response_batch_indices"]
    regions = views["regions"][valid]
    step_ids = views["steps"][valid]
    weights = torch.zeros(batch_ids.numel(), device=batch_ids.device, dtype=torch.float32)
    reasoning = regions.eq(int(TokenRegion.REASONING))
    pairs = torch.stack((batch_ids[reasoning], step_ids[reasoning]), dim=-1)
    unique_pairs = torch.unique(pairs, dim=0)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(
        derived_seed(base_seed, "stage1_step_dropout", global_step, rng_stream, expert_index)
    )
    retained = torch.rand(unique_pairs.shape[0], generator=generator) >= drop_probability
    dropped = int((~retained).sum().item())
    for index, pair in enumerate(unique_pairs):
        mask = reasoning & batch_ids.eq(pair[0]) & step_ids.eq(pair[1])
        if bool(retained[index]):
            weights[mask] = 1.0 / int(mask.sum().item())
    answer = regions.eq(int(TokenRegion.ANSWER))
    fixed_blocks = 0
    for sample in torch.unique(batch_ids):
        for region_mask in (answer, ~reasoning & ~answer):
            mask = batch_ids.eq(sample) & region_mask
            if mask.any():
                weights[mask] = 1.0 / int(mask.sum().item())
                fixed_blocks += 1
    return weights, int(unique_pairs.shape[0]) + fixed_blocks, dropped
