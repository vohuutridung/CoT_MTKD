from __future__ import annotations

from collections import OrderedDict

import unittest

import torch

from cot_mtkd.data.collator import shifted_token_views
from cot_mtkd.data.schema import TokenRegion
from cot_mtkd.stage1.dpp import step_dpp_loss
from cot_mtkd.stage1.dropout import step_level_token_weights
from cot_mtkd.stage1.grassmann import (
    grassmann_repulsion_updates,
    grassmann_squared_distances,
)


def _group(a: torch.Tensor, b: torch.Tensor) -> OrderedDict[str, torch.nn.Parameter]:
    return OrderedDict(
        (
            ("layer.lora_A.weight", torch.nn.Parameter(a.clone().float())),
            ("layer.lora_B.weight", torch.nn.Parameter(b.clone().float())),
        )
    )


class Phase1ProposalTest(unittest.TestCase):
    def test_grassmann_distance_ignores_invertible_lora_basis_change(self) -> None:
        generator = torch.Generator().manual_seed(23)
        groups = [
            _group(
                torch.randn(2, 5, generator=generator),
                torch.randn(6, 2, generator=generator),
            )
            for _ in range(3)
        ]
        original = grassmann_squared_distances(groups, rank_epsilon=1.0e-7)
        transform = torch.tensor([[1.7, 0.2], [-0.3, 0.9]])
        changed = list(groups)
        changed[1] = _group(
            transform @ groups[1]["layer.lora_A.weight"].detach(),
            groups[1]["layer.lora_B.weight"].detach() @ torch.linalg.inv(transform),
        )
        transformed = grassmann_squared_distances(changed, rank_epsilon=1.0e-7)
        assert torch.allclose(original, transformed, atol=2.0e-4, rtol=2.0e-4)
        assert torch.allclose(original, original.T)


    def test_grassmann_force_moves_experts_apart_and_handles_zero_b(self) -> None:
        groups = [
            _group(torch.tensor([[1.0, 0.0, 0.0]]), torch.tensor([[1.0], [0.0], [0.0]])),
            _group(torch.tensor([[0.8, 0.6, 0.0]]), torch.tensor([[0.8], [0.0], [0.6]])),
        ]
        before = grassmann_squared_distances(groups)[0, 1]
        updates, kernel, distances, bandwidth = grassmann_repulsion_updates(groups)
        assert bandwidth > 0.0
        assert torch.isfinite(kernel).all()
        assert torch.isfinite(distances).all()
        assert all(torch.isfinite(force).all() for expert in updates for force in expert)
        with torch.no_grad():
            for group, forces in zip(groups, updates, strict=True):
                for parameter, force in zip(group.values(), forces, strict=True):
                    parameter.add_(0.01 * force)
        after = grassmann_squared_distances(groups)[0, 1]
        assert after > before

        zero_b = [
            _group(torch.tensor([[1.0, 0.0, 0.0]]), torch.zeros(3, 1)),
            _group(torch.tensor([[0.8, 0.6, 0.0]]), torch.zeros(3, 1)),
        ]
        forces, kernel, distances, bandwidth = grassmann_repulsion_updates(zero_b)
        assert bandwidth > 0.0
        assert torch.isfinite(kernel).all()
        assert torch.isfinite(distances).all()
        assert all(torch.isfinite(force).all() for expert in forces for force in expert)


    def _short_batch(self) -> dict[str, torch.Tensor]:
        return {
            "labels": torch.tensor([[-100, 10, 11, 12, 13, 14, 15, 16]]),
            "region_ids": torch.tensor(
                [[
                    int(TokenRegion.PROMPT),
                    int(TokenRegion.REASONING),
                    int(TokenRegion.REASONING),
                    int(TokenRegion.DELIMITER),
                    int(TokenRegion.REASONING),
                    int(TokenRegion.ANSWER_MARKER),
                    int(TokenRegion.ANSWER),
                    int(TokenRegion.EOS),
                ]]
            ),
            "step_ids": torch.tensor([[-1, 0, 0, 0, 1, -1, -1, -1]]),
        }


    def test_step_dropout_normalizes_each_step_and_keeps_final_block_weight_one(self) -> None:
        batch = self._short_batch()
        weights, segment_count, dropped = step_level_token_weights(
            batch, expert_index=0, base_seed=7, global_step=3, rng_stream=0,
            drop_probability=0.0,
        )
        assert dropped == 0
        assert segment_count == 4
        assert torch.allclose(
            weights, torch.tensor([0.5, 0.5, 1.0 / 3.0, 1.0, 1.0 / 3.0, 1.0, 1.0 / 3.0])
        )
        assert torch.isclose(weights.sum(), torch.tensor(4.0))

        dropped_weights, dropped_segment_count, _ = step_level_token_weights(
            batch, expert_index=0, base_seed=7, global_step=3, rng_stream=0,
            drop_probability=0.9,
        )
        assert dropped_segment_count == segment_count
        assert torch.allclose(dropped_weights[[2, 4, 5, 6]], weights[[2, 4, 5, 6]])


    def _many_steps_batch(self) -> dict[str, torch.Tensor]:
        reasoning_count = 24
        return {
            "labels": torch.tensor([[-100] + list(range(1, reasoning_count + 2))]),
            "region_ids": torch.tensor([[
                int(TokenRegion.PROMPT),
                *([int(TokenRegion.REASONING)] * reasoning_count),
                int(TokenRegion.ANSWER),
            ]]),
            "step_ids": torch.tensor([[-1, *range(reasoning_count), -1]]),
        }


    def test_step_dropout_masks_differ_by_expert_and_replay_deterministically(self) -> None:
        batch = self._many_steps_batch()
        kwargs = dict(base_seed=928, global_step=4, rng_stream=1, drop_probability=0.5)
        expert_zero = step_level_token_weights(batch, expert_index=0, **kwargs)[0]
        replay = step_level_token_weights(batch, expert_index=0, **kwargs)[0]
        expert_one = step_level_token_weights(batch, expert_index=1, **kwargs)[0]
        reasoning = shifted_token_views(batch)["reasoning"]
        assert torch.equal(expert_zero, replay)
        assert not torch.equal(expert_zero[:-1], expert_one[:-1])
        assert expert_zero[-1] == expert_one[-1] == 1.0
        assert int(reasoning.sum().item()) == expert_zero.numel() - 1


    def test_dpp_averages_token_logdets_before_step_reduction(self) -> None:
        features = torch.tensor(
            [
                [[1.0, 0.0], [1.0, 0.0]],
                [[1.0, 0.0], [0.0, 1.0]],
            ]
        )
        sample_ids = torch.zeros(2, dtype=torch.long)
        step_ids = torch.zeros(2, dtype=torch.long)
        jitter = 1.0e-4
        actual, _ = step_dpp_loss(features, sample_ids, step_ids, jitter=jitter)
        identity = torch.eye(2)
        token_grams = [features[:, token] @ features[:, token].T for token in range(2)]
        expected = -torch.stack(
            [torch.linalg.slogdet(gram + jitter * identity).logabsdet for gram in token_grams]
        ).mean()
        old_step_gram = torch.stack(token_grams).mean(dim=0)
        old_aggregated = -torch.linalg.slogdet(old_step_gram + jitter * identity).logabsdet
        assert torch.allclose(actual, expected, atol=1.0e-3, rtol=1.0e-4)
        assert actual > old_aggregated + 1.0


if __name__ == "__main__":
    unittest.main()
