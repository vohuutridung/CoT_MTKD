from __future__ import annotations

import inspect
import unittest
from collections import OrderedDict

import torch

from cot_mtkd.data.collator import shifted_token_views
from cot_mtkd.data.schema import TokenRegion
from cot_mtkd.stage1.dpp import step_dpp_loss
from cot_mtkd.stage1.dropout import step_level_token_weights
from cot_mtkd.stage1.gac_gradient import phase1_data_gradients
from cot_mtkd.stage1.grassmann import (
    grassmann_repulsion_updates,
    grassmann_squared_distances,
)
from cot_mtkd.stage1.trainer import probe_stage1_dpp


def _group(a: torch.Tensor, b: torch.Tensor) -> OrderedDict[str, torch.nn.Parameter]:
    return OrderedDict(
        (
            ("layer.lora_A.weight", torch.nn.Parameter(a.clone().float())),
            ("layer.lora_B.weight", torch.nn.Parameter(b.clone().float())),
        )
    )


def _batch_from_regions(regions: list[int], steps: list[int]) -> dict[str, torch.Tensor]:
    return {
        "labels": torch.tensor([[-100, *range(len(regions))]]),
        "region_ids": torch.tensor([[int(TokenRegion.PROMPT), *regions]]),
        "step_ids": torch.tensor([[-1, *steps]]),
    }


def _sft_mean(
    token_nll: torch.Tensor, weights: torch.Tensor, contributing: int
) -> torch.Tensor:
    return (token_nll * weights).sum() / max(contributing, 1)


def _weights_with_dropped(
    batch: dict[str, torch.Tensor],
    dropped: int,
    drop_probability: float = 0.5,
) -> tuple[torch.Tensor, int, int]:
    for seed in range(10_000):
        weights, count, actual = step_level_token_weights(
            batch,
            expert_index=0,
            base_seed=seed,
            global_step=0,
            rng_stream=0,
            drop_probability=drop_probability,
        )
        if actual == dropped:
            return weights, count, actual
    raise AssertionError(f"could not find a dropout mask with dropped={dropped}")


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

        dropped_weights, dropped_segment_count, dropped_n = step_level_token_weights(
            batch, expert_index=0, base_seed=7, global_step=3, rng_stream=0,
            drop_probability=0.9,
        )
        assert dropped_segment_count == 4 - dropped_n
        assert torch.isclose(
            dropped_weights.sum(), torch.tensor(float(dropped_segment_count))
        )
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

    def test_sft_mean_without_dropout_uses_all_steps_and_answer(self) -> None:
        batch = _batch_from_regions(
            [int(TokenRegion.REASONING)] * 3 + [int(TokenRegion.ANSWER)],
            [0, 1, 2, -1],
        )
        weights, count, dropped = step_level_token_weights(
            batch, expert_index=0, base_seed=1, global_step=0, rng_stream=0,
            drop_probability=0.0,
        )
        token_nll = torch.tensor([1.0, 2.0, 3.0, 4.0])
        assert dropped == 0
        assert count == 4
        assert torch.isclose(_sft_mean(token_nll, weights, count), torch.tensor(2.5))

    def test_sft_mean_excludes_dropped_step_from_denominator(self) -> None:
        batch = _batch_from_regions(
            [int(TokenRegion.REASONING)] * 3 + [int(TokenRegion.ANSWER)],
            [0, 1, 2, -1],
        )
        weights, count, dropped = _weights_with_dropped(batch, dropped=1)
        token_nll = torch.tensor([10.0, 20.0, 30.0, 40.0])
        actual = _sft_mean(token_nll, weights, count)
        expected = (token_nll * weights).sum() / 3.0
        diluted = (token_nll * weights).sum() / 4.0
        assert dropped == 1
        assert count == 3
        assert torch.isclose(actual, expected)
        assert not torch.isclose(actual, diluted)

    def test_sft_mean_all_reasoning_dropped_is_answer_only(self) -> None:
        batch = _batch_from_regions(
            [int(TokenRegion.REASONING)] * 3 + [int(TokenRegion.ANSWER)],
            [0, 1, 2, -1],
        )
        weights, count, dropped = _weights_with_dropped(
            batch, dropped=3, drop_probability=0.9
        )
        token_nll = torch.tensor([10.0, 20.0, 30.0, 40.0])
        assert dropped == 3
        assert count == 1
        assert torch.isclose(weights[:3].sum(), torch.tensor(0.0))
        assert torch.isclose(_sft_mean(token_nll, weights, count), torch.tensor(40.0))

    def test_sft_mean_equalizes_variable_step_lengths(self) -> None:
        regions = (
            [int(TokenRegion.REASONING)] * 100
            + [int(TokenRegion.REASONING)] * 10
            + [int(TokenRegion.ANSWER)]
        )
        steps = [0] * 100 + [1] * 10 + [-1]
        batch = _batch_from_regions(regions, steps)
        weights, count, dropped = step_level_token_weights(
            batch, expert_index=0, base_seed=1, global_step=0, rng_stream=0,
            drop_probability=0.0,
        )
        token_nll = torch.tensor([1.0] * 100 + [2.0] * 10 + [3.0])
        assert dropped == 0
        assert count == 3
        assert torch.isclose(weights[:100].sum(), torch.tensor(1.0))
        assert torch.isclose(weights[100:110].sum(), torch.tensor(1.0))
        assert torch.isclose(_sft_mean(token_nll, weights, count), torch.tensor(2.0))
        token_average = token_nll.mean()
        assert not torch.isclose(_sft_mean(token_nll, weights, count), token_average)

    def test_answer_stays_always_on_under_reasoning_dropout(self) -> None:
        batch = _batch_from_regions(
            [int(TokenRegion.REASONING)] * 3 + [int(TokenRegion.ANSWER)] * 2,
            [0, 1, 2, -1, -1],
        )
        full, _, _ = step_level_token_weights(
            batch, expert_index=0, base_seed=3, global_step=1, rng_stream=0,
            drop_probability=0.0,
        )
        dropped_weights, count, dropped = _weights_with_dropped(
            batch, dropped=2, drop_probability=0.8
        )
        assert dropped == 2
        assert count == 2
        assert torch.allclose(dropped_weights[-2:], full[-2:])
        assert torch.allclose(dropped_weights[-2:], torch.tensor([0.5, 0.5]))

    def test_dpp_does_not_use_sft_dropout_mask(self) -> None:
        assert "drop" not in inspect.signature(step_dpp_loss).parameters
        assert "drop_probability" not in inspect.signature(probe_stage1_dpp).parameters
        batch = self._many_steps_batch()
        kwargs = dict(base_seed=44, global_step=2, rng_stream=3, drop_probability=0.5)
        weights_a, _, _ = step_level_token_weights(batch, expert_index=0, **kwargs)
        weights_b, _, _ = step_level_token_weights(batch, expert_index=1, **kwargs)
        assert not torch.equal(weights_a[:-1], weights_b[:-1])
        features = torch.eye(3).unsqueeze(1).repeat(1, 4, 1)
        samples = torch.zeros(4, dtype=torch.long)
        steps = torch.zeros(4, dtype=torch.long)
        loss_a, _ = step_dpp_loss(features, samples, steps)
        loss_b, _ = step_dpp_loss(features, samples, steps)
        assert torch.equal(loss_a, loss_b)

    def test_grassmann_force_is_outside_data_gradient(self) -> None:
        task = [[torch.tensor([1.0, -2.0])], [torch.tensor([0.5, 0.5])]]
        diversity = [[torch.tensor([3.0, 3.0])], [torch.tensor([4.0, -1.0])]]
        force = [[torch.tensor([100.0, -100.0])], [torch.tensor([50.0, 50.0])]]
        final, diagnostics = phase1_data_gradients(
            task, diversity, force, diversity_weight=0.2
        )
        assert torch.allclose(final[0][0], torch.tensor([1.6, -1.4]))
        assert torch.allclose(final[1][0], torch.tensor([1.3, 0.3]))
        assert diagnostics.repulsion_norms[0] > 0.0
        assert diagnostics.repulsion_norms[1] > 0.0


if __name__ == "__main__":
    unittest.main()
