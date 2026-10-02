from __future__ import annotations

import unittest

import torch

from cot_mtkd.models.chunked_head import full_vocab_probe
from cot_mtkd.stage1.kneedle import (
    build_union_support,
    capped_k_from_probe,
    council_kneedle_candidates,
    dpp_support_rates,
)


class KneedleTest(unittest.TestCase):
    def test_flat_logits_use_minimum_k(self) -> None:
        probe = torch.zeros(3, 16)
        selected = capped_k_from_probe(
            probe,
            torch.zeros(3),
            torch.zeros(3),
            vocab_size=100,
            min_k=5,
            max_k=12,
        )
        self.assertTrue(torch.equal(selected, torch.full((3,), 5)))

    def test_union_contains_each_expert_support(self) -> None:
        ids = torch.tensor(
            [
                [[1, 2, 3, 4], [3, 4, 5, 6]],
                [[2, 7, 8, 9], [4, 8, 9, 10]],
            ]
        )
        selected = torch.tensor([[2, 3], [2, 2]])
        support, mask = build_union_support(ids, selected)
        self.assertEqual(set(support[0, mask[0]].tolist()), {1, 2, 7})
        self.assertEqual(set(support[1, mask[1]].tolist()), {3, 4, 5, 8})

    def test_sharp_elbow_is_clipped_to_upper_cap(self) -> None:
        probe = torch.tensor([[1.0] * 10 + [0.0] * 6])
        selected = capped_k_from_probe(
            probe,
            torch.zeros(1),
            torch.ones(1),
            vocab_size=100,
            min_k=5,
            max_k=8,
        )
        self.assertEqual(selected.item(), 8)

    def test_flat_council_distribution_keeps_every_rank(self) -> None:
        probabilities = torch.full((2, 6), 1.0 / 6.0)
        targets = torch.zeros(2, dtype=torch.long)
        _, elbow, _, _ = council_kneedle_candidates(probabilities, targets)
        self.assertTrue(torch.equal(elbow, torch.full((2,), 6)))
        _, capped, _, _ = council_kneedle_candidates(probabilities, targets, k_max=3)
        self.assertTrue(torch.equal(capped, torch.full((2,), 3)))

    def test_cap_equal_to_vocab_matches_uncapped_k(self) -> None:
        generator = torch.Generator().manual_seed(4)
        logits = torch.randn(5, 11, generator=generator)
        probabilities = torch.softmax(logits, dim=-1)
        targets = torch.randint(0, 11, (5,), generator=generator)
        _, full, _, _ = council_kneedle_candidates(probabilities, targets, k_max=None)
        _, capped, _, _ = council_kneedle_candidates(probabilities, targets, k_max=11)
        self.assertTrue(torch.equal(full, capped))

    def test_capped_axis_and_support_stats_on_synthetic_logits(self) -> None:
        # Top-4 masses 0.50, 0.20, 0.15, 0.10; six leftover tokens share 0.05.
        # With N'=4 the elbow is k=2, so V_k is the first two ids.
        top = torch.tensor([0.50, 0.20, 0.15, 0.10])
        rest = torch.full((6,), 0.05 / 6.0)
        row = torch.cat([top, rest])
        logits = row.log().unsqueeze(0).repeat(2, 1)
        probabilities = torch.softmax(logits, dim=-1)
        targets = torch.tensor([0, 9])
        candidates, elbow, tail, outside = council_kneedle_candidates(
            probabilities, targets, k_max=4
        )
        self.assertTrue(torch.equal(elbow, torch.tensor([2, 2])))
        self.assertAlmostEqual(float(tail[0]), 0.30, places=5)
        self.assertAlmostEqual(float(tail[1]), 0.30, places=5)
        self.assertFalse(bool(outside[0]))
        self.assertTrue(bool(outside[1]))
        self.assertNotIn(0, candidates[0].tolist())
        self.assertIn(1, candidates[0].tolist())
        tail_mass, outside_rate = dpp_support_rates(
            float(tail.sum()), float(outside.sum()), float(tail.numel())
        )
        self.assertAlmostEqual(tail_mass, 0.30, places=5)
        self.assertAlmostEqual(outside_rate, 0.5, places=6)

    def test_full_vocab_probe_excludes_gold_target(self) -> None:
        head = torch.nn.Linear(3, 6, bias=False)
        with torch.no_grad():
            head.weight.copy_(
                torch.tensor(
                    [
                        [0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0],
                        [2.0, 0.0, 0.0],
                        [3.0, 0.0, 0.0],
                        [4.0, 0.0, 0.0],
                        [5.0, 0.0, 0.0],
                    ]
                )
            )
        _, ids, minimum, maximum = full_vocab_probe(
            torch.tensor([[1.0, 0.0, 0.0]]),
            head,
            targets=torch.tensor([5]),
            probe_k=3,
            chunk_tokens=1,
        )
        self.assertNotIn(5, ids[0].tolist())
        self.assertEqual(float(minimum.item()), 0.0)
        self.assertEqual(float(maximum.item()), 4.0)


if __name__ == "__main__":
    unittest.main()
