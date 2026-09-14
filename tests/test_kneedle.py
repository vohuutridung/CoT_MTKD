from __future__ import annotations

import unittest

import torch

from cot_mtkd.models.chunked_head import full_vocab_probe
from cot_mtkd.stage1.kneedle import build_union_support, capped_k_from_probe


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
