from __future__ import annotations

import unittest

import torch

from cot_mtkd.models.chunked_head import full_vocab_probe
from cot_mtkd.stage1.kneedle import build_union_support, capped_k_from_probe


def reference_union_support(
    ids: torch.Tensor, selected: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Original per-token construction, with empty supports padded by zero."""
    unions = [
        torch.unique(
            torch.cat(
                [ids[expert, token, : int(selected[expert, token])] for expert in range(ids.shape[0])]
            ),
            sorted=True,
        )
        for token in range(ids.shape[1])
    ]
    maximum = max((union.numel() for union in unions), default=0)
    support = torch.zeros((ids.shape[1], maximum), dtype=torch.int64)
    mask = torch.zeros_like(support, dtype=torch.bool)
    for token, union in enumerate(unions):
        support[token, : union.numel()] = union
        mask[token, : union.numel()] = True
        if union.numel():
            support[token, union.numel() :] = union[0]
    return support, mask


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

    def test_union_matches_reference_with_duplicates_and_variable_k(self) -> None:
        generator = torch.Generator().manual_seed(17)
        for experts, tokens, probe_k in [(1, 7, 9), (3, 23, 16), (5, 13, 31)]:
            ids = torch.randint(0, 11, (experts, tokens, probe_k), generator=generator)
            selected = torch.randint(0, probe_k + 1, (experts, tokens), generator=generator)
            # The first token has an empty support while other tokens set width.
            selected[:, 0] = 0
            actual, mask = build_union_support(ids, selected)
            expected, expected_mask = reference_union_support(ids, selected)
            self.assertTrue(torch.equal(actual, expected))
            self.assertTrue(torch.equal(mask, expected_mask))
            self.assertEqual(actual.device, ids.device)
            self.assertEqual(mask.device, ids.device)

    def test_union_preserves_slice_semantics_and_padding(self) -> None:
        ids = torch.tensor([[[9, 4, 4, 1], [8, 7, 6, 5]], [[4, 3, 2, 1], [7, 3, 9, 2]]])
        selected = torch.tensor([[-1, 99], [-99, 1]])
        actual, mask = build_union_support(ids, selected)
        expected, expected_mask = reference_union_support(ids, selected)
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(torch.equal(mask, expected_mask))
        self.assertEqual(actual[0].tolist(), [4, 9, 4, 4])

    def test_union_handles_empty_dimensions_and_all_zero_k(self) -> None:
        for shape in [(3, 0, 8), (3, 4, 0), (0, 4, 8), (3, 4, 8)]:
            ids = torch.zeros(shape, dtype=torch.int64)
            selected = torch.zeros(shape[:2], dtype=torch.int64)
            support, mask = build_union_support(ids, selected)
            self.assertEqual(support.shape, (shape[1], 0))
            self.assertEqual(mask.shape, support.shape)
            self.assertEqual(support.dtype, torch.int64)
            self.assertEqual(mask.dtype, torch.bool)

    def test_union_matches_reference_for_training_probe_and_cap(self) -> None:
        generator = torch.Generator().manual_seed(29)
        ids = torch.randint(0, 152064, (3, 113, 512), generator=generator)
        selected = torch.full((3, 113), 128, dtype=torch.int64)
        actual, mask = build_union_support(ids, selected)
        expected, expected_mask = reference_union_support(ids, selected)
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(torch.equal(mask, expected_mask))
        self.assertEqual(actual.shape, (113, 384))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_union_stays_on_cuda_and_matches_cpu_reference(self) -> None:
        generator = torch.Generator().manual_seed(23)
        ids = torch.randint(0, 50, (3, 113, 512), generator=generator)
        selected = torch.randint(0, 129, (3, 113), generator=generator)
        selected[:, 0] = 0
        expected, expected_mask = reference_union_support(ids, selected)
        actual, mask = build_union_support(ids.cuda(), selected.cuda())
        self.assertEqual(actual.device.type, "cuda")
        self.assertEqual(mask.device.type, "cuda")
        self.assertTrue(torch.equal(actual.cpu(), expected))
        self.assertTrue(torch.equal(mask.cpu(), expected_mask))

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
