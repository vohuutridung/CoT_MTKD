from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from cot_mtkd.models.chunked_head import full_vocab_probe
from cot_mtkd.signals.predictive import _dpp_uniqueness, gather_support_logits
from cot_mtkd.stage1.kneedle import build_union_support, local_k_from_probe
from cot_mtkd.stage1.trainer import probe_stage1_dpp


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
        raw, selected = local_k_from_probe(probe)
        self.assertTrue(torch.equal(raw, torch.ones(3, dtype=torch.long)))
        self.assertTrue(torch.equal(selected, torch.full((3,), 8)))

    def test_small_and_empty_windows_never_exceed_candidates(self) -> None:
        for window in (0, 1, 2, 7, 8):
            with self.subTest(window=window):
                raw, selected = local_k_from_probe(torch.zeros(3, window))
                self.assertTrue(torch.equal(raw, torch.full((3,), min(window, 1))))
                self.assertTrue(torch.equal(selected, torch.full((3,), window)))
                self.assertEqual(selected.dtype, torch.int64)
        raw, selected = local_k_from_probe(torch.empty(0, 512))
        self.assertEqual(raw.shape, (0,))
        self.assertEqual(selected.shape, (0,))

    def test_sharp_head_retains_lower_bound(self) -> None:
        raw, selected = local_k_from_probe(torch.tensor([[20.0] + [0.0] * 511]))
        self.assertEqual(raw.item(), 2)
        self.assertEqual(selected.item(), 8)

    def test_deep_knee_is_not_clipped(self) -> None:
        probe = torch.tensor([[10.0] * 300 + [0.0] * 212])
        raw, selected = local_k_from_probe(probe)
        self.assertEqual(raw.item(), 301)
        self.assertEqual(selected.item(), 301)

    def test_nearly_flat_logits_use_local_span_and_additive_epsilon(self) -> None:
        # A tiny absolute span still has a meaningful local drop. Adding epsilon
        # to the span matters once the span becomes smaller than epsilon.
        for amplitude, expected in ((1e-9, 301), (1e-13, 1)):
            with self.subTest(amplitude=amplitude):
                probe = torch.tensor([[amplitude] * 300 + [0.0] * 212], dtype=torch.float64)
                raw, selected = local_k_from_probe(probe)
                self.assertEqual(raw.item(), expected)
                self.assertEqual(selected.item(), max(expected, 8))

    def test_both_axes_match_independent_local_reference(self) -> None:
        generator = torch.Generator().manual_seed(71)
        for window in (9, 31, 512):
            logits = torch.randn(7, window, generator=generator, dtype=torch.float64)
            logits = logits.sort(dim=-1, descending=True).values
            x = torch.arange(window, dtype=torch.float64) / (window - 1)
            u = (logits - logits[:, -1:]) / (logits[:, :1] - logits[:, -1:] + 1e-12)
            expected = ((1 - x) - u).argmax(dim=-1) + 1
            raw, selected = local_k_from_probe(logits)
            self.assertTrue(torch.equal(raw, expected))
            self.assertTrue(torch.equal(selected, expected.clamp(min=8, max=window)))

    def test_shift_and_scale_of_local_logits_preserve_knee(self) -> None:
        probe = torch.tensor([[10.0] * 200 + [0.0] * 312], dtype=torch.float64)
        raw, selected = local_k_from_probe(probe)
        for transformed in (probe + 1234.0, probe * 37.0):
            actual_raw, actual_selected = local_k_from_probe(transformed)
            self.assertTrue(torch.equal(actual_raw, raw))
            self.assertTrue(torch.equal(actual_selected, selected))

    def test_selection_is_detached_and_support_uses_full_window(self) -> None:
        probe = torch.zeros(2, 512, requires_grad=True)
        raw, selected = local_k_from_probe(probe, k_min=512)
        self.assertFalse(raw.requires_grad)
        self.assertFalse(selected.requires_grad)
        self.assertTrue(torch.equal(selected, torch.full((2,), 512)))

    def test_invalid_selection_arguments(self) -> None:
        for probe, arguments in (
            (torch.zeros(16), {}),
            (torch.zeros(1, 513), {}),
            (torch.zeros(1, 16), {"k_min": 0}),
            (torch.zeros(1, 16), {"epsilon": 0}),
        ):
            with self.subTest(shape=probe.shape, arguments=arguments), self.assertRaises(ValueError):
                local_k_from_probe(probe, **arguments)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_local_selection_stays_on_cuda_and_matches_cpu(self) -> None:
        probe = torch.tensor([[10.0] * 300 + [0.0] * 212, [0.0] * 512])
        expected = local_k_from_probe(probe)
        actual = local_k_from_probe(probe.cuda())
        for cpu, cuda in zip(expected, actual, strict=True):
            self.assertEqual(cuda.device.type, "cuda")
            self.assertTrue(torch.equal(cpu, cuda.cpu()))

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

    def test_disjoint_full_window_supports_can_have_1536_union_tokens(self) -> None:
        ids = torch.arange(1536).reshape(3, 1, 512).expand(3, 113, 512)
        selected = torch.full((3, 113), 512, dtype=torch.int64)
        actual, mask = build_union_support(ids, selected)
        expected, expected_mask = reference_union_support(ids, selected)
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(torch.equal(mask, expected_mask))
        self.assertEqual(actual.shape, (113, 1536))
        self.assertTrue(mask.all())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_union_stays_on_cuda_and_matches_cpu_reference(self) -> None:
        generator = torch.Generator().manual_seed(23)
        ids = torch.randint(0, 50, (3, 113, 512), generator=generator)
        selected = torch.randint(0, 513, (3, 113), generator=generator)
        selected[:, 0] = 0
        expected, expected_mask = reference_union_support(ids, selected)
        actual, mask = build_union_support(ids.cuda(), selected.cuda())
        self.assertEqual(actual.device.type, "cuda")
        self.assertEqual(mask.device.type, "cuda")
        self.assertTrue(torch.equal(actual.cpu(), expected))
        self.assertTrue(torch.equal(mask.cpu(), expected_mask))

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
        values, ids = full_vocab_probe(
            torch.tensor([[1.0, 0.0, 0.0]]),
            head,
            targets=torch.tensor([5]),
            probe_k=512,
            chunk_tokens=1,
        )
        self.assertNotIn(5, ids[0].tolist())
        torch.testing.assert_close(values, torch.tensor([[4.0, 3.0, 2.0, 1.0, 0.0]]))
        raw, selected = local_k_from_probe(values)
        self.assertEqual(raw.item(), 1)
        self.assertEqual(selected.item(), 5)

    def test_probe_ignores_vocabulary_tail_outside_local_window(self) -> None:
        head = torch.nn.Linear(1, 800, bias=False)
        with torch.no_grad():
            head.weight[:, 0] = torch.tensor([99.0] + [10.0] * 300 + [0.0] * 212 + [-1.0] * 287)
        hidden, target = torch.ones(1, 1), torch.zeros(1, dtype=torch.long)
        before, before_ids = full_vocab_probe(hidden, head, target, 512, 1)
        with torch.no_grad():
            head.weight[513:, 0] = -1e6
        after, after_ids = full_vocab_probe(hidden, head, target, 512, 1)
        self.assertTrue(torch.equal(before, after))
        self.assertTrue(torch.equal(before_ids, after_ids))
        for values in (before, after):
            raw, selected = local_k_from_probe(values)
            self.assertEqual(raw.item(), 301)
            self.assertEqual(selected.item(), 301)

    def test_probe_empty_tokens_and_invalid_vocabulary(self) -> None:
        values, ids = full_vocab_probe(
            torch.empty(0, 2), torch.nn.Linear(2, 4), torch.empty(0, dtype=torch.long), 512, 1
        )
        self.assertEqual(values.shape, (0, 3))
        self.assertEqual(ids.shape, (0, 3))
        with self.assertRaises(ValueError):
            full_vocab_probe(torch.ones(1, 2), torch.nn.Linear(2, 1), torch.tensor([0]), 512, 1)

    def test_trainer_and_offline_signals_share_independent_deep_knee_union(self) -> None:
        # Each expert has a different plausible head, with its local knee at
        # rank 301. Gold is highest for every expert but must never enter DPP.
        head = torch.nn.Linear(3, 1600, bias=False)
        with torch.no_grad():
            head.weight.fill_(-10)
            for expert in range(3):
                start = expert * 512
                head.weight[start:start + 300, expert] = 10
                head.weight[start + 300:start + 512, expert] = 0
            head.weight[1599] = 99
        hidden = [torch.eye(3)[expert:expert + 1] for expert in range(3)]
        batch = {
            "input_ids": torch.tensor([[1, 1599]]),
            "attention_mask": torch.ones(1, 2, dtype=torch.long),
            "labels": torch.tensor([[-100, 1599]]),
            "region_ids": torch.tensor([[0, 2]]),
            "step_ids": torch.tensor([[-1, 0]]),
        }
        config = {
            "kneedle": {},  # Exercise search_k=512 and k_min=8 defaults.
            "runtime": {"lm_head_chunk_tokens": 32768, "probe_hidden_device": "cpu"},
            "dpp": {"jitter": 1e-4, "max_jitter": 1e-2},
        }
        with patch("cot_mtkd.stage1.trainer.decoder_and_lm_head", return_value=(None, head)):
            probe = probe_stage1_dpp(
                torch.nn.Identity(), ["a", "b", "c"], batch, 0, 0, 42, config,
                torch.device("cpu"), reasoning_hidden_by_expert=hidden,
            )
        self.assertEqual(probe.mean_raw_k, 301)
        self.assertEqual(probe.mean_selected_k, 301)
        self.assertEqual(probe.selection_count, 3)
        self.assertEqual(probe.raw_k_histogram[301].item(), 3)
        self.assertEqual(probe.selected_k_histogram[301].item(), 3)
        self.assertEqual(int(probe.support_mask.sum()), 903)
        self.assertNotIn(1599, probe.support_ids.tolist()[0])
        self.assertTrue(torch.isfinite(probe.dpp_logit_gradients).all())
        with patch(
            "cot_mtkd.signals.predictive.gather_support_logits", wraps=gather_support_logits
        ) as gathers:
            uniqueness = _dpp_uniqueness(
                hidden, head, torch.tensor([1599]), torch.tensor([0]), 32768, {}, 1e-4
            )
        self.assertEqual(gathers.call_count, 3)
        for call in gathers.call_args_list:
            self.assertTrue(torch.equal(call.args[2], probe.support_ids))
        self.assertEqual(uniqueness.shape, (1, 3))
        self.assertTrue(torch.isfinite(uniqueness).all())


if __name__ == "__main__":
    unittest.main()
