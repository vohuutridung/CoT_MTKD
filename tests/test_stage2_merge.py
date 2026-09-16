from __future__ import annotations

import unittest

import torch

from cot_mtkd.stage2.merge import (
    MERGE_METHODS,
    lora_scaling,
    merge_adapter_states,
    normalize_merge_method,
)


def _state(a: torch.Tensor, b: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        "layer.lora_A.{adapter}.weight": a.clone(),
        "layer.lora_B.{adapter}.weight": b.clone(),
    }


def _delta(state: dict[str, torch.Tensor], rank: int = 2, alpha: float = 2.0) -> torch.Tensor:
    scale = lora_scaling(rank, alpha)
    return scale * (
        state["layer.lora_B.{adapter}.weight"].float()
        @ state["layer.lora_A.{adapter}.weight"].float()
    )


class Stage2MergeTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.rank = 2
        self.alpha = 2.0
        self.experts = [
            _state(torch.randn(2, 6), torch.randn(8, 2)) for _ in range(3)
        ]

    def test_normalize_aliases(self) -> None:
        self.assertEqual(normalize_merge_method("TA"), "ta")
        self.assertEqual(normalize_merge_method("DARE-TIES"), "dare_ties")
        self.assertEqual(normalize_merge_method("Iso-C"), "iso_c")
        self.assertEqual(set(MERGE_METHODS), {"ta", "ties", "dare_ties", "tsv", "iso_c"})

    def test_task_arithmetic_is_rank_r_projection_of_mean_delta(self) -> None:
        merged = merge_adapter_states(
            self.experts, method="ta", rank=self.rank, alpha=self.alpha
        )
        expected = torch.stack(
            [_delta(state, self.rank, self.alpha) for state in self.experts]
        ).mean(dim=0)
        left, values, right = torch.linalg.svd(expected, full_matrices=False)
        projected = (left[:, : self.rank] * values[: self.rank]) @ right[: self.rank]
        self.assertTrue(
            torch.allclose(_delta(merged, self.rank, self.alpha), projected, atol=1e-5)
        )

    def test_identical_experts_are_fixed_points_of_ta(self) -> None:
        copies = [self.experts[0], {key: value.clone() for key, value in self.experts[0].items()}]
        merged = merge_adapter_states(
            copies, method="ta", rank=self.rank, alpha=self.alpha
        )
        self.assertTrue(
            torch.allclose(
                _delta(merged, self.rank, self.alpha),
                _delta(self.experts[0], self.rank, self.alpha),
                atol=1e-5,
            )
        )

    def test_ties_density_one_matches_ta_when_signs_agree(self) -> None:
        positive = [
            _state(torch.ones(2, 5), torch.ones(4, 2)),
            _state(2 * torch.ones(2, 5), torch.ones(4, 2)),
        ]
        ta = merge_adapter_states(positive, method="ta", rank=2, alpha=2.0)
        ties = merge_adapter_states(
            positive, method="ties", rank=2, alpha=2.0, ties_density=1.0
        )
        self.assertTrue(torch.allclose(_delta(ta), _delta(ties), atol=1e-5))

    def test_dare_without_drop_matches_ties(self) -> None:
        ties = merge_adapter_states(
            self.experts, method="ties", rank=self.rank, alpha=self.alpha, ties_density=1.0
        )
        dare = merge_adapter_states(
            self.experts,
            method="dare_ties",
            rank=self.rank,
            alpha=self.alpha,
            ties_density=1.0,
            dare_drop_prob=0.0,
            seed=0,
        )
        self.assertTrue(torch.allclose(_delta(ties), _delta(dare), atol=1e-5))

    def test_iso_c_flattens_singular_spectrum(self) -> None:
        merged = merge_adapter_states(
            self.experts, method="iso_c", rank=self.rank, alpha=self.alpha
        )
        delta = _delta(merged, self.rank, self.alpha)
        values = torch.linalg.svdvals(delta.float())[: self.rank]
        nonzero = values[values > 1.0e-5]
        self.assertGreaterEqual(nonzero.numel(), 1)
        self.assertTrue(torch.allclose(nonzero, nonzero.mean().expand_as(nonzero), atol=1e-4))

    def test_tsv_returns_finite_same_shape(self) -> None:
        merged = merge_adapter_states(
            self.experts, method="tsv", rank=self.rank, alpha=self.alpha
        )
        self.assertEqual(
            merged["layer.lora_A.{adapter}.weight"].shape,
            self.experts[0]["layer.lora_A.{adapter}.weight"].shape,
        )
        self.assertTrue(torch.isfinite(_delta(merged, self.rank, self.alpha)).all())

    def test_unknown_method_raises(self) -> None:
        with self.assertRaises(ValueError):
            merge_adapter_states(self.experts, method="soup", rank=2, alpha=2.0)


if __name__ == "__main__":
    unittest.main()
