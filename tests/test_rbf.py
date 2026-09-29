from __future__ import annotations

import unittest
from collections import OrderedDict

import torch

from cot_mtkd.stage1.rbf import (
    effective_update_distances,
    low_rank_squared_distance,
    repulsion_updates,
)


def group(a: torch.Tensor, b: torch.Tensor):
    return OrderedDict(
        [
            ("layer.lora_A.{adapter}.weight", torch.nn.Parameter(a.clone())),
            ("layer.lora_B.{adapter}.weight", torch.nn.Parameter(b.clone())),
        ]
    )


class RBFTest(unittest.TestCase):
    def test_low_rank_formula_matches_materialized_update(self) -> None:
        torch.manual_seed(7)
        a_i, b_i = torch.randn(2, 5), torch.randn(4, 2)
        a_j, b_j = torch.randn(2, 5), torch.randn(4, 2)
        expected = ((b_i @ a_i) - (b_j @ a_j)).pow(2).sum()
        actual = low_rank_squared_distance(a_i, b_i, a_j, b_j, scaling=1.0)
        self.assertTrue(torch.allclose(expected, actual, atol=1e-5, rtol=1e-5))

    def test_low_rank_gradient_matches_materialized_update(self) -> None:
        torch.manual_seed(19)
        factors = [
            torch.randn(3, 7, requires_grad=True),
            torch.randn(5, 3, requires_grad=True),
            torch.randn(3, 7, requires_grad=True),
            torch.randn(5, 3, requires_grad=True),
        ]
        a_i, b_i, a_j, b_j = factors
        scaling = 1.25
        dense = ((b_i @ a_i) - (b_j @ a_j)).square().sum() * scaling**2
        expected = torch.autograd.grad(dense, factors)
        low_rank = low_rank_squared_distance(*factors, scaling=scaling)
        actual = torch.autograd.grad(low_rank, factors)
        self.assertTrue(torch.allclose(dense.detach(), low_rank.detach(), atol=1e-4))
        for left, right in zip(expected, actual, strict=True):
            self.assertTrue(torch.allclose(left, right, atol=1e-4, rtol=1e-5))

    def test_lora_gauge_invariance(self) -> None:
        torch.manual_seed(2)
        a, b = torch.randn(2, 3), torch.randn(4, 2)
        first = group(a, b)
        second = group(3.0 * a, b / 3.0)
        distance = effective_update_distances([first, second], scaling=1.0)
        self.assertLess(float(distance[0, 1].detach()), 1e-6)

    def test_repulsion_update_increases_distance(self) -> None:
        first = group(torch.tensor([[1.0]]), torch.tensor([[0.5]]))
        second = group(torch.tensor([[1.0]]), torch.tensor([[0.8]]))
        groups = [first, second]
        before = float(effective_update_distances(groups, 1.0)[0, 1].detach())
        updates, _, _ = repulsion_updates(groups, scaling=1.0, bandwidth=0.1)
        with torch.no_grad():
            for current, direction in zip(groups, updates, strict=True):
                for parameter, delta in zip(current.values(), direction, strict=True):
                    parameter.add_(delta, alpha=1.0e-3)
        after = float(effective_update_distances(groups, 1.0)[0, 1].detach())
        self.assertGreater(after, before)

    def test_reusing_distances_preserves_repulsion(self) -> None:
        groups = [
            group(torch.tensor([[1.0]]), torch.tensor([[0.5]])),
            group(torch.tensor([[1.0]]), torch.tensor([[0.8]])),
            group(torch.tensor([[1.0]]), torch.tensor([[1.1]])),
        ]
        recomputed, kernel, distance = repulsion_updates(groups, 1.0, 0.1)
        prepared = effective_update_distances(groups, 1.0)
        reused, reused_kernel, reused_distance = repulsion_updates(
            groups, 1.0, 0.1, distances=prepared
        )
        self.assertTrue(torch.equal(kernel, reused_kernel))
        self.assertTrue(torch.equal(distance, reused_distance))
        for old_group, new_group in zip(recomputed, reused, strict=True):
            for old, new in zip(old_group, new_group, strict=True):
                self.assertTrue(torch.equal(old, new))


if __name__ == "__main__":
    unittest.main()
