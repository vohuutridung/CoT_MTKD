from __future__ import annotations

import unittest
from collections import OrderedDict

import torch

from cot_mtkd.stage1.rbf import (
    effective_update_distances,
    low_rank_squared_distance,
    rbf_repulsion_gradients,
    rbf_repulsion_loss,
)


def group(a: torch.Tensor, b: torch.Tensor):
    return OrderedDict(
        [
            ("layer.lora_A.{adapter}.weight", torch.nn.Parameter(a.clone())),
            ("layer.lora_B.{adapter}.weight", torch.nn.Parameter(b.clone())),
        ]
    )


class RBFTest(unittest.TestCase):
    def test_potential_is_unordered_pair_mean_and_bandwidth_is_detached(self) -> None:
        distances = torch.tensor(
            [[0.0, 0.1, 0.2], [0.1, 0.0, 0.3], [0.2, 0.3, 0.0]], requires_grad=True
        )
        bandwidth = torch.tensor(0.5, requires_grad=True)
        loss = rbf_repulsion_loss(distances, bandwidth)
        expected = torch.exp(-torch.tensor([0.1, 0.2, 0.3]) / 0.5).mean()
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertIsNone(bandwidth.grad)
        self.assertTrue(torch.all(torch.diagonal(distances.grad) == 0))
        self.assertAlmostEqual(
            float(distances.grad[0, 1]), -float(torch.exp(torch.tensor(-0.2))) / 1.5, places=6
        )

    def test_scalar_rbf_gradient_matches_dense_effective_updates_across_modules(self) -> None:
        torch.manual_seed(21)
        groups = []
        for _ in range(3):
            current = group(torch.randn(2, 5), torch.randn(4, 2))
            extra = group(torch.randn(2, 3), torch.randn(6, 2))
            current.update(
                (key.replace("layer.", "other."), parameter) for key, parameter in extra.items()
            )
            groups.append(current)
        scaling, bandwidth = 1.25, 2.0
        distances = []
        for left in range(3):
            for right in range(left + 1, 3):
                modules = []
                for key in groups[left]:
                    if "lora_A" in key:
                        bkey = key.replace("lora_A", "lora_B")
                        wi = scaling * groups[left][bkey] @ groups[left][key]
                        wj = scaling * groups[right][bkey] @ groups[right][key]
                        modules.append((wi - wj).square().mean())
                distances.append(torch.stack(modules).mean())
        dense_loss = torch.exp(-torch.stack(distances) / bandwidth).mean()
        parameters = [value for current in groups for value in current.values()]
        expected = torch.autograd.grad(dense_loss, parameters)
        actual = rbf_repulsion_gradients(groups, scaling, bandwidth)
        self.assertAlmostEqual(actual.loss, float(dense_loss.detach()), places=6)
        for dense, low_rank in zip(
            expected, [value for current in actual.gradients for value in current], strict=True
        ):
            torch.testing.assert_close(low_rank, dense, atol=2e-7, rtol=2e-5)

    def test_zero_b_initialization_has_finite_unit_loss_and_zero_gradient(self) -> None:
        groups = [group(torch.randn(2, 5), torch.zeros(4, 2)) for _ in range(3)]
        actual = rbf_repulsion_gradients(groups, 1.0, 1e-12)
        self.assertEqual(actual.loss, 1.0)
        for current in actual.gradients:
            for value in current:
                self.assertTrue(torch.equal(value, torch.zeros_like(value)))

    def test_invalid_bandwidth_and_single_expert_are_rejected(self) -> None:
        for bandwidth in [0, -1, float("nan"), float("inf")]:
            with self.subTest(bandwidth=bandwidth), self.assertRaisesRegex(ValueError, "bandwidth"):
                rbf_repulsion_loss(torch.zeros(2, 2), bandwidth)
        with self.assertRaisesRegex(ValueError, "at least two experts"):
            rbf_repulsion_loss(torch.zeros(1, 1), 0.1)

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
        result = rbf_repulsion_gradients(groups, scaling=1.0, bandwidth=0.1)
        with torch.no_grad():
            for current, direction in zip(groups, result.gradients, strict=True):
                for parameter, delta in zip(current.values(), direction, strict=True):
                    parameter.add_(delta, alpha=-1.0e-3)
        after = float(effective_update_distances(groups, 1.0)[0, 1].detach())
        self.assertGreater(after, before)

    def test_reusing_distances_preserves_repulsion(self) -> None:
        groups = [
            group(torch.tensor([[1.0]]), torch.tensor([[0.5]])),
            group(torch.tensor([[1.0]]), torch.tensor([[0.8]])),
            group(torch.tensor([[1.0]]), torch.tensor([[1.1]])),
        ]
        recomputed = rbf_repulsion_gradients(groups, 1.0, 0.1)
        prepared = effective_update_distances(groups, 1.0)
        reused = rbf_repulsion_gradients(groups, 1.0, 0.1, distances=prepared)
        self.assertTrue(torch.equal(recomputed.kernel, reused.kernel))
        self.assertTrue(torch.equal(recomputed.distances, reused.distances))
        for old_group, new_group in zip(recomputed.gradients, reused.gradients, strict=True):
            for old, new in zip(old_group, new_group, strict=True):
                self.assertTrue(torch.equal(old, new))


if __name__ == "__main__":
    unittest.main()
