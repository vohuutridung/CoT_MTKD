from __future__ import annotations

import math
import unittest
from collections import OrderedDict

import torch

from cot_mtkd.stage1.repulsion import (
    b_side_active,
    gram_stats,
    kernel_from_d2,
    pairwise_d2,
    projection_repulsion_updates,
    repulsion_force,
)


def _group(a: torch.Tensor, b: torch.Tensor) -> OrderedDict[str, torch.nn.Parameter]:
    return OrderedDict(
        (
            ("layer.lora_A.weight", torch.nn.Parameter(a.clone())),
            ("layer.lora_B.weight", torch.nn.Parameter(b.clone())),
        )
    )


def _ridge_overlap(left: torch.Tensor, right: torch.Tensor, eps_rel: float, eps_abs: float) -> torch.Tensor:
    rank = left.shape[0]

    def ridge(gram: torch.Tensor) -> torch.Tensor:
        return gram + (eps_rel * torch.trace(gram) / rank + eps_abs) * torch.eye(
            rank, dtype=gram.dtype, device=gram.device
        )

    cross = left @ right.T
    return torch.trace(
        torch.linalg.solve(ridge(left @ left.T), cross)
        @ torch.linalg.solve(ridge(right @ right.T), cross.T)
    )


class ProjectionRepulsionTest(unittest.TestCase):
    def test_force_matches_autograd_of_distance(self) -> None:
        torch.manual_seed(0)
        rank, width = 3, 5
        eps_rel, eps_abs = 1.0e-8, 1.0e-12
        left = torch.randn(rank, width, dtype=torch.float64, requires_grad=True)
        right = torch.randn(rank, width, dtype=torch.float64)
        factors = torch.stack([left.detach(), right], dim=0).unsqueeze(0)
        stats = gram_stats(factors, eps_rel, eps_abs)
        kernel = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float64)
        force = repulsion_force(factors, stats, kernel, n_modules=1)
        potential = 0.5 * (rank - _ridge_overlap(left, right, eps_rel, eps_abs))
        expected = torch.autograd.grad(potential, left)[0]
        self.assertLess(float((force[0, 0] - expected).abs().max()), 1.0e-8)

    def test_force_is_orthogonal_to_row_space(self) -> None:
        torch.manual_seed(1)
        factors = torch.randn(1, 2, 3, 7, dtype=torch.float64)
        stats = gram_stats(factors, eps_rel=1.0e-12, eps_abs=0.0)
        kernel = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float64)
        force = repulsion_force(factors, stats, kernel, n_modules=1)
        leakage = force[0, 0] @ factors[0, 0].T
        self.assertLess(float(leakage.norm()), 1.0e-6)

    def test_force_vanishes_on_identical_subspaces(self) -> None:
        torch.manual_seed(2)
        left = torch.randn(4, 8, dtype=torch.float64)
        transform = torch.randn(4, 4, dtype=torch.float64)
        transform = transform + 2.0 * torch.eye(4, dtype=torch.float64)
        right = transform @ left
        factors = torch.stack([left, right], dim=0).unsqueeze(0)
        stats = gram_stats(factors, eps_rel=1.0e-12, eps_abs=0.0)
        kernel = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float64)
        force = repulsion_force(factors, stats, kernel, n_modules=1)
        self.assertLess(float(force.norm()), 1.0e-6 * float(left.norm()))

    def test_gl_invariance_of_overlap(self) -> None:
        torch.manual_seed(3)
        factors = torch.randn(2, 3, 4, 6, dtype=torch.float64)
        changed = factors.clone()
        for expert in range(3):
            transform = torch.randn(4, 4, dtype=torch.float64)
            transform = transform + 3.0 * torch.eye(4, dtype=torch.float64)
            changed[:, expert] = transform @ factors[:, expert]
        original = gram_stats(factors, eps_rel=0.0, eps_abs=1.0e-14)[3]
        updated = gram_stats(changed, eps_rel=0.0, eps_abs=1.0e-14)[3]
        self.assertTrue(torch.allclose(original, updated, atol=1.0e-6, rtol=1.0e-6))
        distance = pairwise_d2(original, None, rank=4, n_modules=2)
        kernel, _bandwidth = kernel_from_d2(distance)
        distance_b = pairwise_d2(updated, None, rank=4, n_modules=2)
        kernel_b, _bandwidth_b = kernel_from_d2(distance_b)
        self.assertTrue(torch.allclose(distance, distance_b, atol=1.0e-6, rtol=1.0e-6))
        self.assertTrue(torch.allclose(kernel, kernel_b, atol=1.0e-6, rtol=1.0e-6))

    def test_overlap_bounds_for_identical_and_orthogonal_subspaces(self) -> None:
        eye = torch.eye(2, dtype=torch.float64)
        identical = torch.zeros(1, 2, 2, 4, dtype=torch.float64)
        identical[0, 0, :, :2] = eye
        identical[0, 1, :, :2] = eye
        orthogonal = identical.clone()
        orthogonal[0, 1].zero_()
        orthogonal[0, 1, :, 2:] = eye
        same = gram_stats(identical, eps_rel=0.0, eps_abs=1.0e-14)[3]
        apart = gram_stats(orthogonal, eps_rel=0.0, eps_abs=1.0e-14)[3]
        self.assertAlmostEqual(float(same[0, 0, 1]), 2.0, places=5)
        self.assertAlmostEqual(float(apart[0, 0, 1]), 0.0, places=5)
        self.assertTrue(bool((same >= -1.0e-6).all() and (same <= 2.0 + 1.0e-5).all()))

    def test_batched_gram_matches_per_module_loop(self) -> None:
        torch.manual_seed(4)
        factors = torch.randn(5, 4, 3, 8, dtype=torch.float64)
        batched = gram_stats(factors, eps_rel=1.0e-6, eps_abs=1.0e-10)
        for layer in range(factors.shape[0]):
            single = gram_stats(factors[layer : layer + 1], eps_rel=1.0e-6, eps_abs=1.0e-10)
            for left, right in zip(batched, single, strict=True):
                self.assertTrue(torch.allclose(left[layer], right[0], atol=1.0e-8, rtol=1.0e-6))

    def test_step_along_force_increases_distance(self) -> None:
        torch.manual_seed(5)
        factors = torch.randn(1, 2, 3, 6, dtype=torch.float64)
        stats = gram_stats(factors, eps_rel=1.0e-8, eps_abs=1.0e-12)
        kernel = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float64)
        force = repulsion_force(factors, stats, kernel, n_modules=1)
        before = pairwise_d2(stats[3], None, rank=3, n_modules=1)[0, 1]
        after_stats = gram_stats(factors + 1.0e-2 * force, eps_rel=1.0e-8, eps_abs=1.0e-12)
        after = pairwise_d2(after_stats[3], None, rank=3, n_modules=1)[0, 1]
        self.assertGreater(float(after), float(before))

    def test_projection_distance_tracks_small_geodesic_angles(self) -> None:
        theta = 0.15
        left = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
        right = torch.tensor([[math.cos(theta), math.sin(theta), 0.0]], dtype=torch.float64)
        factors = torch.stack([left, right], dim=0).unsqueeze(0)
        overlap = gram_stats(factors, eps_rel=0.0, eps_abs=1.0e-14)[3][0, 0, 1]
        projection = 1.0 - float(overlap)
        cosine = torch.linalg.svdvals(left @ right.T).clamp(0, 1)
        geodesic = float(torch.arccos(cosine).square().sum())
        self.assertLess(abs(projection - geodesic) / geodesic, 0.05)
        self.assertLessEqual(projection, geodesic + 1.0e-8)
        self.assertLessEqual(geodesic, (math.pi**2 / 4.0) * projection + 1.0e-8)

        wide = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float64)
        wide_factors = torch.stack([left, wide], dim=0).unsqueeze(0)
        wide_overlap = gram_stats(wide_factors, eps_rel=0.0, eps_abs=1.0e-14)[3][0, 0, 1]
        wide_projection = 1.0 - float(wide_overlap)
        wide_geodesic = float(torch.arccos(torch.linalg.svdvals(left @ wide.T).clamp(0, 1)).square().sum())
        self.assertLessEqual(wide_projection, wide_geodesic + 1.0e-8)
        self.assertLessEqual(wide_geodesic, (math.pi**2 / 4.0) * wide_projection + 1.0e-8)

    def test_zero_b_is_skipped_and_turns_on_at_the_configured_step(self) -> None:
        groups = [
            _group(torch.tensor([[1.0, 0.0, 0.2]]), torch.zeros(3, 1)),
            _group(torch.tensor([[0.2, 1.0, 0.0]]), torch.zeros(3, 1)),
        ]
        stats = projection_repulsion_updates(
            groups, eps_rel=1.0e-8, eps_abs=1.0e-12, b_active=False
        )
        self.assertFalse(stats.b_active)
        self.assertTrue(all(torch.isfinite(item).all() for expert in stats.updates for item in expert))
        for expert, group in zip(stats.updates, groups, strict=True):
            b_index = list(group).index("layer.lora_B.weight")
            self.assertTrue(torch.count_nonzero(expert[b_index]) == 0)
            self.assertGreater(float(expert[0].norm()), 0.0)
        self.assertFalse(b_side_active(199, min_b_norm=1.0, start_step=200, min_norm=0.0))
        self.assertFalse(b_side_active(200, min_b_norm=0.0, start_step=200, min_norm=0.0))
        self.assertTrue(b_side_active(200, min_b_norm=1.0e-3, start_step=200, min_norm=0.0))

    def test_short_training_separates_experts_and_reduces_loss(self) -> None:
        torch.manual_seed(6)
        rank, width, experts, steps = 2, 5, 3, 20
        base = [torch.randn(rank, width) for _ in range(experts)]

        def run(lambda_rep: float) -> tuple[float, float]:
            groups = [
                _group(base[index], torch.randn(4, rank) * 0.05) for index in range(experts)
            ]
            parameters = [parameter for group in groups for parameter in group.values()]
            optimizer = torch.optim.AdamW(parameters, lr=0.05, weight_decay=0.0)
            targets = [item + 0.1 * torch.randn_like(item) for item in base]
            for _step in range(steps):
                optimizer.zero_grad()
                loss = sum(
                    (group["layer.lora_A.weight"] - target).square().sum()
                    for group, target in zip(groups, targets, strict=True)
                )
                loss.backward()
                optimizer.step()
                if lambda_rep:
                    stats = projection_repulsion_updates(
                        groups, eps_rel=1.0e-6, eps_abs=1.0e-10, b_active=True
                    )
                    with torch.no_grad():
                        for group, forces in zip(groups, stats.updates, strict=True):
                            for parameter, force in zip(group.values(), forces, strict=True):
                                parameter.add_(force, alpha=0.05 * lambda_rep)
            final = projection_repulsion_updates(
                groups, eps_rel=1.0e-6, eps_abs=1.0e-10, b_active=True
            )
            loss_value = float(
                sum(
                    (group["layer.lora_A.weight"].detach() - target).square().sum()
                    for group, target in zip(groups, targets, strict=True)
                )
            )
            off = final.distances[
                torch.triu(torch.ones(experts, experts, dtype=torch.bool), diagonal=1)
            ]
            return float(off.mean()), loss_value

        plain_distance, plain_loss = run(0.0)
        pushed_distance, pushed_loss = run(1.0)
        self.assertGreater(pushed_distance, plain_distance)
        self.assertLess(pushed_loss, 1.0)
        self.assertTrue(math.isfinite(pushed_distance) and math.isfinite(plain_loss))


if __name__ == "__main__":
    unittest.main()
