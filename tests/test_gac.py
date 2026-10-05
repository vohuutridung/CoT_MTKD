from __future__ import annotations

import unittest
from pathlib import Path

import torch
import yaml

from cot_mtkd.stage1.gac_gradient import STAGE1_METHOD, stable_gac_gradients
from cot_mtkd.stage1.trainer import _stage1_update_mode, _validate_stage1_config


def basis_tasks():
    return [[row.clone()] for row in torch.eye(3)]


def mix(kernel: torch.Tensor, beta: float = 0.5):
    return stable_gac_gradients(
        basis_tasks(), [[torch.zeros(3)] for _ in range(3)], kernel, beta=beta, rbf_weight=0.0
    )


class GACTest(unittest.TestCase):
    def test_three_exact_neighbor_cases(self) -> None:
        for kernels, coefficients in (
            ((0.0, 0.0), (1.0, 0.0, 0.0)),
            ((1.0, 0.0), (0.75, 0.25, 0.0)),
            ((1.0, 1.0), (0.5, 0.25, 0.25)),
        ):
            with self.subTest(kernels=kernels):
                kernel = torch.eye(3)
                kernel[1, 0] = kernel[0, 1] = kernels[0]
                kernel[2, 0] = kernel[0, 2] = kernels[1]
                final, diagnostics = mix(kernel)
                torch.testing.assert_close(final[0][0], torch.tensor(coefficients), atol=0, rtol=0)
                self.assertEqual(diagnostics.self_coefficients[0], coefficients[0])
                self.assertEqual(diagnostics.cross_coefficients[0], sum(coefficients[1:]))

    def test_self_is_excluded_from_cross_sum_and_diagonal_has_no_effect(self) -> None:
        kernel = torch.tensor([[1.0, 0.2, 0.4], [0.6, 1.0, 0.8], [0.1, 0.3, 1.0]])
        first, diagnostics = mix(kernel)
        kernel.fill_diagonal_(0.0)
        second, changed = mix(kernel)
        for left, right in zip(first, second, strict=True):
            torch.testing.assert_close(left[0], right[0], atol=0, rtol=0)
        self.assertEqual(diagnostics.cross_coefficients, changed.cross_coefficients)
        # Source j is row j, receiving expert i is column i; diagonal contributes zero.
        expected = torch.tensor([1 - 0.25 * (0.6 + 0.1), 0.25 * 0.6, 0.25 * 0.1])
        torch.testing.assert_close(first[0][0], expected)

    def test_distant_neighbors_are_not_renormalized(self) -> None:
        kernel = torch.eye(3)
        kernel[1, 0] = kernel[2, 0] = 1e-8
        final, diagnostics = mix(kernel)
        self.assertAlmostEqual(diagnostics.cross_coefficients[0], 5e-9, places=15)
        self.assertGreater(diagnostics.self_coefficients[0], 0.99999999)
        self.assertLess(final[0][0][1].item(), 1e-8)

    def test_beta_zero_keeps_own_task_and_general_beta_preserves_bound(self) -> None:
        torch.manual_seed(5)
        for beta in (0.0, 0.1, 0.5, 1.0):
            for kernel in (torch.rand(3, 3), torch.ones(3, 3)):
                with self.subTest(beta=beta):
                    final, diagnostics = mix(kernel, beta)
                    for self_coefficient, cross in zip(
                        diagnostics.self_coefficients, diagnostics.cross_coefficients, strict=True
                    ):
                        self.assertGreaterEqual(self_coefficient, 1 - beta - 1e-12)
                        self.assertAlmostEqual(self_coefficient + cross, 1.0)
                        if beta == 0.5:
                            self.assertGreaterEqual(self_coefficient, 0.5)
                    if beta == 0:
                        for actual, expected in zip(final, basis_tasks(), strict=True):
                            torch.testing.assert_close(actual[0], expected[0], atol=0, rtol=0)

    def test_repulsion_cap_and_descent_sign(self) -> None:
        task = [[torch.tensor([3.0, 4.0])], [torch.tensor([0.0, 2.0])]]
        repulsion = [[torch.tensor([60.0, 80.0])], [torch.tensor([0.0, -1.0])]]
        final, diagnostics = stable_gac_gradients(task, repulsion, torch.eye(2), rbf_weight=0.5)
        torch.testing.assert_close(final[0][0], torch.tensor([1.5, 2.0]))
        torch.testing.assert_close(final[1][0], torch.tensor([0.0, 2.5]))
        self.assertAlmostEqual(diagnostics.repulsion_cap_factors[0], 0.05)
        for capped, mixed, weighted in zip(
            diagnostics.capped_repulsion_norms, diagnostics.mixed_task_norms,
            diagnostics.weighted_repulsion_norms, strict=True,
        ):
            self.assertLessEqual(capped, mixed + 1e-12)
            self.assertAlmostEqual(weighted, 0.5 * capped)

    def test_cap_uses_mixed_task_norm_across_all_parameters(self) -> None:
        tasks = [[torch.tensor([float(i + 1)]), torch.tensor([2.0, -1.0])] for i in range(3)]
        repulsion = [[torch.tensor([100.0]), torch.tensor([200.0, -300.0])] for _ in tasks]
        mixed, _ = stable_gac_gradients(tasks, repulsion, torch.ones(3, 3), rbf_weight=0.0)
        final, diagnostics = stable_gac_gradients(tasks, repulsion, torch.ones(3, 3), rbf_weight=0.5)
        for before, after, norm in zip(mixed, final, diagnostics.mixed_task_norms, strict=True):
            capped = torch.cat([(a - b).flatten() / 0.5 for a, b in zip(before, after, strict=True)])
            self.assertLessEqual(capped.norm().item(), norm + 1e-6)

    def test_hard_switch_at_exact_ten_percent(self) -> None:
        self.assertEqual(_stage1_update_mode(0, 11), "sft_only")
        self.assertEqual(_stage1_update_mode(1, 11), "full_interaction")  # 1/(11-1) = .10
        self.assertEqual(_stage1_update_mode(2, 11), "full_interaction")
        self.assertEqual(_stage1_update_mode(1, 21), "sft_only")
        self.assertEqual(_stage1_update_mode(2, 21), "full_interaction")
        self.assertEqual(_stage1_update_mode(0, 1), "sft_only")
        self.assertEqual(_stage1_update_mode(0, 1, 0.0), "full_interaction")

    def test_default_config_and_no_legacy_schedule(self) -> None:
        path = Path(__file__).resolve().parents[1] / "configs/stage1/qwen25_7b_m3.yaml"
        config = yaml.safe_load(path.read_text())
        _validate_stage1_config(config)
        expected = {
            "objective": STAGE1_METHOD, "num_experts": 3, "forward_mode": "one_pass",
            "dpp_weight": 0.1, "gac_beta": 0.5, "gac_bandwidth_scale": 0.5,
            "rbf_weight": 0.5, "rbf_bandwidth_scale": 1.0, "sft_warmup_fraction": 0.10,
            "epochs": 3, "global_batch_size": 16, "micro_batch_size": 1,
            "learning_rate": 5e-5, "max_grad_norm": 1.0,
        }
        self.assertEqual({key: config["stage1"][key] for key in expected}, expected)
        self.assertEqual(config["kneedle"], {"search_k": 512, "k_min": 8})
        self.assertEqual([config["lora"][key] for key in ("rank", "alpha", "dropout")], [16, 16, 0.05])
        self.assertFalse(any(key.startswith(("interaction_", "gamma_")) for key in config["stage1"]))
        for legacy in ("interaction_off_until", "interaction_ramp_until", "gamma_schedule"):
            with self.subTest(legacy=legacy), self.assertRaisesRegex(ValueError, "obsolete"):
                _validate_stage1_config({**config, "stage1": {**config["stage1"], legacy: 0.1}})
        for beta in (-0.1, 1.1, float("nan")):
            with self.subTest(beta=beta), self.assertRaisesRegex(ValueError, "gac_beta"):
                _validate_stage1_config({**config, "stage1": {**config["stage1"], "gac_beta": beta}})

    def test_invalid_inputs_are_rejected(self) -> None:
        for beta in (-1, 1.1, float("nan")):
            with self.subTest(beta=beta), self.assertRaisesRegex(ValueError, "gac_beta"):
                mix(torch.eye(3), beta)
        for value in (-0.1, 1.1, float("nan")):
            kernel = torch.eye(3)
            kernel[1, 0] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "Off-diagonal"):
                mix(kernel)
        with self.assertRaisesRegex(ValueError, "shapes"):
            stable_gac_gradients(basis_tasks(), [[torch.zeros(2)]] * 3, torch.eye(3))


if __name__ == "__main__":
    unittest.main()
