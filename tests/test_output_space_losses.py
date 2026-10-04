from __future__ import annotations

import math
import unittest
from decimal import Decimal, localcontext

import torch
import torch.nn.functional as F

from cot_mtkd.stage2.output_space_losses import (
    reduced_log_distribution,
    reduced_kd_sft_tokens,
    local_js_log_distribution,
    normalized_js_disagreement,
    power_mean_log_target,
)


class OutputSpaceLossTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.thread_count = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.thread_count)

    def test_barycenter_endpoints_and_teacher_permutation(self):
        generator = torch.Generator().manual_seed(107)
        logp = F.log_softmax(torch.randn(3, 5, 17, generator=generator).double(), -1)
        geometric = F.softmax(logp.mean(0), -1)
        arithmetic = logp.exp().mean(0)
        torch.testing.assert_close(power_mean_log_target(logp, 0).exp(), geometric)
        torch.testing.assert_close(power_mean_log_target(logp, 1).exp(), arithmetic)
        rho = 0.37
        target = logp.exp().pow(rho).mean(0).pow(1 / rho)
        target /= target.sum(-1, keepdim=True)
        actual = power_mean_log_target(logp, rho).exp()
        torch.testing.assert_close(actual, target)
        torch.testing.assert_close(actual, power_mean_log_target(logp[[2, 0, 1]], rho).exp())
        torch.testing.assert_close(actual.sum(-1), torch.ones(5, dtype=actual.dtype))

    def test_near_zero_matches_high_precision_and_does_not_truncate_rho(self):
        logp = F.log_softmax(
            torch.tensor(
                [
                    [[8.0, -1.0, -4.0]],
                    [[-3.0, 7.0, 0.0]],
                    [[1.0, -2.0, 6.0]],
                ],
                dtype=torch.float64,
            ),
            -1,
        )
        for rho in (1.0e-12, 1.0e-7):
            with localcontext() as context:
                context.prec = 80
                r = Decimal(str(rho))
                means = [
                    sum((Decimal(str(float(logp[m, 0, v]))) * r).exp() for m in range(3))
                    / Decimal(3)
                    for v in range(3)
                ]
                masses = [(value.ln() / r).exp() for value in means]
                total = sum(masses)
                expected = torch.tensor(
                    [[float(value / total) for value in masses]], dtype=torch.float64
                )
            torch.testing.assert_close(
                power_mean_log_target(logp, rho).exp(), expected, atol=2.0e-14, rtol=2.0e-13
            )
        difference = (
            (power_mean_log_target(logp, 1.0e-7) - power_mean_log_target(logp, 0)).abs().max()
        )
        self.assertGreater(float(difference), 1.0e-8)
        extreme = F.log_softmax(logp * 1000, -1)
        for rho in (0, 1.0e-14, 0.5, 1):
            target = power_mean_log_target(extreme, rho)
            self.assertTrue(bool(torch.isfinite(target).all()))
            torch.testing.assert_close(target.exp().sum(-1), torch.ones(1, dtype=target.dtype))

    def test_js_matches_definition_bounds_and_identical_teachers(self):
        generator = torch.Generator().manual_seed(227)
        logp = F.log_softmax(torch.randn(3, 7, 11, generator=generator).double() * 3, -1)
        probabilities = logp.exp()
        mixture = probabilities.mean(0)
        expected = (probabilities * (logp - mixture.log())).sum(-1).mean(0) / math.log(3)
        actual = normalized_js_disagreement(logp)
        torch.testing.assert_close(actual, expected, atol=2.0e-14, rtol=2.0e-13)
        self.assertTrue(bool(((actual >= 0) & (actual <= 1)).all()))
        repeated = logp[:1].repeat(3, 1, 1)
        self.assertLess(float(normalized_js_disagreement(repeated).max()), 1.0e-25)
        distinct = F.log_softmax(torch.eye(3, dtype=torch.float64)[:, None, :] * 1000, -1)
        torch.testing.assert_close(
            normalized_js_disagreement(distinct), torch.ones(1, dtype=torch.float64)
        )
        close = logp[:1].repeat(3, 1, 1)
        close[1, :, 0] += 1.0e-8
        close = F.log_softmax(close, -1)
        self.assertGreater(float(normalized_js_disagreement(close).max()), 0)

    def test_tail_matches_dense_probability_and_has_finite_gradients(self):
        for logits, ids in (
            (torch.tensor([[8.0, -8.0, -9.0, -10.0]], dtype=torch.float64), [[0]]),
            (torch.zeros(1, 4, dtype=torch.float64), [[1]]),
            (torch.tensor([[1000.0, -1000.0, -1200.0, -1400.0]], dtype=torch.float64), [[0]]),
            (torch.randn(2, 4, dtype=torch.float64), [[0, 1, 2, 3], [0, 1, 2, 3]]),
        ):
            logits.requires_grad_()
            ids = torch.tensor(ids)
            mask = torch.ones_like(ids, dtype=torch.bool)
            anomalies = {}
            logp = reduced_log_distribution(logits, ids, mask, 2.0, anomalies)
            dense = F.softmax(logits / 2, -1)
            selected = dense.gather(-1, ids)
            outside = dense.clone().scatter(1, ids, 0).sum(-1)
            torch.testing.assert_close(logp[:, :-1].exp(), selected, atol=1e-14, rtol=1e-12)
            torch.testing.assert_close(logp[:, -1].exp(), outside, atol=1e-14, rtol=1e-12)
            torch.testing.assert_close(
                logp.exp().sum(-1), torch.ones(len(logits), dtype=logp.dtype)
            )
            self.assertTrue(bool(torch.isfinite(logp).all()))
            grad = torch.autograd.grad(logp[:, -1].sum(), logits)[0]
            self.assertTrue(bool(torch.isfinite(grad).all()))
        self.assertEqual(anomalies["tail_probability_clamps"], 2)

    def test_small_support_has_large_tail_and_variable_padding_is_ignored(self):
        logits = torch.zeros(2, 10)
        ids = torch.tensor([[0, 0, 0], [0, 1, 2]])
        mask = torch.tensor([[True, False, False], [True, True, True]])
        p = reduced_log_distribution(logits, ids, mask, 2.0).exp()
        torch.testing.assert_close(p[:, -1], torch.tensor([0.9, 0.7]))
        self.assertEqual(float(p[0, 1:3].sum()), 0.0)
        torch.testing.assert_close(p.sum(-1), torch.ones(2))

    def test_separate_temperature_paths_and_local_js_restriction(self):
        logits = torch.tensor([[3.0, 0.0, -1.0, -2.0]], dtype=torch.float64)
        ids = torch.tensor([[0, 2]])
        mask = torch.ones_like(ids, dtype=torch.bool)
        js = local_js_log_distribution(logits, ids, mask, 1.0)
        dense = F.softmax(logits, -1).gather(-1, ids)
        torch.testing.assert_close(js.exp(), dense / dense.sum(-1, keepdim=True))
        kd = reduced_log_distribution(logits, ids, mask, 2.0)
        torch.testing.assert_close(kd[:, :-1].exp(), F.softmax(logits / 2.0, -1).gather(-1, ids))
        self.assertFalse(torch.allclose(js.exp(), F.softmax(logits[:, [0, 2]] / 2.0, -1)))

    def test_equal_target_zero_kl_and_temperature_squared_compensation(self):
        logits = torch.tensor([[3.0, 0.0, -1.0, -2.0]], requires_grad=True)
        ids = torch.tensor([[0, 2]])
        mask = torch.ones_like(ids, dtype=torch.bool)
        q = reduced_log_distribution(logits, ids, mask, 2.0).detach()
        kd, _, _, _ = reduced_kd_sft_tokens(logits, torch.tensor([0]), ids, mask, q, 2.0)
        self.assertAlmostEqual(float(kd.detach()), 0.0, delta=1e-6)
        q = torch.tensor([[0.2, 0.3, 0.5]]).log()
        kd, _, _, _ = reduced_kd_sft_tokens(logits, torch.tensor([0]), ids, mask, q, 2.0)
        expected = 4 * (q.exp() * (q - reduced_log_distribution(logits, ids, mask, 2.0))).sum(-1)
        torch.testing.assert_close(kd, expected)

    def test_reduced_power_mean_with_tiny_tail_and_rho_endpoints(self):
        logits = torch.tensor([[1000.0, -1000.0, -1200.0]], dtype=torch.float64)
        ids, mask = torch.tensor([[0, 1]]), torch.ones(1, 2, dtype=torch.bool)
        teacher = torch.stack(
            [reduced_log_distribution(logits + i, ids, mask, 2.0) for i in range(3)]
        )
        for rho in (0.0, 1e-14, 0.5, 1.0):
            q = power_mean_log_target(teacher, rho)
            self.assertTrue(bool(torch.isfinite(q).all()))
            torch.testing.assert_close(q.exp().sum(-1), torch.ones(1, dtype=q.dtype))

    def test_bfloat16_chunked_combined_cotangent_and_step_balanced_sft(self):
        from cot_mtkd.stage2.output_space_losses import cached_kd_sft_hidden_gradient

        torch.manual_seed(182)
        head = torch.nn.Linear(4, 13, bias=False).bfloat16()
        head.requires_grad_(False)
        hidden = torch.randn(5, 4, dtype=torch.bfloat16, requires_grad=True)
        gold = torch.tensor([0, 1, 2, 3, 4])
        ids = torch.tensor([[0, 1, 2, 3, 4]] * 5)
        mask = torch.ones_like(ids, dtype=torch.bool)
        target = torch.tensor([[0.1, 0.1, 0.1, 0.1, 0.1, 0.5]] * 5).log()
        weights = torch.tensor([0.25, 0.25, 1 / 6, 1 / 6, 1 / 6])
        kd, sft, _, _ = reduced_kd_sft_tokens(head(hidden), gold, ids, mask, target, 2.0)
        expected_loss = ((kd + 0.25 * sft) * weights).sum()
        expected_grad = torch.autograd.grad(expected_loss, hidden)[0]
        gradient, diagnostics = cached_kd_sft_hidden_gradient(
            hidden, head, gold, ids, mask, target, weights, 2.0, 0.25, 2
        )
        torch.testing.assert_close(gradient, expected_grad, atol=2e-3, rtol=2e-2)
        actual_sft = float((diagnostics["sft"] * weights).sum())
        expected_sft = float(((sft[:2].mean() + sft[2:].mean()) / 2).detach())
        self.assertAlmostEqual(actual_sft, expected_sft, delta=2e-6)
        self.assertTrue(bool(torch.isfinite(gradient).all()))
        self.assertFalse(gradient.requires_grad)
        self.assertIsNone(hidden.grad)

    def test_fp32_log_cache_preserves_extreme_power_mean_target(self):
        from safetensors.torch import save_file, load_file
        import tempfile
        from pathlib import Path

        values = F.log_softmax(
            torch.tensor(
                [[[0.0, -1000.0, -2000.0]], [[0.0, -1200.0, -2400.0]], [[0.0, -1400.0, -2800.0]]],
                dtype=torch.float64,
            ),
            -1,
        )
        for rho in (0.0, 1e-14, 0.25, 1.0):
            reference = power_mean_log_target(values, rho)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "target.safetensors"
                save_file({"logq": reference.float()}, str(path))
                actual = load_file(str(path))["logq"]
            self.assertTrue(bool(torch.isfinite(actual).all()))
            torch.testing.assert_close(actual.double(), reference, atol=1.5e-4, rtol=1e-7)
            torch.testing.assert_close(actual.exp().sum(-1), torch.ones(1), atol=1e-7, rtol=1e-7)


if __name__ == "__main__":
    unittest.main()
