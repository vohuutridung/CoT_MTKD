from __future__ import annotations

import math
import unittest
from decimal import Decimal, localcontext

import torch
import torch.nn.functional as F

from cot_mtkd.stage2.output_space_losses import (
    adaptive_kd_hidden_gradient,
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

    def test_chunked_step_loss_and_hidden_gradient_match_dense_autograd(self):
        torch.manual_seed(717)
        head = torch.nn.Linear(6, 19, bias=False)
        student = torch.randn(7, 6, requires_grad=True)
        teachers = [torch.randn(7, 6, requires_grad=True) for _ in range(3)]
        temperature = 2.0
        with torch.no_grad():
            logp = torch.stack(
                [F.log_softmax(head(h).double() / temperature, -1) for h in teachers]
            )
            probabilities = logp.exp()
            mixture = probabilities.mean(0)
            ds = float(
                (probabilities * (logp - mixture.log())).sum(-1).mean() / math.log(len(teachers))
            )
            raw = probabilities.pow(ds).mean(0).pow(1 / ds)
            target = raw / raw.sum(-1, keepdim=True)
        log_student = F.log_softmax(head(student).float() / temperature, -1)
        loss = temperature**2 * (target * (target.log() - log_student)).sum(-1).mean()
        expected = torch.autograd.grad(loss, student)[0]
        for chunk in (1, 3, 7, 20):
            gradient, actual_loss, disagreement = adaptive_kd_hidden_gradient(
                student, teachers, head, temperature, chunk
            )
            torch.testing.assert_close(gradient, expected, atol=2.0e-7, rtol=2.0e-5)
            self.assertAlmostEqual(actual_loss, float(loss.detach()), delta=2.0e-7)
            # FP32 head GEMMs can round differently for different chunk shapes.
            self.assertAlmostEqual(disagreement, ds, delta=2.0e-8)
            self.assertFalse(gradient.requires_grad)
        self.assertIsNone(student.grad)
        self.assertIsNone(head.weight.grad)
        self.assertTrue(all(value.grad is None for value in teachers))

    def test_invalid_inputs_are_rejected(self):
        logp = F.log_softmax(torch.zeros(2, 3, 4), -1)
        for rho in (-0.1, 1.1, float("nan")):
            with self.assertRaises(ValueError):
                power_mean_log_target(logp, rho)
        with self.assertRaises(ValueError):
            normalized_js_disagreement(logp[:1])
        with self.assertRaises(ValueError):
            adaptive_kd_hidden_gradient(
                torch.zeros(3, 2), [torch.zeros(3, 2)], torch.nn.Linear(2, 4), 2.0, 1
            )


if __name__ == "__main__":
    unittest.main()
