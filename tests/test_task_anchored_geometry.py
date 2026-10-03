from __future__ import annotations

import math
import unittest

import torch
import torch.nn.functional as F

from cot_mtkd.stage2.geometry import task_anchored_weights
from cot_mtkd.stage2.geometry_losses import (
    blended_kd_hidden_gradient,
    teacher_kd_hidden_gradient,
)


class TaskAnchoredGeometryTest(unittest.TestCase):
    def test_raw_council_formula_keeps_excluded_teachers_in_agreement(self) -> None:
        first = torch.tensor([4.0, 0.0], requires_grad=True)
        second = torch.tensor([1.0, 1.0], requires_grad=True)
        anchor = torch.tensor([0.0, 1.0], requires_grad=True)
        result = task_anchored_weights([[first], [second]], [anchor])
        expected_agreement = 6.5 / (6.5 + 2.5 + 1.0e-12)
        expected_second_utility = 1.0 / (math.sqrt(2.0) + 1.0e-12)
        expected_mean_utility = 0.5 / (math.sqrt(6.5) + 1.0e-12)
        self.assertAlmostEqual(result.agreement, expected_agreement, places=12)
        self.assertEqual(result.utilities[0], 0.0)
        self.assertAlmostEqual(result.utilities[1], expected_second_utility, places=12)
        self.assertAlmostEqual(result.mean_utility, expected_mean_utility, places=12)
        self.assertAlmostEqual(result.step_weight, expected_second_utility / 2, places=12)
        self.assertAlmostEqual(
            result.consensus_weight, expected_agreement * expected_mean_utility, places=12
        )
        torch.testing.assert_close(result.teacher_weights, torch.tensor([0.0, 1.0]))
        self.assertEqual(result.teacher_weights.device.type, "cpu")
        self.assertFalse(result.teacher_weights.requires_grad)
        normalized = task_anchored_weights(
            [[first / first.norm()], [second / second.norm()]], [anchor]
        )
        self.assertGreater(abs(result.agreement - normalized.agreement), 0.1)
        self.assertIsNone(first.grad)
        self.assertIsNone(second.grad)
        self.assertIsNone(anchor.grad)

    def test_geometry_uses_one_norm_over_all_parameter_blocks(self) -> None:
        teachers = [
            [torch.tensor([3.0, 0.0]), torch.tensor([[4.0]])],
            [torch.tensor([-1.0, 2.0]), torch.tensor([[2.0]])],
            [torch.tensor([0.0, -3.0]), torch.tensor([[1.0]])],
        ]
        anchor = [torch.tensor([0.0, 1.0]), torch.tensor([[2.0]])]
        result = task_anchored_weights(teachers, anchor, epsilon_a=0.2, epsilon_u=0.3)
        flat = task_anchored_weights(
            [[torch.cat([block.flatten() for block in gradient])] for gradient in teachers],
            [torch.cat([block.flatten() for block in anchor])],
            epsilon_a=0.2,
            epsilon_u=0.3,
        )
        self.assertAlmostEqual(result.agreement, flat.agreement, places=12)
        self.assertAlmostEqual(result.mean_utility, flat.mean_utility, places=12)
        self.assertAlmostEqual(result.step_weight, flat.step_weight, places=12)
        for actual, expected in zip(result.utilities, flat.utilities, strict=True):
            self.assertAlmostEqual(actual, expected, places=12)
        torch.testing.assert_close(result.teacher_weights, flat.teacher_weights)

    def test_zero_anchor_negative_utilities_and_zero_teacher(self) -> None:
        anchor = torch.tensor([1.0, 0.0])
        negative = task_anchored_weights(
            [[torch.tensor([-1.0, 1.0])], [torch.tensor([-2.0, -1.0])]], [anchor]
        )
        self.assertEqual(negative.step_weight, 0.0)
        self.assertEqual(negative.consensus_weight, 0.0)
        torch.testing.assert_close(negative.teacher_weights, torch.zeros(2))
        zero_anchor = task_anchored_weights([[anchor], [anchor]], [torch.zeros_like(anchor)])
        self.assertEqual(zero_anchor.utilities, [0.0, 0.0])
        self.assertEqual(zero_anchor.step_weight, 0.0)
        zeros = task_anchored_weights(
            [[torch.zeros_like(anchor)], [torch.zeros_like(anchor)]], [anchor]
        )
        self.assertEqual(zeros.agreement, 0.0)
        self.assertEqual(zeros.mean_utility, 0.0)
        one_teacher_zero = task_anchored_weights([[torch.zeros_like(anchor)], [anchor]], [anchor])
        self.assertEqual(one_teacher_zero.utilities[0], 0.0)
        torch.testing.assert_close(one_teacher_zero.teacher_weights, torch.tensor([0.0, 1.0]))
        self.assertAlmostEqual(one_teacher_zero.step_weight, 0.5, places=10)

    def test_equal_gradients_have_equal_weights_and_exact_epsilon_denominators(self) -> None:
        gradient = torch.tensor([2.0, 0.0])
        result = task_anchored_weights(
            [[gradient], [gradient], [gradient]], [gradient], epsilon_a=2.0, epsilon_u=4.0
        )
        self.assertAlmostEqual(result.agreement, 8.0 / 10.0, places=12)
        self.assertEqual(result.utilities, [0.5, 0.5, 0.5])
        self.assertEqual(result.mean_utility, 0.5)
        self.assertEqual(result.step_weight, 0.5)
        self.assertAlmostEqual(result.consensus_weight, 0.4, places=12)
        torch.testing.assert_close(result.teacher_weights, torch.full((3,), 1.0 / 3.0))

    def test_invalid_geometry_fails_before_returning_weights(self) -> None:
        gradient = torch.ones(2)
        for invalid in (float("nan"), float("inf")):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                task_anchored_weights([[gradient], [torch.tensor([invalid, 0.0])]], [gradient])
        with self.assertRaises(ValueError):
            task_anchored_weights([[gradient]], [gradient])
        with self.assertRaises(ValueError):
            task_anchored_weights([[gradient], [torch.ones(3)]], [gradient])
        with self.assertRaises(ValueError):
            task_anchored_weights([[gradient], [gradient]], [gradient], epsilon_u=0)


class FullVocabularyGeometryLossTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(101)
        self.head = torch.nn.Linear(5, 13, bias=True)
        self.head.requires_grad_(False)
        self.hidden = torch.randn(7, 5)
        self.teachers = [torch.randn_like(self.hidden) for _ in range(3)]
        self.temperature = 2.4

    def _dense_reference(
        self,
        hidden: torch.Tensor,
        weights: torch.Tensor,
        consensus_weight: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            teacher_logits = torch.stack([self.head(value).float() for value in self.teachers])
            probabilities = torch.softmax(teacher_logits / self.temperature, dim=-1)
            coverage = (weights[:, None, None] * probabilities).sum(dim=0)
            consensus = torch.softmax(
                (weights[:, None, None] * teacher_logits).sum(dim=0) / self.temperature,
                dim=-1,
            )
            target = consensus_weight * consensus + (1 - consensus_weight) * coverage
        loss = F.kl_div(
            F.log_softmax(self.head(hidden).float() / self.temperature, dim=-1),
            target,
            reduction="sum",
        ) * (self.temperature**2 / hidden.shape[0])
        return loss, target

    def test_individual_teacher_chunked_cotangent_matches_dense_autograd(self) -> None:
        dense_hidden = self.hidden.detach().requires_grad_(True)
        with torch.no_grad():
            teacher_log_probabilities = F.log_softmax(
                self.head(self.teachers[0]).float() / self.temperature, dim=-1
            )
        loss = F.kl_div(
            F.log_softmax(self.head(dense_hidden).float() / self.temperature, dim=-1),
            teacher_log_probabilities,
            log_target=True,
            reduction="sum",
        ) * (self.temperature**2 / dense_hidden.shape[0])
        expected = torch.autograd.grad(loss, dense_hidden)[0]
        teacher = self.teachers[0].detach().requires_grad_(True)
        for chunk_tokens in (1, 3, 100):
            with self.subTest(chunk_tokens=chunk_tokens):
                gradient, mean_kl = teacher_kd_hidden_gradient(
                    self.hidden, teacher, self.head, self.temperature, chunk_tokens
                )
                torch.testing.assert_close(gradient, expected, atol=1e-7, rtol=1e-5)
                self.assertAlmostEqual(mean_kl, float(loss.detach()), places=6)
                self.assertEqual(gradient.dtype, self.hidden.dtype)
                self.assertEqual(gradient.device, self.hidden.device)
                self.assertFalse(gradient.requires_grad)
        self.assertIsNone(teacher.grad)
        self.assertTrue(all(parameter.grad is None for parameter in self.head.parameters()))

    def test_blended_chunked_cotangent_matches_dense_arithmetic_and_logit_targets(self) -> None:
        weights = torch.tensor([0.25, 0.0, 0.75], requires_grad=True)
        for consensus_weight in (0.0, 0.37, 1.0):
            dense_hidden = self.hidden.detach().requires_grad_(True)
            loss, _ = self._dense_reference(dense_hidden, weights.detach(), consensus_weight)
            expected = torch.autograd.grad(loss, dense_hidden)[0]
            for chunk_tokens in (1, 3, 100):
                with self.subTest(consensus_weight=consensus_weight, chunk_tokens=chunk_tokens):
                    gradient, mean_kl = blended_kd_hidden_gradient(
                        self.hidden, self.teachers, self.head, weights,
                        consensus_weight, self.temperature, chunk_tokens,
                    )
                    torch.testing.assert_close(gradient, expected, atol=1e-7, rtol=1e-5)
                    self.assertAlmostEqual(mean_kl, float(loss.detach()), places=6)
                    self.assertFalse(gradient.requires_grad)
        self.assertIsNone(weights.grad)

    def test_hidden_cotangent_preserves_dense_parameter_gradients(self) -> None:
        projection = torch.nn.Linear(4, 5)
        inputs = torch.randn(7, 4)
        hidden = projection(inputs)
        weights = torch.tensor([0.2, 0.3, 0.5])
        cotangent, _ = blended_kd_hidden_gradient(
            hidden, self.teachers, self.head, weights, 0.6, self.temperature, 3
        )
        actual = torch.autograd.grad(hidden, tuple(projection.parameters()), grad_outputs=cotangent)
        dense_hidden = projection(inputs)
        dense_loss, _ = self._dense_reference(dense_hidden, weights, 0.6)
        expected = torch.autograd.grad(dense_loss, tuple(projection.parameters()))
        for value, reference in zip(actual, expected, strict=True):
            torch.testing.assert_close(value, reference, atol=1e-7, rtol=1e-5)

    def test_equal_student_and_teacher_produce_zero_kd(self) -> None:
        gradient, mean_kl = teacher_kd_hidden_gradient(
            self.hidden, self.hidden.clone(), self.head, self.temperature, 3
        )
        torch.testing.assert_close(gradient, torch.zeros_like(gradient), atol=1e-7, rtol=0)
        self.assertAlmostEqual(mean_kl, 0.0, places=7)

    def test_normalized_geometric_mixture_equals_softmax_weighted_logits(self) -> None:
        weights = torch.tensor([0.2, 0.3, 0.5])
        logits = torch.stack([self.head(value).float() for value in self.teachers])
        log_probabilities = F.log_softmax(logits / self.temperature, dim=-1)
        geometric = torch.softmax((weights[:, None, None] * log_probabilities).sum(0), dim=-1)
        logit_mixture = torch.softmax(
            (weights[:, None, None] * logits).sum(0) / self.temperature, dim=-1
        )
        torch.testing.assert_close(geometric, logit_mixture, atol=1e-7, rtol=1e-5)

    def test_log_space_mixture_handles_extreme_teacher_probabilities_without_floors(self) -> None:
        head = torch.nn.Linear(3, 3, bias=False)
        head.requires_grad_(False)
        with torch.no_grad():
            head.weight.copy_(torch.eye(3))
        student = torch.tensor([[0.0, 1.0, -1.0]])
        teachers = [torch.tensor([[1000.0, -1000.0, 0.0]]), torch.tensor([[-1000.0, 1000.0, 0.0]])]
        for consensus_weight in (0.0, 0.5, 1.0):
            gradient, mean_kl = blended_kd_hidden_gradient(
                student, teachers, head, torch.tensor([0.5, 0.5]), consensus_weight, 1.0, 1
            )
            self.assertTrue(bool(torch.isfinite(gradient).all()))
            self.assertTrue(math.isfinite(mean_kl))
            target = (1 - consensus_weight) * torch.tensor([[0.5, 0.5, 0.0]])
            target += consensus_weight * torch.full((1, 3), 1.0 / 3.0)
            torch.testing.assert_close(gradient, student.softmax(-1) - target, atol=1e-7, rtol=1e-5)

    def test_invalid_kd_inputs_fail_clearly(self) -> None:
        for temperature in (0.0, float("nan")):
            with self.assertRaises(ValueError):
                teacher_kd_hidden_gradient(self.hidden, self.teachers[0], self.head, temperature, 3)
        with self.assertRaises(ValueError):
            teacher_kd_hidden_gradient(self.hidden, self.teachers[0], self.head, 1.0, 0)
        with self.assertRaises(ValueError):
            teacher_kd_hidden_gradient(self.hidden[:0], self.teachers[0][:0], self.head, 1.0, 3)
        with self.assertRaises(ValueError):
            blended_kd_hidden_gradient(
                self.hidden, self.teachers, self.head, torch.zeros(3), 0.5, 1.0, 3
            )
        with self.assertRaises(ValueError):
            blended_kd_hidden_gradient(
                self.hidden, self.teachers, self.head, torch.tensor([0.2, 0.3, 0.5]),
                float("nan"), 1.0, 3,
            )


if __name__ == "__main__":
    unittest.main()
