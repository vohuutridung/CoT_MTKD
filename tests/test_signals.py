from __future__ import annotations

import unittest

import torch

from cot_mtkd.signals.group_importance import group_step_importance
from cot_mtkd.signals.medoid import functional_medoid
from cot_mtkd.signals.teacher_features import reasoning_teacher_weights


class SignalTest(unittest.TestCase):
    def test_importance_has_token_weighted_mean_one(self) -> None:
        pag = torch.tensor(
            [
                [0.1, -0.2, 0.8],
                [0.2, -0.1, 0.7],
                [0.0, 0.1, 0.9],
                [0.3, -0.3, 0.6],
                [0.2, -0.2, 1.0],
            ]
        )
        js = torch.tensor([0.1, 0.2, 0.8])
        counts = torch.tensor([10.0, 40.0, 5.0])
        importance, _ = group_step_importance(pag, js, counts)
        weighted_mean = float((importance * counts).sum() / counts.sum())
        self.assertAlmostEqual(weighted_mean, 1.0, places=5)
        self.assertTrue(torch.isfinite(importance).all())

    def test_teacher_floor_and_normalization(self) -> None:
        competence = torch.tensor([[2.0, 1.0, 0.0, -1.0, -2.0]])
        agreement = torch.zeros_like(competence)
        uniqueness = torch.tensor([[0.0, 0.0, 0.0, 0.0, 20.0]])
        weights, _ = reasoning_teacher_weights(competence, agreement, uniqueness)
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=6)
        self.assertGreaterEqual(float(weights.min()), 0.02 - 1e-7)

    def test_functional_medoid(self) -> None:
        self.assertEqual(functional_medoid(torch.tensor([0.4, 0.1, 0.2])), 1)


if __name__ == "__main__":
    unittest.main()
