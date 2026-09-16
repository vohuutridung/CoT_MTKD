from __future__ import annotations

import unittest

import torch

from cot_mtkd.data.schema import TokenRegion
from cot_mtkd.stage2.weights import (
    phase2_token_weights,
    relative_disagreement,
    step_weights_from_signals,
    zscore_within_sample,
)


def _batch(regions: list[int], steps: list[int], sample_id: str = "s0") -> dict:
    return {
        "sample_ids": [sample_id],
        "labels": torch.tensor([[-100, *range(len(regions))]]),
        "region_ids": torch.tensor([[int(TokenRegion.PROMPT), *regions]]),
        "step_ids": torch.tensor([[-1, *steps]]),
    }


class Stage2WeightTest(unittest.TestCase):
    def test_zscore_is_zero_mean_unit_std(self) -> None:
        values = torch.tensor([1.0, 2.0, 3.0, 4.0])
        z_values = zscore_within_sample(values)
        self.assertAlmostEqual(float(z_values.mean()), 0.0, places=6)
        self.assertAlmostEqual(float(z_values.std(unbiased=False)), 1.0, places=6)

    def test_single_step_zscore_is_zero(self) -> None:
        self.assertTrue(torch.equal(zscore_within_sample(torch.tensor([0.4])), torch.zeros(1)))

    def test_relative_disagreement(self) -> None:
        rho = relative_disagreement(torch.tensor([2.0, 0.0]), torch.tensor([1.0, 0.3]))
        self.assertAlmostEqual(float(rho[0]), 0.5, places=6)
        self.assertGreater(float(rho[1]), 0.0)

    def test_step_weights_match_proposal_formula(self) -> None:
        uncertainty = torch.tensor([1.0, 2.0, 3.0])
        disagreement = torch.tensor([0.1, 0.2, 0.9])
        lambda_u = 0.5
        lambda_d = 0.5
        actual = step_weights_from_signals(uncertainty, disagreement, lambda_u, lambda_d)
        rho = relative_disagreement(uncertainty, disagreement)
        expected = (
            1.0
            + lambda_u * torch.tanh(zscore_within_sample(uncertainty))
            + lambda_d * torch.tanh(zscore_within_sample(rho))
        )
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))
        self.assertTrue((actual > 0).all())
        self.assertTrue(
            (actual >= 1.0 - lambda_u - lambda_d - 1e-6).all()
            and (actual <= 1.0 + lambda_u + lambda_d + 1e-6).all()
        )

    def test_zero_lambdas_give_uniform_weights(self) -> None:
        weights = step_weights_from_signals(
            torch.tensor([0.2, 0.9, 0.1]),
            torch.tensor([0.01, 0.4, 0.2]),
            0.0,
            0.0,
        )
        self.assertTrue(torch.allclose(weights, torch.ones(3)))

    def test_token_weights_length_normalize_steps_and_always_on_blocks(self) -> None:
        regions = [
            int(TokenRegion.ASSISTANT_CONTROL),
            int(TokenRegion.REASONING),
            int(TokenRegion.REASONING),
            int(TokenRegion.DELIMITER),
            int(TokenRegion.REASONING),
            int(TokenRegion.ANSWER),
            int(TokenRegion.ANSWER),
            int(TokenRegion.EOS),
        ]
        steps = [-1, 0, 0, 0, 1, -1, -1, -1]
        batch = _batch(regions, steps)
        uncertainty = torch.tensor([1.0, 3.0])
        disagreement = torch.tensor([0.1, 0.8])
        signals = {
            "s0": {
                "mean_uncertainty": uncertainty,
                "disagreement": disagreement,
            }
        }
        token_weights, denominator = phase2_token_weights(batch, signals, 0.5, 0.5)
        step_w = step_weights_from_signals(uncertainty, disagreement, 0.5, 0.5)
        self.assertEqual(token_weights.numel(), 8)
        self.assertAlmostEqual(float(token_weights[1:3].sum()), float(step_w[0]), places=5)
        self.assertAlmostEqual(float(token_weights[4]), float(step_w[1]), places=5)
        self.assertAlmostEqual(float(token_weights[5:7].sum()), 1.0, places=5)
        self.assertAlmostEqual(float(token_weights[0] + token_weights[3] + token_weights[7]), 1.0, places=5)
        expected_den = float(step_w.sum().item()) + 2.0
        self.assertAlmostEqual(denominator, expected_den, places=5)

    def test_nll_reduction_is_weighted_step_mean(self) -> None:
        token_nll = torch.tensor([1.0, 2.0, 4.0, 3.0, 10.0, 6.0, 8.0, 5.0])
        regions = [
            int(TokenRegion.ASSISTANT_CONTROL),
            int(TokenRegion.REASONING),
            int(TokenRegion.REASONING),
            int(TokenRegion.DELIMITER),
            int(TokenRegion.REASONING),
            int(TokenRegion.ANSWER),
            int(TokenRegion.ANSWER),
            int(TokenRegion.EOS),
        ]
        steps = [-1, 0, 0, 0, 1, -1, -1, -1]
        batch = _batch(regions, steps)
        signals = {
            "s0": {
                "mean_uncertainty": [1.0, 3.0],
                "disagreement": [0.1, 0.8],
            }
        }
        token_weights, denominator = phase2_token_weights(batch, signals, 0.5, 0.5)
        loss = float((token_nll * token_weights).sum() / denominator)
        step_w = step_weights_from_signals(
            torch.tensor([1.0, 3.0]), torch.tensor([0.1, 0.8]), 0.5, 0.5
        )
        step0 = token_nll[1:3].mean()
        step1 = token_nll[4]
        answer = token_nll[5:7].mean()
        fixed = torch.stack([token_nll[0], token_nll[3], token_nll[7]]).mean()
        expected = float(
            (step_w[0] * step0 + step_w[1] * step1 + answer + fixed) / denominator
        )
        self.assertAlmostEqual(loss, expected, places=5)


if __name__ == "__main__":
    unittest.main()
