from __future__ import annotations

import unittest

import torch

from cot_mtkd.stage1.dpp import normalized_support_features, step_dpp_loss


class DPPTest(unittest.TestCase):
    def test_identical_experts_have_larger_loss_than_orthogonal(self) -> None:
        experts = 3
        identical = torch.ones(experts, 4, 3)
        identical = identical / identical.norm(dim=-1, keepdim=True)
        orthogonal = torch.eye(experts).unsqueeze(1).repeat(1, 4, 1)
        samples = torch.zeros(4, dtype=torch.long)
        steps = torch.zeros(4, dtype=torch.long)
        identical_loss, _ = step_dpp_loss(identical, samples, steps)
        orthogonal_loss, _ = step_dpp_loss(orthogonal, samples, steps)
        self.assertGreater(float(identical_loss), float(orthogonal_loss))

    def test_expert_permutation_invariant(self) -> None:
        torch.manual_seed(3)
        features = torch.randn(4, 7, 8)
        features = features / features.norm(dim=-1, keepdim=True)
        samples = torch.tensor([0, 0, 0, 1, 1, 1, 1])
        steps = torch.tensor([0, 0, 1, 0, 0, 1, 1])
        original, _ = step_dpp_loss(features, samples, steps)
        permuted, _ = step_dpp_loss(features[[2, 0, 3, 1]], samples, steps)
        self.assertTrue(torch.allclose(original, permuted, atol=1e-5, rtol=1e-5))

    def test_gradients_are_finite(self) -> None:
        torch.manual_seed(9)
        raw = torch.randn(5, 6, 8, requires_grad=True)
        features = raw / raw.norm(dim=-1, keepdim=True)
        samples = torch.tensor([0, 0, 0, 1, 1, 1])
        steps = torch.tensor([0, 0, 1, 0, 0, 1])
        loss, _ = step_dpp_loss(features, samples, steps)
        loss.backward()
        self.assertIsNotNone(raw.grad)
        self.assertTrue(torch.isfinite(raw.grad).all())

    def test_features_are_l2_of_full_vocab_softmax_slice(self) -> None:
        torch.manual_seed(4)
        logits = torch.tensor(
            [
                [[2.0, 0.5, -1.0, 3.0], [0.0, 1.0, 4.0, -2.0]],
                [[1.0, 1.0, 1.0, 0.0], [5.0, -3.0, 0.5, 0.5]],
            ]
        )
        candidate_ids = torch.tensor([[0, 2], [1, 3]])
        mask = torch.tensor([[True, True], [True, False]])
        log_p = torch.stack(
            [
                torch.log_softmax(logits[0], dim=-1).gather(-1, candidate_ids),
                torch.log_softmax(logits[1], dim=-1).gather(-1, candidate_ids),
            ]
        )
        index = candidate_ids.unsqueeze(0).expand(logits.shape[0], -1, -1)
        features = normalized_support_features(log_p, mask)
        probabilities = torch.softmax(logits, dim=-1).gather(-1, index)
        probabilities = probabilities.masked_fill(~mask.unsqueeze(0), 0.0)
        expected = probabilities / probabilities.norm(dim=-1, keepdim=True).clamp_min(
            1.0e-12
        )
        self.assertTrue(torch.allclose(features, expected, atol=1e-6, rtol=1e-5))
        candidate_only = torch.softmax(
            logits.gather(-1, index).masked_fill(~mask.unsqueeze(0), -torch.inf),
            dim=-1,
        )
        self.assertFalse(
            torch.allclose(
                probabilities[0, 0], candidate_only[0, 0], atol=1e-4, rtol=1e-4
            )
        )


if __name__ == "__main__":
    unittest.main()
