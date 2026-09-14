from __future__ import annotations

import unittest

import torch

from cot_mtkd.stage1.dpp import step_dpp_loss


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


if __name__ == "__main__":
    unittest.main()
