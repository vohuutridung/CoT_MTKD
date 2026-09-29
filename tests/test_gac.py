from __future__ import annotations

import unittest

import torch

from cot_mtkd.stage1.gac_gradient import stable_gac_gradients
from cot_mtkd.utils.training import interaction_scale


class GACTest(unittest.TestCase):
    def test_zero_interaction_is_independent_sft(self) -> None:
        sft = [[torch.tensor([1.0])], [torch.tensor([3.0])]]
        dpp = [[torch.tensor([8.0])], [torch.tensor([9.0])]]
        repulsion = [[torch.tensor([4.0])], [torch.tensor([-4.0])]]
        final, _ = stable_gac_gradients(
            sft, dpp, repulsion, torch.eye(2), 0.0, dpp_weight=0.2, rbf_weight=1.0
        )
        self.assertTrue(torch.equal(final[0][0], sft[0][0]))
        self.assertTrue(torch.equal(final[1][0], sft[1][0]))

    def test_interaction_schedule(self) -> None:
        self.assertEqual(interaction_scale(0.05), 0.0)
        self.assertAlmostEqual(float(interaction_scale(0.20)), 0.5)
        self.assertEqual(interaction_scale(0.50), 1.0)

    def test_full_phase_precombined_task_matches_separate_gradients(self) -> None:
        sft = [[torch.tensor([1.0, 2.0])], [torch.tensor([3.0, 4.0])]]
        dpp = [[torch.tensor([0.5, -1.0])], [torch.tensor([2.0, 0.25])]]
        repulsion = [[torch.tensor([0.1, 0.2])], [torch.tensor([-0.1, 0.3])]]
        kernel = torch.tensor([[1.0, 0.4], [0.4, 1.0]])
        separate, _ = stable_gac_gradients(
            sft, dpp, repulsion, kernel, 1.0, dpp_weight=0.2, rbf_weight=1.0
        )
        combined = [[local[0] + 0.2 * diverse[0]] for local, diverse in zip(sft, dpp)]
        reused, _ = stable_gac_gradients(
            combined, None, repulsion, kernel, 1.0, dpp_weight=0.2, rbf_weight=1.0
        )
        for old_group, new_group in zip(separate, reused, strict=True):
            self.assertTrue(torch.allclose(old_group[0], new_group[0], atol=1.0e-7))


if __name__ == "__main__":
    unittest.main()
