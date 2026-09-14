from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from cot_mtkd.stage2.tail_kl import sparse_topk_with_tail, tail_bucket_kl


class TailBucketKLTest(unittest.TestCase):
    def test_matches_dense_kl_when_support_is_full_vocab(self) -> None:
        torch.manual_seed(11)
        student = torch.randn(4, 7)
        teacher_logits = torch.randn(4, 7)
        teacher = F.softmax(teacher_logits / 2.0, dim=-1)
        ids = torch.arange(7).repeat(4, 1)
        sparse = tail_bucket_kl(
            student,
            ids,
            teacher,
            torch.zeros(4),
            temperature=2.0,
            epsilon=1e-12,
        )
        student_log = F.log_softmax(student / 2.0, dim=-1)
        dense = 4.0 * (teacher * (teacher.log() - student_log)).sum(dim=-1)
        self.assertTrue(torch.allclose(sparse, dense, atol=1e-5, rtol=1e-5))

    def test_sparse_target_preserves_mass(self) -> None:
        probabilities = torch.tensor([[0.5, 0.2, 0.1, 0.1, 0.1]])
        _, top, tail = sparse_topk_with_tail(probabilities, 2)
        self.assertAlmostEqual(float(top.sum() + tail.sum()), 1.0, places=6)

    def test_is_finite_and_nonnegative_with_tiny_tail(self) -> None:
        student = torch.tensor([[2.0, -1.0, 0.5, 0.0]])
        support = torch.tensor([[0, 2]])
        teacher_top = torch.tensor([[0.8, 0.2 - 1.0e-12]])
        value = tail_bucket_kl(
            student,
            support,
            teacher_top,
            torch.tensor([1.0e-12]),
            temperature=2.0,
            epsilon=1.0e-8,
        )
        self.assertTrue(torch.isfinite(value).all())
        self.assertGreaterEqual(float(value.item()), 0.0)


if __name__ == "__main__":
    unittest.main()
