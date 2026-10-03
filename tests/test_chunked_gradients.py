from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from cot_mtkd.models.chunked_head import (
    cross_entropy_hidden_gradient,
    full_vocab_probe,
    gather_support_logits,
    support_vjp_hidden_gradient,
)
from cot_mtkd.stage2.losses import dual_source_hidden_gradients
from cot_mtkd.stage2.tail_kl import sparse_topk_with_tail, tail_bucket_kl
from cot_mtkd.utils.seed import deterministic_rng


class ChunkedGradientTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(17)
        self.head = torch.nn.Linear(7, 19, bias=False)
        self.hidden = torch.randn(9, 7)
        self.targets = torch.randint(0, 19, (9,))

    def test_chunked_cross_entropy_matches_dense_autograd(self) -> None:
        gradient, loss, count = cross_entropy_hidden_gradient(
            self.hidden, self.head, self.targets, chunk_tokens=3
        )
        dense_hidden = self.hidden.detach().requires_grad_(True)
        dense_loss = F.cross_entropy(
            self.head(dense_hidden).float(), self.targets, reduction="sum"
        )
        dense_gradient = torch.autograd.grad(dense_loss, dense_hidden)[0]
        self.assertTrue(
            torch.allclose(gradient, dense_gradient, atol=1.0e-6, rtol=1.0e-5)
        )
        self.assertAlmostEqual(float(loss), float(dense_loss.detach()), places=5)
        self.assertEqual(count, self.hidden.shape[0])

    def test_support_vjp_matches_direct_autograd(self) -> None:
        support = torch.stack([torch.randperm(19)[:5] for _ in range(9)])
        cotangent = torch.randn(9, 5)
        gradient = support_vjp_hidden_gradient(
            self.hidden, self.head, support, cotangent, chunk_tokens=4
        )
        dense_hidden = self.hidden.detach().requires_grad_(True)
        selected = self.head(dense_hidden).float().gather(-1, support)
        dense_gradient = torch.autograd.grad(
            (selected * cotangent).sum(), dense_hidden
        )[0]
        self.assertTrue(
            torch.allclose(gradient, dense_gradient, atol=1.0e-6, rtol=1.0e-5)
        )

    def _assert_large_chunk_parity(self, device: torch.device) -> None:
        head = self.head.to(device)
        hidden = self.hidden.to(device)
        targets = self.targets.to(device)
        small_probe = full_vocab_probe(hidden, head, targets, 6, 3)
        large_probe = full_vocab_probe(
            hidden, head, targets, 6, 32768, output_device=device
        )
        for small, large in zip(small_probe, large_probe, strict=True):
            self.assertEqual(small.device.type, "cpu")
            self.assertEqual(large.device, device)
            if small.is_floating_point():
                torch.testing.assert_close(small, large.cpu(), atol=1e-6, rtol=1e-5)
            else:
                self.assertTrue(torch.equal(small, large.cpu()))
        self.assertEqual(large_probe[1].dtype, torch.int32)
        self.assertFalse((large_probe[1] == targets[:, None]).any())
        support = large_probe[1].long()
        small_logits = gather_support_logits(hidden, head, support, 3)
        large_logits = gather_support_logits(hidden, head, support, 32768)
        torch.testing.assert_close(small_logits, large_logits, atol=1e-6, rtol=1e-5)
        cotangent = torch.randn_like(large_logits)
        for chunk_tokens in (3, 32768):
            ce_gradient, ce_loss, count = cross_entropy_hidden_gradient(
                hidden, head, targets, chunk_tokens
            )
            dpp_gradient = support_vjp_hidden_gradient(
                hidden, head, support, cotangent, chunk_tokens
            )
            if chunk_tokens == 3:
                expected_ce, expected_loss, expected_dpp = ce_gradient, ce_loss, dpp_gradient
            else:
                torch.testing.assert_close(ce_gradient, expected_ce, atol=1e-6, rtol=1e-5)
                torch.testing.assert_close(ce_loss, expected_loss, atol=1e-6, rtol=1e-5)
                torch.testing.assert_close(dpp_gradient, expected_dpp, atol=1e-6, rtol=1e-5)
            self.assertEqual(count, hidden.shape[0])

    def test_chunk_32768_matches_small_chunks(self) -> None:
        self._assert_large_chunk_parity(torch.device("cpu"))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_probe_and_chunk_32768_parity(self) -> None:
        self._assert_large_chunk_parity(torch.device("cuda:0"))

    def test_dual_source_chunking_matches_dense_autograd(self) -> None:
        hard_weights = torch.linspace(0.25, 1.75, self.hidden.shape[0])
        teacher = torch.softmax(torch.randn(9, 19) / 2.0, dim=-1)
        support, teacher_top, teacher_tail = sparse_topk_with_tail(teacher, top_k=6)
        hard_grad, kd_grad, hard_sum, kd_sum, hard_count, kd_count = (
            dual_source_hidden_gradients(
                self.hidden,
                self.head,
                self.targets,
                hard_weights,
                support,
                teacher_top,
                teacher_tail,
                temperature=2.0,
                epsilon=1.0e-8,
                chunk_tokens=3,
            )
        )

        dense_hidden = self.hidden.detach().requires_grad_(True)
        logits = self.head(dense_hidden).float()
        dense_hard = (
            F.cross_entropy(logits, self.targets, reduction="none") * hard_weights
        ).sum()
        dense_kd = tail_bucket_kl(
            logits, support, teacher_top, teacher_tail, temperature=2.0
        ).sum()
        expected_hard = torch.autograd.grad(
            dense_hard, dense_hidden, retain_graph=True
        )[0]
        expected_kd = torch.autograd.grad(dense_kd, dense_hidden)[0]
        self.assertTrue(
            torch.allclose(hard_grad, expected_hard, atol=1.0e-6, rtol=1.0e-5)
        )
        self.assertTrue(torch.allclose(kd_grad, expected_kd, atol=1.0e-6, rtol=1.0e-5))
        self.assertAlmostEqual(hard_sum, float(dense_hard.detach()), places=5)
        self.assertAlmostEqual(kd_sum, float(dense_kd.detach()), places=5)
        self.assertAlmostEqual(hard_count, float(hard_weights.sum()), places=6)
        self.assertEqual(kd_count, self.hidden.shape[0])

    def test_two_pass_rng_replay_matches_direct_parameter_gradient(self) -> None:
        torch.manual_seed(29)
        projection = torch.nn.Linear(6, 7)
        dropout = torch.nn.Dropout(0.35)
        inputs = torch.randn(8, 6)
        support = torch.stack([torch.randperm(19)[:4] for _ in range(8)])
        seed = 4242

        with deterministic_rng(seed, torch.device("cpu")), torch.no_grad():
            probe_hidden = dropout(projection(inputs))
        probe_logits = (
            self.head(probe_hidden).gather(-1, support).detach().requires_grad_(True)
        )
        probe_loss = torch.logsumexp(probe_logits, dim=-1).sum()
        logit_cotangent = torch.autograd.grad(probe_loss, probe_logits)[0]

        with deterministic_rng(seed, torch.device("cpu")):
            replay_hidden = dropout(projection(inputs))
        hidden_cotangent = support_vjp_hidden_gradient(
            replay_hidden, self.head, support, logit_cotangent, chunk_tokens=3
        )
        replay_gradient = torch.autograd.grad(
            replay_hidden, projection.weight, grad_outputs=hidden_cotangent
        )[0]

        with deterministic_rng(seed, torch.device("cpu")):
            direct_hidden = dropout(projection(inputs))
            direct_logits = self.head(direct_hidden).gather(-1, support)
            direct_loss = torch.logsumexp(direct_logits, dim=-1).sum()
        direct_gradient = torch.autograd.grad(direct_loss, projection.weight)[0]
        self.assertTrue(torch.equal(probe_hidden, replay_hidden))
        self.assertTrue(
            torch.allclose(replay_gradient, direct_gradient, atol=1.0e-6, rtol=1.0e-5)
        )


if __name__ == "__main__":
    unittest.main()
