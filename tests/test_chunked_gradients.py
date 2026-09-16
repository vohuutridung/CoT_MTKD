from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from cot_mtkd.models.chunked_head import (
    cross_entropy_hidden_gradient,
    gather_support_log_probabilities,
    support_logprob_vjp_hidden_gradient,
    support_vjp_hidden_gradient,
)
from cot_mtkd.stage1.dpp import normalized_support_features
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

    def test_support_logprob_matches_full_vocab_softmax_slice(self) -> None:
        support = torch.stack([torch.randperm(19)[:5] for _ in range(9)])
        mask = torch.ones(9, 5, dtype=torch.bool)
        mask[:, -1] = False
        log_p = gather_support_log_probabilities(
            self.hidden, self.head, support, chunk_tokens=4
        )
        dense = F.log_softmax(self.head(self.hidden).float(), dim=-1)
        expected = dense.gather(-1, support)
        self.assertTrue(torch.allclose(log_p, expected, atol=1e-6, rtol=1e-5))

        candidate_softmax = F.softmax(
            self.head(self.hidden).float().gather(-1, support).masked_fill(
                ~mask, -torch.inf
            ),
            dim=-1,
        )
        self.assertFalse(
            torch.allclose(
                log_p.exp().masked_fill(~mask, 0.0),
                candidate_softmax.masked_fill(~mask, 0.0),
                atol=1e-5,
            )
        )

        features = normalized_support_features(log_p.unsqueeze(0), mask)
        full_slice = dense.exp().gather(-1, support).masked_fill(~mask, 0.0)
        expected_features = full_slice / full_slice.norm(dim=-1, keepdim=True).clamp_min(
            1.0e-12
        )
        self.assertTrue(
            torch.allclose(features[0], expected_features, atol=1e-6, rtol=1e-5)
        )

    def test_support_logprob_vjp_matches_direct_autograd(self) -> None:
        support = torch.stack([torch.randperm(19)[:5] for _ in range(9)])
        cotangent = torch.randn(9, 5)
        gradient = support_logprob_vjp_hidden_gradient(
            self.hidden, self.head, support, cotangent, chunk_tokens=4
        )
        dense_hidden = self.hidden.detach().requires_grad_(True)
        logits = self.head(dense_hidden).float()
        log_p = logits.gather(-1, support) - torch.logsumexp(logits, dim=-1, keepdim=True)
        dense_gradient = torch.autograd.grad((log_p * cotangent).sum(), dense_hidden)[0]
        self.assertTrue(
            torch.allclose(gradient, dense_gradient, atol=1.0e-6, rtol=1.0e-5)
        )

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
            torch.allclose(gradient, dense_gradient, atol=1.0e-6, rtol=1e-5)
        )

    def test_two_pass_logprob_replay_matches_direct_parameter_gradient(self) -> None:
        torch.manual_seed(29)
        projection = torch.nn.Linear(6, 7)
        dropout = torch.nn.Dropout(0.35)
        inputs = torch.randn(8, 6)
        support = torch.stack([torch.randperm(19)[:4] for _ in range(8)])
        seed = 4242
        mask = torch.ones(8, 4, dtype=torch.bool)

        with deterministic_rng(seed, torch.device("cpu")), torch.no_grad():
            probe_hidden = dropout(projection(inputs))
        logits = self.head(probe_hidden).float()
        log_p = (
            logits.gather(-1, support) - torch.logsumexp(logits, dim=-1, keepdim=True)
        ).detach().requires_grad_(True)
        features = normalized_support_features(log_p.unsqueeze(0), mask)
        probe_loss = features.pow(2).sum()
        logprob_cotangent = torch.autograd.grad(probe_loss, log_p)[0]

        with deterministic_rng(seed, torch.device("cpu")):
            replay_hidden = dropout(projection(inputs))
        hidden_cotangent = support_logprob_vjp_hidden_gradient(
            replay_hidden, self.head, support, logprob_cotangent, chunk_tokens=3
        )
        replay_gradient = torch.autograd.grad(
            replay_hidden, projection.weight, grad_outputs=hidden_cotangent
        )[0]

        with deterministic_rng(seed, torch.device("cpu")):
            direct_hidden = dropout(projection(inputs))
            direct_logits = self.head(direct_hidden).float()
            direct_log_p = direct_logits.gather(-1, support) - torch.logsumexp(
                direct_logits, dim=-1, keepdim=True
            )
            direct_features = normalized_support_features(
                direct_log_p.unsqueeze(0), mask
            )
            direct_loss = direct_features.pow(2).sum()
        direct_gradient = torch.autograd.grad(direct_loss, projection.weight)[0]
        self.assertTrue(torch.equal(probe_hidden, replay_hidden))
        self.assertTrue(
            torch.allclose(replay_gradient, direct_gradient, atol=1.0e-6, rtol=1.0e-5)
        )

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
