from __future__ import annotations

import unittest

import torch

from cot_mtkd.stage1.dpp import (
    DPPMetrics,
    _batched_cholesky_logdet,
    normalized_support_features,
    step_dpp_loss,
)


def reference_logdet(
    gram: torch.Tensor, jitter: float, maximum_jitter: float
) -> tuple[torch.Tensor, float, int]:
    """The original scalar retry implementation, independent of the batched path."""
    identity = torch.eye(gram.shape[-1], device=gram.device, dtype=torch.float32)
    fallbacks = 0
    while True:
        factor, info = torch.linalg.cholesky_ex(gram.float() + jitter * identity)
        if int(info.item()) == 0:
            return 2 * torch.log(torch.diagonal(factor)).sum(), jitter, fallbacks
        if jitter >= maximum_jitter:
            sign, value = torch.linalg.slogdet(gram.float() + maximum_jitter * identity)
            if float(sign.item()) <= 0:
                raise FloatingPointError("non-positive determinant")
            return value, maximum_jitter, fallbacks + 1
        jitter = min(maximum_jitter, jitter * 10)
        fallbacks += 1


def reference_step_loss(
    features: torch.Tensor,
    samples: torch.Tensor,
    steps: torch.Tensor,
    reduction: str,
) -> tuple[torch.Tensor, DPPMetrics]:
    pairs = torch.stack([samples, steps], dim=-1)
    unique_pairs = torch.unique(pairs, dim=0)
    per_sample: dict[int, list[torch.Tensor]] = {}
    fallbacks = 0
    largest_jitter = 1e-4
    for pair in unique_pairs:
        mask = (pairs == pair).all(dim=-1)
        selected = features[:, mask, :]
        gram = torch.einsum("mtk,ntk->mn", selected, selected) / mask.sum()
        value, used_jitter, count = reference_logdet(gram, 1e-4, 1e-2)
        per_sample.setdefault(int(pair[0]), []).append(-value / features.shape[0])
        largest_jitter = max(largest_jitter, used_jitter)
        fallbacks += count
    values = torch.stack([torch.stack(parts).mean() for parts in per_sample.values()])
    loss = values.mean() if reduction == "mean" else values.sum() if reduction == "sum" else values
    return loss, DPPMetrics(len(unique_pairs), len(values), fallbacks, largest_jitter)


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

    def assert_grouped_parity(self, device: str) -> None:
        torch.manual_seed(24)
        # Unordered tokens, sparse metadata, unequal numbers of tokens and
        # steps: a global token/step mean would fail this comparison.
        samples = torch.tensor([19, 4, 19, 4, 4, 19, 31, 4, 19, 4, 19], device=device)
        steps = torch.tensor([8, 7, 2, 1, 7, 8, 20, 7, 2, 9, 8], device=device)
        for reduction in ("mean", "sum", "none"):
            with self.subTest(device=device, reduction=reduction):
                actual_features = torch.randn(3, 11, 13, device=device)
                actual_features /= actual_features.norm(dim=-1, keepdim=True)
                actual_features.requires_grad_(True)
                reference_features = actual_features.detach().clone().requires_grad_(True)
                actual, actual_metrics = step_dpp_loss(
                    actual_features, samples, steps, reduction=reduction
                )
                expected, expected_metrics = reference_step_loss(
                    reference_features, samples, steps, reduction
                )
                actual_gradient = torch.autograd.grad(actual.sum(), actual_features)[0]
                expected_gradient = torch.autograd.grad(expected.sum(), reference_features)[0]
                torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
                torch.testing.assert_close(
                    actual_gradient, expected_gradient, atol=2e-6, rtol=2e-5
                )
                self.assertEqual(actual_metrics, expected_metrics)
                self.assertEqual(actual.device.type, device)

    def test_grouped_loss_and_gradient_match_original(self) -> None:
        self.assert_grouped_parity("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_grouped_loss_and_gradient_match_original_on_cuda(self) -> None:
        self.assert_grouped_parity("cuda")

    def test_single_expert_and_token_permutation(self) -> None:
        features = torch.randn(1, 7, 5, requires_grad=True)
        samples = torch.tensor([8, 8, 3, 3, 3, 8, 10])
        steps = torch.tensor([9, 2, 2, 2, 4, 9, 1])
        ordering = torch.tensor([4, 6, 1, 0, 5, 3, 2])
        original, metrics = step_dpp_loss(features, samples, steps, reduction="none")
        permuted, reordered_metrics = step_dpp_loss(
            features[:, ordering], samples[ordering], steps[ordering], reduction="none"
        )
        torch.testing.assert_close(original, permuted)
        self.assertEqual(metrics, reordered_metrics)
        self.assertEqual(metrics.groups, 5)
        self.assertEqual(metrics.samples, 3)

    def test_empty_input_preserves_differentiable_zero(self) -> None:
        for reduction in ("mean", "sum", "none"):
            features = torch.empty(3, 0, 7, requires_grad=True)
            metadata = torch.empty(0, dtype=torch.long)
            loss, metrics = step_dpp_loss(features, metadata, metadata, reduction=reduction)
            self.assertEqual(float(loss.detach()), 0)
            self.assertEqual(metrics, DPPMetrics(0, 0, 0, 1e-4))
            loss.backward()
            self.assertEqual(features.grad.shape, features.shape)

    def test_batched_retry_counts_and_gradients_match_scalar(self) -> None:
        matrices = torch.stack(
            [
                torch.eye(3),
                torch.diag(torch.tensor([-5e-4, 1.0, 1.0])),
                torch.diag(torch.tensor([-5e-3, 1.0, 1.0])),
                torch.diag(torch.tensor([-2e-2, -2e-2, 1.0])),
            ]
        ).requires_grad_(True)
        reference = matrices.detach().clone().requires_grad_(True)
        actual, used_jitter, fallbacks = _batched_cholesky_logdet(matrices, 1e-4, 1e-2)
        scalar_results = [reference_logdet(matrix, 1e-4, 1e-2) for matrix in reference]
        expected = torch.stack([result[0] for result in scalar_results])
        self.assertEqual(fallbacks, 6)
        self.assertEqual(fallbacks, sum(result[2] for result in scalar_results))
        self.assertEqual(used_jitter, max(result[1] for result in scalar_results))
        torch.testing.assert_close(actual, expected)
        actual_gradient = torch.autograd.grad(actual.sum(), matrices)[0]
        expected_gradient = torch.autograd.grad(expected.sum(), reference)[0]
        self.assertTrue(torch.isfinite(actual_gradient).all())
        torch.testing.assert_close(actual_gradient, expected_gradient)

    def test_batched_retry_rejects_nonpositive_maximum_determinant(self) -> None:
        matrices = torch.stack([torch.eye(3), torch.diag(torch.tensor([-1.0, 1.0, 1.0]))])
        with self.assertRaises(FloatingPointError):
            _batched_cholesky_logdet(matrices, 1e-4, 1e-2)

    def assert_nearly_identical_parity(self, device: str) -> None:
        torch.manual_seed(71)
        samples = torch.tensor([17] * 51 + [2] * 37, device=device)
        steps = torch.tensor([8] * 30 + [14] * 21 + [3] * 17 + [10] * 20, device=device)
        mask = torch.ones(88, 64, dtype=torch.bool, device=device)
        mask[:, 48:] = False
        base = torch.randn(1, 88, 64, device=device)
        for noise in (0.0, 1e-3):
            with self.subTest(device=device, noise=noise):
                actual_logits = (
                    base.repeat(3, 1, 1) + noise * torch.randn(3, 88, 64, device=device)
                ).requires_grad_(True)
                reference_logits = actual_logits.detach().clone().requires_grad_(True)
                actual, metrics = step_dpp_loss(
                    normalized_support_features(actual_logits, mask), samples, steps
                )
                expected, expected_metrics = reference_step_loss(
                    normalized_support_features(reference_logits, mask), samples, steps, "mean"
                )
                actual_gradient = torch.autograd.grad(actual, actual_logits)[0]
                expected_gradient = torch.autograd.grad(expected, reference_logits)[0]
                # Near rank-one matrices amplify FP32 sum-order differences by
                # 1 / jitter. Both implementations use the same 1e-4 jitter.
                torch.testing.assert_close(actual, expected, atol=2e-3, rtol=5e-4)
                torch.testing.assert_close(
                    actual_gradient, expected_gradient, atol=5e-5, rtol=5e-3
                )
                self.assertTrue(torch.isfinite(actual_gradient).all())
                self.assertEqual(metrics, expected_metrics)

    def test_identical_and_nearly_identical_loss_and_gradient_parity(self) -> None:
        self.assert_nearly_identical_parity("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_identical_and_nearly_identical_parity_on_cuda(self) -> None:
        self.assert_nearly_identical_parity("cuda")

    def assert_long_step_stability(self, device: str) -> None:
        torch.manual_seed(104)
        token_count, support = 4096, 384
        base = torch.randn(1, token_count, support, device=device)
        logits = (
            base.repeat(3, 1, 1) + 1e-3 * torch.randn(3, token_count, support, device=device)
        ).requires_grad_(True)
        reference_logits = logits.detach().clone().requires_grad_(True)
        mask = torch.ones(token_count, support, dtype=torch.bool, device=device)
        # CPU metadata also exercises transfer to the feature device on CUDA.
        samples = torch.full((token_count,), 5, dtype=torch.long)
        steps = torch.tensor([8] * 2048 + [12] * 1024 + [9] * 1024)
        actual, metrics = step_dpp_loss(
            normalized_support_features(logits, mask), samples, steps
        )
        expected_features = normalized_support_features(reference_logits, mask).double()
        identity = torch.eye(3, dtype=torch.float64, device=device)
        losses = []
        for step in torch.unique(steps):
            selected = expected_features[:, (steps == step).to(device), :]
            gram = torch.einsum("mtk,ntk->mn", selected, selected) / selected.shape[1]
            factor = torch.linalg.cholesky(gram + 1e-4 * identity)
            losses.append(-2 * torch.log(torch.diagonal(factor)).sum() / 3)
        expected = torch.stack(losses).mean()
        actual_gradient = torch.autograd.grad(actual, logits)[0]
        expected_gradient = torch.autograd.grad(expected, reference_logits)[0]
        # A scalar FP32 reference itself loses accuracy for long rank-one
        # steps. Compare to FP64 mathematically identical Gram/logdet instead.
        self.assertLess(float((actual.double() - expected).detach().abs()), 8e-4)
        relative_gradient_error = (actual_gradient - expected_gradient).norm() / (
            expected_gradient.norm()
        )
        self.assertLess(float(relative_gradient_error), 5e-3)
        self.assertTrue(torch.isfinite(actual_gradient).all())
        self.assertEqual(metrics, DPPMetrics(3, 1, 0, 1e-4))

    def test_long_nearly_identical_steps_match_double_precision_oracle(self) -> None:
        self.assert_long_step_stability("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_long_nearly_identical_steps_match_double_precision_oracle_on_cuda(self) -> None:
        self.assert_long_step_stability("cuda")

    def test_bfloat16_features_preserve_scalar_loss_and_gradient(self) -> None:
        torch.manual_seed(58)
        features = torch.randn(3, 71, 31)
        features /= features.norm(dim=-1, keepdim=True)
        features = features.bfloat16().requires_grad_(True)
        reference_features = features.detach().clone().requires_grad_(True)
        samples = torch.tensor([3] * 41 + [10] * 30)
        steps = torch.tensor([5] * 20 + [8] * 21 + [8] * 30)
        actual, metrics = step_dpp_loss(features, samples, steps)
        expected, expected_metrics = reference_step_loss(
            reference_features, samples, steps, "mean"
        )
        actual_gradient = torch.autograd.grad(actual, features)[0]
        expected_gradient = torch.autograd.grad(expected, reference_features)[0]
        torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)
        torch.testing.assert_close(actual_gradient, expected_gradient, atol=1e-4, rtol=1e-2)
        self.assertEqual(metrics, expected_metrics)

    def test_bfloat16_support_logits_keep_features_and_loss_float32(self) -> None:
        logits = torch.randn(3, 7, 11, dtype=torch.bfloat16, requires_grad=True)
        mask = torch.ones(7, 11, dtype=torch.bool)
        features = normalized_support_features(logits, mask)
        loss, _ = step_dpp_loss(
            features, torch.zeros(7, dtype=torch.long), torch.zeros(7, dtype=torch.long)
        )
        self.assertEqual(features.dtype, torch.float32)
        self.assertEqual(loss.dtype, torch.float32)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_metadata_and_reduction_validation(self) -> None:
        features = torch.ones(3, 2, 4)
        metadata = torch.zeros(2, dtype=torch.long)
        with self.assertRaises(ValueError):
            step_dpp_loss(features, metadata[:1], metadata)
        with self.assertRaises(ValueError):
            step_dpp_loss(features, metadata, torch.tensor([0, -1]))
        with self.assertRaises(ValueError):
            step_dpp_loss(features, metadata, torch.tensor([0.0, -0.5]))
        with self.assertRaises(ValueError):
            step_dpp_loss(features, metadata, metadata, reduction="invalid")


if __name__ == "__main__":
    unittest.main()
