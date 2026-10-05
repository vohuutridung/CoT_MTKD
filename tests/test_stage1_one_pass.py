from __future__ import annotations

import unittest
from unittest.mock import patch

import torch
from torch.utils.data import DistributedSampler
from transformers import Qwen2Config, Qwen2ForCausalLM

from cot_mtkd.cli.stress_stage1_memory import extend_reasoning
from cot_mtkd.data.schema import PreparedRecord
from cot_mtkd.models.chunked_head import forward_hidden
from cot_mtkd.models.multi_adapter import (
    adapter_parameter_groups,
    create_multi_adapter_model,
    set_all_adapters_trainable,
)
from cot_mtkd.stage1.gac_gradient import stable_gac_gradients
from cot_mtkd.stage1.rbf import (
    BandwidthEMA,
    effective_update_distances,
    interaction_bandwidths,
    rbf_kernel,
    repulsion_updates,
)
from cot_mtkd.stage1.trainer import (
    _planned_window_counts,
    _run_compatible_config,
    one_pass_expert_gradients,
    probe_stage1_dpp,
    replay_expert_gradients,
    sft_only_expert_gradients,
)


class Stage1OnePassTest(unittest.TestCase):
    def test_window_denominators_cross_epoch_boundaries(self) -> None:
        records = []
        for index, (tokens, reasoning) in enumerate(((2, True), (3, False), (4, True))):
            regions = [0] + ([2] if reasoning else [5]) + [5] * (tokens - 1)
            records.append(
                PreparedRecord(
                    sample_id=str(index),
                    input_ids=list(range(tokens + 1)),
                    labels=[-100] + list(range(1, tokens + 1)),
                    attention_mask=[1] * (tokens + 1),
                    offset_mapping=[(0, 0)] * (tokens + 1),
                    region_ids=regions,
                    step_ids=[-1] + ([0] if reasoning else [-1]) + [-1] * (tokens - 1),
                    question="",
                    thinking="",
                    solution="",
                    deepseek_grade=None,
                    original_length=tokens + 1,
                    kept_length=tokens + 1,
                    original_steps=int(reasoning),
                    kept_steps=int(reasoning),
                    truncated=False,
                    answer_start=1,
                    reasoning_start=1,
                    tokenizer_fingerprint="x",
                )
            )
        sampler = DistributedSampler(records, num_replicas=1, rank=0, shuffle=False)
        self.assertEqual(
            _planned_window_counts(records, sampler, epochs=2, micro_batch=1, accumulation_steps=2),
            [(5, 1), (6, 2), (7, 1)],
        )

    def test_forward_mode_does_not_block_resume(self) -> None:
        first = {"stage1": {"forward_mode": "one_pass", "resume_from": None, "epochs": 3}}
        second = {
            "stage1": {
                "forward_mode": "two_pass",
                "resume_from": "checkpoint.pt",
                "epochs": 3,
            }
        }
        self.assertEqual(_run_compatible_config(first), _run_compatible_config(second))

    def test_synthetic_case_is_exact_length_and_preserves_answer(self) -> None:
        record = PreparedRecord(
            sample_id="real",
            input_ids=[1, 2, 3, 4, 5, 6, 7, 8],
            labels=[-100, -100, 3, 4, 5, 6, 7, 8],
            attention_mask=[1] * 8,
            offset_mapping=[(0, 0)] * 8,
            region_ids=[0, 0, 2, 2, 2, 3, 5, 6],
            step_ids=[-1, -1, 0, 0, 1, -1, -1, -1],
            question="",
            thinking="",
            solution="",
            deepseek_grade=None,
            original_length=8,
            kept_length=8,
            original_steps=2,
            kept_steps=2,
            truncated=False,
            answer_start=6,
            reasoning_start=2,
            tokenizer_fingerprint="x",
        )
        result = extend_reasoning(record, 12)
        for field in (
            "input_ids",
            "labels",
            "attention_mask",
            "offset_mapping",
            "region_ids",
            "step_ids",
        ):
            self.assertEqual(len(getattr(result, field)), 12)
        self.assertEqual(result.input_ids[:5], record.input_ids[:5])
        self.assertEqual(result.input_ids[-3:], record.input_ids[-3:])
        self.assertEqual(result.answer_start, 10)

    def test_matches_two_pass_with_checkpointing_and_dropout(self) -> None:
        torch.manual_seed(123)
        base = Qwen2ForCausalLM(
            Qwen2Config(
                vocab_size=64,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=1,
                max_position_embeddings=32,
            )
        )
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        base.enable_input_require_grads()
        lora = {
            "rank": 2,
            "alpha": 2,
            "dropout": 0.2,
            "target_modules": [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        }
        with patch("cot_mtkd.models.multi_adapter.load_base_causal_lm", return_value=base):
            model, names = create_multi_adapter_model({}, lora, 3, torch.device("cpu"))
        groups = adapter_parameter_groups(model, names)
        parameters = [list(group.values()) for group in groups]
        with torch.no_grad():
            for current in groups:
                for key, value in current.items():
                    if "lora_B" in key:
                        value.normal_(std=0.02)
        ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]])
        batch = {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
            "labels": torch.tensor([[-100, -100, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]]),
            "region_ids": torch.tensor([[0, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 4, 5, 6]]),
            "step_ids": torch.tensor([[-1, -1, -1, 0, 0, 0, 0, 1, 1, 1, 1, -1, -1, -1]]),
        }
        config = {
            "runtime": {"lm_head_chunk_tokens": 4, "probe_hidden_device": "cpu"},
            "kneedle": {"search_k": 6, "k_min": 8},
            "dpp": {"jitter": 1.0e-4, "max_jitter": 1.0e-2},
        }
        device = torch.device("cpu")
        old_probe = probe_stage1_dpp(model, names, batch, 0, 0, 42, config, device)
        old_results = [
            replay_expert_gradients(
                model, name, index, parameter, batch, old_probe, 0, 0, 42, 4, device
            )
            for index, (name, parameter) in enumerate(zip(names, parameters, strict=True))
        ]
        original_grad = torch.autograd.grad
        transformer_vjps: list[int] = []

        def count_transformer_vjp(outputs, inputs, *args, **kwargs):
            if isinstance(inputs, list) and isinstance(outputs, torch.Tensor) and outputs.ndim == 2:
                transformer_vjps.append(1)
            return original_grad(outputs, inputs, *args, **kwargs)

        with (
            patch("cot_mtkd.stage1.trainer.forward_hidden", wraps=forward_hidden) as traced,
            patch("torch.autograd.grad", side_effect=count_transformer_vjp),
        ):
            new_probe, new_results = one_pass_expert_gradients(
                model, names, parameters, batch, 0, 0, 42, config, device
            )
        self.assertEqual(traced.call_count, 3)
        self.assertEqual(len(transformer_vjps), 6)
        self.assertTrue(torch.equal(old_probe.support_ids, new_probe.support_ids))
        self.assertTrue(torch.equal(old_probe.support_mask, new_probe.support_mask))
        self.assertEqual(new_probe.selection_count, 24)
        self.assertEqual(new_probe.mean_selected_k, 6)
        self.assertEqual(new_probe.selected_k_histogram.tolist(), [0] * 6 + [24])
        self.assertEqual(int(new_probe.raw_k_histogram.sum()), 24)
        self.assertTrue(torch.equal(old_probe.raw_k_histogram, new_probe.raw_k_histogram))
        self.assertTrue(
            torch.allclose(
                old_probe.dpp_logit_gradients,
                new_probe.dpp_logit_gradients,
                atol=1.0e-6,
                rtol=1.0e-5,
            )
        )
        self.assertAlmostEqual(old_probe.dpp_loss_sum, new_probe.dpp_loss_sum, places=6)
        for old, new in zip(old_results, new_results, strict=True):
            self.assertAlmostEqual(old[2], new[2], places=5)
            self.assertEqual(old[3], new[3])
            for old_set, new_set in zip(old[:2], new[:2], strict=True):
                for old_gradient, new_gradient in zip(old_set, new_set, strict=True):
                    self.assertTrue(
                        torch.allclose(old_gradient, new_gradient, atol=2.0e-5, rtol=1.0e-4)
                    )
        large_config = {**config, "runtime": {**config["runtime"], "lm_head_chunk_tokens": 32768}}
        large_probe, large_results = one_pass_expert_gradients(
            model, names, parameters, batch, 0, 0, 42, large_config, device
        )
        self.assertTrue(torch.equal(new_probe.support_ids, large_probe.support_ids))
        self.assertTrue(torch.equal(new_probe.support_mask, large_probe.support_mask))
        torch.testing.assert_close(
            new_probe.dpp_logit_gradients,
            large_probe.dpp_logit_gradients,
            atol=1e-6,
            rtol=1e-5,
        )
        self.assertAlmostEqual(new_probe.dpp_loss_sum, large_probe.dpp_loss_sum, places=6)
        for small, large in zip(new_results, large_results, strict=True):
            self.assertAlmostEqual(small[2], large[2], places=5)
            self.assertEqual(small[3], large[3])
            for small_set, large_set in zip(small[:2], large[:2], strict=True):
                for small_gradient, large_gradient in zip(small_set, large_set, strict=True):
                    torch.testing.assert_close(small_gradient, large_gradient, atol=2e-5, rtol=1e-4)
        scale = 0.75
        transformer_vjps.clear()
        with patch("torch.autograd.grad", side_effect=count_transformer_vjp):
            _, combined_results = one_pass_expert_gradients(
                model,
                names,
                parameters,
                batch,
                0,
                0,
                42,
                large_config,
                device,
                combined_dpp_scale=scale,
            )
        self.assertEqual(len(transformer_vjps), 3)
        for old, combined in zip(old_results, combined_results, strict=True):
            self.assertEqual(combined[1], [])
            for sft, dpp, actual in zip(old[0], old[1], combined[0], strict=True):
                self.assertTrue(torch.allclose(actual, sft + scale * dpp, atol=2.0e-5, rtol=1.0e-4))
        fallback_combined = replay_expert_gradients(
            model,
            names[0],
            0,
            parameters[0],
            batch,
            old_probe,
            0,
            0,
            42,
            4,
            device,
            combined_dpp_scale=scale,
        )
        for sft, dpp, actual in zip(
            old_results[0][0], old_results[0][1], fallback_combined[0], strict=True
        ):
            self.assertTrue(torch.allclose(actual, sft + scale * dpp, atol=2.0e-5, rtol=1.0e-4))
        # Compare the full new pseudo-gradient, including nonzero RBF repulsion.
        tokens = old_results[0][3]
        full_scale = 0.1 * tokens / old_probe.dpp_sample_count
        _, full_one_pass = one_pass_expert_gradients(
            model, names, parameters, batch, 0, 0, 42, config, device,
            combined_dpp_scale=full_scale,
        )
        full_two_pass = [
            replay_expert_gradients(
                model, name, expert, current, batch, old_probe, 0, 0, 42, 4, device,
                combined_dpp_scale=full_scale,
            )
            for expert, (name, current) in enumerate(zip(names, parameters, strict=True))
        ]
        set_all_adapters_trainable(model, names)
        distances = effective_update_distances(groups, 1.0)
        h_base = BandwidthEMA().update(distances)
        h_gac, h_rbf = interaction_bandwidths(h_base)
        repulsion, _, _ = repulsion_updates(groups, 1.0, h_rbf, distances=distances)
        self.assertTrue(any(value.abs().max() > 0 for current in repulsion for value in current))
        kg = rbf_kernel(distances.detach(), h_gac)
        finals = []
        for results in (full_one_pass, full_two_pass):
            tasks = [[value / tokens for value in result[0]] for result in results]
            final, _ = stable_gac_gradients(tasks, repulsion, kg, beta=0.5, rbf_weight=0.5)
            finals.append(final)
        for left, right in zip(finals[0], finals[1], strict=True):
            for one, two in zip(left, right, strict=True):
                torch.testing.assert_close(one, two, atol=2e-5, rtol=1e-4)

        keep = [index for index in range(ids.shape[1]) if index not in (5, 6)]
        short_batch = {key: value[:, keep] for key, value in batch.items()}
        short_probe = probe_stage1_dpp(model, names, short_batch, 0, 1, 42, config, device)
        short_separate = [
            replay_expert_gradients(
                model,
                name,
                index,
                parameter,
                short_batch,
                short_probe,
                0,
                1,
                42,
                4,
                device,
            )
            for index, (name, parameter) in enumerate(zip(names, parameters, strict=True))
        ]
        total_tokens = old_results[0][3] + short_separate[0][3]
        total_dpp_samples = old_probe.dpp_sample_count + short_probe.dpp_sample_count
        dpp_weight = 0.1
        window_scale = dpp_weight * total_tokens / total_dpp_samples
        _, long_combined = one_pass_expert_gradients(
            model,
            names,
            parameters,
            batch,
            0,
            0,
            42,
            config,
            device,
            combined_dpp_scale=window_scale,
        )
        _, short_combined = one_pass_expert_gradients(
            model,
            names,
            parameters,
            short_batch,
            0,
            1,
            42,
            config,
            device,
            combined_dpp_scale=window_scale,
        )
        for long_old, short_old, long_new, short_new in zip(
            old_results, short_separate, long_combined, short_combined, strict=True
        ):
            for (
                sft_long,
                dpp_long,
                sft_short,
                dpp_short,
                combined_long,
                combined_short,
            ) in zip(
                long_old[0],
                long_old[1],
                short_old[0],
                short_old[1],
                long_new[0],
                short_new[0],
                strict=True,
            ):
                expected = (sft_long + sft_short) / total_tokens + dpp_weight * (
                    dpp_long + dpp_short
                ) / total_dpp_samples
                actual = (combined_long + combined_short) / total_tokens
                self.assertTrue(torch.allclose(expected, actual, atol=2.0e-5, rtol=1.0e-4))
        with patch("cot_mtkd.stage1.trainer.full_vocab_probe", side_effect=AssertionError):
            warmup = sft_only_expert_gradients(
                model, names[0], 0, parameters[0], batch, 0, 0, 42, 4, device
            )
        self.assertEqual(warmup[1], [])
        for expected, actual in zip(old_results[0][0], warmup[0], strict=True):
            self.assertTrue(torch.allclose(expected, actual, atol=2.0e-5, rtol=1.0e-4))


if __name__ == "__main__":
    unittest.main()
