from __future__ import annotations

import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F
from transformers import Qwen2Config, Qwen2ForCausalLM

from cot_mtkd.data.collator import shifted_token_views
from cot_mtkd.models.multi_adapter import (
    adapter_parameter_groups,
    create_multi_adapter_model,
    set_active_adapter,
    set_all_adapters_trainable,
)
from cot_mtkd.stage1.dpp import normalized_support_features, step_dpp_loss
from cot_mtkd.stage1.objective import STAGE1_METHOD, compose_objective_gradients
from cot_mtkd.stage1.rbf import rbf_repulsion_gradients
from cot_mtkd.stage1.trainer import _run_compatible_config, one_pass_expert_gradients
from cot_mtkd.utils.training import global_clip_grad_list_


class Stage1ObjectiveTest(unittest.TestCase):
    def test_accumulated_gradients_match_one_dense_scalar_objective(self) -> None:
        """Compare the actual split VJPs with backward of all three loss terms."""
        torch.manual_seed(17)
        base = Qwen2ForCausalLM(
            Qwen2Config(
                vocab_size=32,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=1,
            )
        )
        lora = {"rank": 2, "alpha": 2, "dropout": 0.0, "target_modules": ["q_proj", "v_proj"]}
        with patch("cot_mtkd.models.multi_adapter.load_base_causal_lm", return_value=base):
            model, names = create_multi_adapter_model({}, lora, 3, torch.device("cpu"), 42)
        groups = adapter_parameter_groups(model, names)
        parameters = [list(current.values()) for current in groups]
        with torch.no_grad():
            for current in groups:
                for key, value in current.items():
                    if "lora_B" in key:
                        value.normal_(std=0.1)
        config = {
            "runtime": {"lm_head_chunk_tokens": 4, "probe_hidden_device": "cpu"},
            "kneedle": {"search_k": 8, "k_min": 8},
            "dpp": {"jitter": 1e-4, "max_jitter": 1e-2},
        }
        batches = []
        for ids, regions, steps in [
            (
                [1, 2, 3, 4, 5, 6, 7, 8, 9],
                [0, 0, 1, 2, 2, 3, 2, 2, 6],
                [-1, -1, -1, 0, 0, 0, 1, 1, -1],
            ),
            ([1, 2, 3, 10, 11, 12, 13], [0, 0, 1, 2, 2, 3, 6], [-1, -1, -1, 0, 0, 0, -1]),
        ]:
            inputs = torch.tensor([ids])
            batches.append(
                {
                    "input_ids": inputs,
                    "attention_mask": torch.ones_like(inputs),
                    "labels": torch.tensor(
                        [[-100 if region == 0 else token for token, region in zip(ids, regions)]]
                    ),
                    "region_ids": torch.tensor([regions]),
                    "step_ids": torch.tensor([steps]),
                }
            )
        total_tokens = sum(
            shifted_token_views(batch)["response_targets"].numel() for batch in batches
        )
        task = [[torch.zeros_like(value) for value in current] for current in parameters]
        probes = []
        for stream, batch in enumerate(batches):
            probe, results = one_pass_expert_gradients(
                model,
                names,
                parameters,
                batch,
                0,
                stream,
                42,
                config,
                torch.device("cpu"),
                combined_dpp_scale=0.2 * total_tokens / len(batches),
            )
            probes.append(probe)
            for destination, result in zip(task, results, strict=True):
                for value, gradient in zip(destination, result[0], strict=True):
                    value.add_(gradient / total_tokens)
        set_all_adapters_trainable(model, names)
        fixed_bandwidth = 0.01
        rbf = rbf_repulsion_gradients(groups, 1.0, fixed_bandwidth)
        actual, _ = compose_objective_gradients(task, rbf.gradients, 0.01)

        sft_sum, dpp_sum = torch.tensor(0.0), torch.tensor(0.0)
        for batch, probe in zip(batches, probes, strict=True):
            views = shifted_token_views(batch)
            supported = []
            for name in names:
                set_active_adapter(model, name)
                logits = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    use_cache=False,
                ).logits
                response_logits = logits[
                    views["response_batch_indices"], views["response_hidden_indices"]
                ]
                sft_sum = sft_sum + F.cross_entropy(
                    response_logits.float(), views["response_targets"], reduction="sum"
                )
                reasoning_logits = logits[
                    views["reasoning_batch_indices"], views["reasoning_hidden_indices"]
                ]
                supported.append(reasoning_logits.gather(1, probe.support_ids))
            features = normalized_support_features(torch.stack(supported), probe.support_mask)
            dpp, _ = step_dpp_loss(
                features,
                views["reasoning_batch_indices"],
                views["reasoning_step_ids"],
                jitter=1e-4,
                maximum_jitter=1e-2,
                reduction="sum",
            )
            dpp_sum = dpp_sum + dpp
        set_all_adapters_trainable(model, names)
        pair_distances = []
        for left in range(3):
            for right in range(left + 1, 3):
                modules = []
                for key in groups[left]:
                    if "lora_A" in key:
                        bkey = key.replace("lora_A", "lora_B")
                        difference = (
                            groups[left][bkey] @ groups[left][key]
                            - groups[right][bkey] @ groups[right][key]
                        )
                        modules.append(difference.square().mean())
                pair_distances.append(torch.stack(modules).mean())
        dense_rbf = torch.exp(-torch.stack(pair_distances) / fixed_bandwidth).mean()
        loss = sft_sum / total_tokens + 0.2 * dpp_sum / len(batches) + 0.01 * dense_rbf
        expected = torch.autograd.grad(loss, [value for current in parameters for value in current])
        for full, split in zip(
            expected, [value for current in actual for value in current], strict=True
        ):
            torch.testing.assert_close(split, full, atol=3e-5, rtol=5e-4)

    def test_gradients_are_local_and_large_rbf_gradient_is_not_capped(self) -> None:
        task = [[torch.tensor([1.0, 2.0])], [torch.tensor([3.0, 4.0])]]
        rbf = [[torch.tensor([1000.0, -2000.0])], [torch.tensor([-5.0, 7.0])]]
        final, diagnostics = compose_objective_gradients(task, rbf, 0.01)
        torch.testing.assert_close(final[0][0], torch.tensor([11.0, -18.0]))
        torch.testing.assert_close(final[1][0], torch.tensor([2.95, 4.07]))
        self.assertAlmostEqual(diagnostics.rbf_task_ratios[0], 10.0, places=5)
        # Altering another expert's task never changes expert zero's update.
        changed, _ = compose_objective_gradients(
            [task[0], [torch.tensor([-999.0, 88.0])]], rbf, 0.01
        )
        torch.testing.assert_close(changed[0][0], final[0][0], atol=0, rtol=0)
        norm = global_clip_grad_list_(final[0], 1.0)
        self.assertGreater(norm, 1.0)
        self.assertAlmostEqual(float(final[0][0].norm()), 1.0, places=6)

    def test_zero_weight_is_exactly_independent_task_gradient(self) -> None:
        task = [[torch.tensor([1.0])], [torch.tensor([3.0])]]
        final, diagnostics = compose_objective_gradients(
            task, [[torch.tensor([40.0])], [torch.tensor([-40.0])]], 0.0
        )
        for expected, actual in zip(task, final, strict=True):
            torch.testing.assert_close(expected[0], actual[0], atol=0, rtol=0)
        self.assertEqual(diagnostics.weighted_rbf_norms, (0.0, 0.0))

    def test_rejects_invalid_weight_and_mismatched_parameter_shapes(self) -> None:
        for weight in [-1.0, float("nan"), float("inf")]:
            with self.subTest(weight=weight), self.assertRaisesRegex(ValueError, "finite"):
                compose_objective_gradients([[torch.ones(1)]], [[torch.ones(1)]], weight)
        with self.assertRaisesRegex(ValueError, "matching shapes"):
            compose_objective_gradients([[torch.ones(1)]], [[torch.ones(2)]], 0.01)

    def test_resume_fingerprint_always_encodes_the_scalar_objective(self) -> None:
        settings = _run_compatible_config({"stage1": {"forward_mode": "one_pass"}})
        self.assertEqual(settings["stage1"]["objective"], STAGE1_METHOD)


if __name__ == "__main__":
    unittest.main()
