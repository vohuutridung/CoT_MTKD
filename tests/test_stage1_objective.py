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
from cot_mtkd.stage1.gac_gradient import STAGE1_METHOD, stable_gac_gradients
from cot_mtkd.stage1.rbf import rbf_kernel, rbf_repulsion_gradients
from cot_mtkd.stage1.trainer import _run_compatible_config, one_pass_expert_gradients


class Stage1ObjectiveTest(unittest.TestCase):
    def test_accumulated_task_gradients_and_gac_match_dense_reference(self) -> None:
        """Compare split VJPs with dense SFT/DPP backward, then verify GAC and RBF sign."""
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
                combined_dpp_scale=0.1 * total_tokens / len(batches),
            )
            probes.append(probe)
            for destination, result in zip(task, results, strict=True):
                for value, gradient in zip(destination, result[0], strict=True):
                    value.add_(gradient / total_tokens)
        set_all_adapters_trainable(model, names)
        fixed_bandwidth = 0.01
        rbf = rbf_repulsion_gradients(groups, 1.0, fixed_bandwidth)
        repulsion = [[-value for value in current] for current in rbf.gradients]
        actual, _ = stable_gac_gradients(
            task, repulsion, rbf_kernel(rbf.distances, 0.5 * fixed_bandwidth),
            beta=0.5, rbf_weight=0.5,
        )

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
        task_loss = sft_sum / total_tokens + 0.1 * dpp_sum / len(batches)
        flat_parameters = [value for current in parameters for value in current]
        dense_task = torch.autograd.grad(task_loss, flat_parameters)
        dense_rbf_grad = torch.autograd.grad(dense_rbf, flat_parameters)
        expected_task = torch.stack([
            torch.cat([value.flatten() for value in dense_task[index * len(parameters[0]):(index + 1) * len(parameters[0])]])
            for index in range(3)
        ])
        split_task = torch.stack([torch.cat([value.flatten() for value in current]) for current in task])
        torch.testing.assert_close(split_task, expected_task, atol=3e-5, rtol=5e-4)
        dense_regularizer = torch.stack([
            torch.cat([value.flatten() for value in dense_rbf_grad[index * len(parameters[0]):(index + 1) * len(parameters[0])]])
            for index in range(3)
        ])
        kernel = torch.eye(3)
        indices = torch.triu_indices(3, 3, 1)
        kernel[indices[0], indices[1]] = torch.exp(-torch.stack(pair_distances).detach() / (0.5 * fixed_bandwidth))
        kernel[indices[1], indices[0]] = kernel[indices[0], indices[1]]
        kernel.fill_diagonal_(0.0)
        cross = 0.25 * kernel
        expected_mixed = (1 - cross.sum(dim=0))[:, None] * expected_task + cross.T @ expected_task
        caps = (expected_mixed.norm(dim=1) / (dense_regularizer.norm(dim=1) + 1e-12)).clamp(max=1)
        expected = expected_mixed + 0.5 * caps[:, None] * dense_regularizer
        actual_flat = torch.stack([torch.cat([value.flatten() for value in current]) for current in actual])
        torch.testing.assert_close(actual_flat, expected, atol=3e-5, rtol=5e-4)

    def test_resume_fingerprint_encodes_gac_and_differs_from_no_gac(self) -> None:
        from cot_mtkd.utils.manifest import fingerprint
        settings = _run_compatible_config({"stage1": {"forward_mode": "one_pass"}})
        self.assertEqual(settings["stage1"]["objective"], STAGE1_METHOD)
        for previous_method in ("sft_dpp_rbf", "sft_dpp_rbf_gac"):
            previous = {"stage1": {"objective": previous_method}}
            self.assertNotEqual(fingerprint(settings), fingerprint(previous))


if __name__ == "__main__":
    unittest.main()
