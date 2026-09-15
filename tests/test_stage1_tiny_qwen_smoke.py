from __future__ import annotations

import unittest

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import Qwen2Config, Qwen2ForCausalLM

from cot_mtkd.data.schema import TokenRegion
from cot_mtkd.models.multi_adapter import (
    adapter_parameter_groups,
    set_active_adapter,
    set_all_adapters_trainable,
)
from cot_mtkd.stage1.gac_gradient import apply_grassmann_force_
from cot_mtkd.stage1.grassmann import grassmann_repulsion_updates
from cot_mtkd.stage1.trainer import probe_stage1_dpp, replay_expert_gradients


class Stage1TinyQwenSmokeTest(unittest.TestCase):
    def test_outward_force_applies_literal_post_adamw_increment(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = torch.optim.AdamW([parameter], lr=0.1, weight_decay=0.0)
        parameter.grad = torch.tensor([1.0])
        optimizer.step()
        data_only_result = parameter.detach().clone()
        apply_grassmann_force_(
            [parameter], [torch.tensor([2.0])],
            learning_rate=0.1, repulsion_weight=0.5,
        )
        self.assertTrue(torch.allclose(parameter, data_only_result + 0.1))

    def test_probe_replay_and_zero_b_grassmann_force(self) -> None:
        torch.manual_seed(13)
        base = Qwen2ForCausalLM(
            Qwen2Config(
                vocab_size=40,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=1,
                num_attention_heads=4,
                num_key_value_heads=2,
                max_position_embeddings=64,
                attention_dropout=0.0,
                use_cache=False,
            )
        )
        lora = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=2,
            lora_alpha=4,
            lora_dropout=0.0,
            target_modules=["q_proj", "v_proj"],
            bias="none",
            inference_mode=False,
        )
        model = get_peft_model(
            base, lora, adapter_name="expert_0", autocast_adapter_dtype=False
        )
        model.add_adapter("expert_1", lora)
        names = ["expert_0", "expert_1"]
        set_active_adapter(model, names[0])
        groups = adapter_parameter_groups(model, names)
        for group in groups:
            for key, parameter in group.items():
                if "lora_B" in key:
                    self.assertTrue(torch.count_nonzero(parameter) == 0)

        input_ids = torch.randint(3, 40, (2, 9))
        labels = input_ids.clone()
        labels[:, 0] = -100
        regions = [
            TokenRegion.PROMPT,
            TokenRegion.REASONING,
            TokenRegion.REASONING,
            TokenRegion.DELIMITER,
            TokenRegion.REASONING,
            TokenRegion.REASONING,
            TokenRegion.ANSWER_MARKER,
            TokenRegion.ANSWER,
            TokenRegion.EOS,
        ]
        batch = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": torch.ones_like(input_ids),
            "region_ids": torch.tensor([[int(region) for region in regions]] * 2),
            "step_ids": torch.tensor([[-1, 0, 0, 0, 1, 1, -1, -1, -1]] * 2),
        }
        config = {
            "runtime": {
                "probe_hidden_device": "cpu",
                "lm_head_chunk_tokens": 3,
                "council_chunk_tokens": 2,
            },
            "dpp": {"jitter": 1.0e-5, "max_jitter": 1.0e-2},
        }
        device = torch.device("cpu")
        probe = probe_stage1_dpp(
            model, names, batch, global_step=0, rng_stream=0, base_seed=13,
            config=config, device=device,
        )
        self.assertEqual(probe.support_ids.shape[0], 8)
        self.assertEqual(probe.support_mask.shape, probe.support_ids.shape)
        self.assertEqual(probe.dpp_logit_gradients.shape[0], 2)
        self.assertEqual(probe.dpp_logit_gradients.shape[1], 8)
        self.assertEqual(probe.valid_candidate_count, 8)
        self.assertTrue(torch.isfinite(probe.dpp_logit_gradients).all())
        self.assertGreater(probe.dpp_logit_gradients.abs().sum().item(), 0.0)

        captured_data_gradients = []
        for expert, name in enumerate(names):
            sft, dpp, loss, segment_count, _dropped = replay_expert_gradients(
                model, name, expert, list(groups[expert].values()), batch, probe,
                global_step=0, rng_stream=0, base_seed=13, drop_probability=0.3,
                chunk_tokens=3, device=device,
            )
            self.assertEqual(len(sft), len(groups[expert]))
            self.assertEqual(len(dpp), len(groups[expert]))
            self.assertEqual(segment_count, 8)
            self.assertGreater(loss, 0.0)
            self.assertTrue(all(torch.isfinite(grad).all() for grad in sft + dpp))
            captured_data_gradients.append(
                [local + 0.2 * diversity for local, diversity in zip(sft, dpp, strict=True)]
            )

        set_all_adapters_trainable(model, names)
        forces, kernel, distances, bandwidth = grassmann_repulsion_updates(groups)
        self.assertEqual(kernel.shape, (2, 2))
        self.assertEqual(distances.shape, (2, 2))
        self.assertGreater(bandwidth, 0.0)
        self.assertTrue(torch.isfinite(kernel).all())
        self.assertTrue(torch.isfinite(distances).all())
        self.assertTrue(all(torch.isfinite(force).all() for expert in forces for force in expert))

        parameters = list(groups[0].values())
        optimizer = torch.optim.AdamW(parameters, lr=1.0e-3, weight_decay=0.0)
        for parameter, gradient in zip(parameters, captured_data_gradients[0], strict=True):
            parameter.grad = gradient.clone()
        optimizer.step()
        data_only_result = [parameter.detach().clone() for parameter in parameters]
        apply_grassmann_force_(
            parameters, forces[0], learning_rate=1.0e-3, repulsion_weight=0.5
        )
        for parameter, data_only, force in zip(
            parameters, data_only_result, forces[0], strict=True
        ):
            self.assertTrue(torch.allclose(parameter, data_only + 5.0e-4 * force))


if __name__ == "__main__":
    unittest.main()
