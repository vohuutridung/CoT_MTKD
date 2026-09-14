from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from cot_mtkd.models.multi_adapter import (
    adapter_parameter_groups,
    create_multi_adapter_model,
    extract_adapter_state,
    load_adapter_bundle,
    save_adapter_bundle,
    set_active_adapter,
)


def tiny_qwen() -> Qwen2ForCausalLM:
    configuration = Qwen2Config(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=64,
    )
    return Qwen2ForCausalLM(configuration)


LORA = {
    "rank": 2,
    "alpha": 2,
    "dropout": 0.05,
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


def build(seed: int):
    base = tiny_qwen()
    with patch("cot_mtkd.models.multi_adapter.load_base_causal_lm", return_value=base):
        return create_multi_adapter_model({}, LORA, 2, torch.device("cpu"), seed)


class MultiAdapterTest(unittest.TestCase):
    def test_all_adapter_parameters_follow_bfloat16_base_dtype(self) -> None:
        base = tiny_qwen().to(torch.bfloat16)
        with patch(
            "cot_mtkd.models.multi_adapter.load_base_causal_lm", return_value=base
        ):
            model, names = create_multi_adapter_model(
                {}, LORA, 2, torch.device("cpu"), 42
            )
        for name in names:
            self.assertTrue(
                all(
                    parameter.dtype == torch.bfloat16
                    for parameter in adapter_parameter_groups(model, [name])[0].values()
                )
            )

    def test_per_expert_seed_is_explicit_and_reproducible(self) -> None:
        torch.manual_seed(1)
        first, names = build(42)
        torch.manual_seed(999)
        second, second_names = build(42)
        self.assertEqual(names, second_names)
        first_states = [extract_adapter_state(first, name) for name in names]
        second_states = [extract_adapter_state(second, name) for name in names]
        for left, right in zip(first_states, second_states, strict=True):
            for key in left:
                self.assertTrue(torch.equal(left[key], right[key]))
        a_keys = [key for key in first_states[0] if "lora_A" in key]
        b_keys = [key for key in first_states[0] if "lora_B" in key]
        self.assertTrue(
            any(
                not torch.equal(first_states[0][key], first_states[1][key])
                for key in a_keys
            )
        )
        self.assertTrue(
            all(torch.count_nonzero(first_states[0][key]) == 0 for key in b_keys)
        )

    def test_parameter_alignment_activation_and_bundle_round_trip(self) -> None:
        model, names = build(42)
        groups = adapter_parameter_groups(model, names)
        self.assertEqual(list(groups[0]), list(groups[1]))
        set_active_adapter(model, names[1])
        trainable = [
            name for name, value in model.named_parameters() if value.requires_grad
        ]
        self.assertTrue(trainable)
        self.assertTrue(all(f".{names[1]}." in name for name in trainable))
        with tempfile.TemporaryDirectory() as directory:
            bundle_path = save_adapter_bundle(model, names, directory)
            restored = load_adapter_bundle(bundle_path)
            self.assertEqual(set(restored), set(names))
            for name in names:
                self.assertTrue(
                    Path(directory, "adapters", name, "adapter_config.json").is_file()
                )


if __name__ == "__main__":
    unittest.main()
