from __future__ import annotations

import unittest
from unittest.mock import patch

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from cot_mtkd.data.schema import PreparedRecord, TokenRegion
from cot_mtkd.models.multi_adapter import (
    adapter_parameter_map,
    create_multi_adapter_model,
)
from cot_mtkd.stage2.cache_builder import compile_sample_target


class CacheBuilderTest(unittest.TestCase):
    def test_compiles_mass_preserving_target_and_dense_sparse_audit(self) -> None:
        base = Qwen2ForCausalLM(
            Qwen2Config(
                vocab_size=64,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=1,
                max_position_embeddings=64,
            )
        )
        lora = {
            "rank": 2,
            "alpha": 2,
            "dropout": 0.0,
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
        with patch(
            "cot_mtkd.models.multi_adapter.load_base_causal_lm",
            return_value=base,
        ):
            model, names = create_multi_adapter_model(
                {}, lora, 2, torch.device("cpu"), 42
            )
        with torch.no_grad():
            for key, parameter in adapter_parameter_map(model, names[1]).items():
                if "lora_B" in key:
                    parameter.fill_(0.05)

        regions = [
            int(TokenRegion.PROMPT),
            int(TokenRegion.PROMPT),
            int(TokenRegion.ASSISTANT_CONTROL),
            int(TokenRegion.REASONING),
            int(TokenRegion.DELIMITER),
            int(TokenRegion.ANSWER_MARKER),
            int(TokenRegion.ANSWER),
        ]
        record = PreparedRecord(
            sample_id="tiny",
            input_ids=[1, 2, 3, 4, 5, 6, 7],
            labels=[-100, -100, 3, 4, 5, 6, 7],
            attention_mask=[1] * 7,
            offset_mapping=[(index, index + 1) for index in range(7)],
            region_ids=regions,
            step_ids=[-1, -1, -1, 0, 0, -1, -1],
            question="q",
            thinking="r",
            attempt="a",
            solution="a",
            deepseek_grade="Yes",
            original_length=7,
            kept_length=7,
            original_steps=1,
            kept_steps=1,
            truncated=False,
            answer_start=5,
            reasoning_start=3,
            tokenizer_fingerprint="tiny",
        )
        positions, ids, probabilities, tail, audit = compile_sample_target(
            model,
            names,
            record,
            {"teacher_weights": [[0.25, 0.75]], "answer_weights": [0.6, 0.4]},
            torch.device("cpu"),
            top_k=8,
            temperature=2.0,
            chunk_tokens=2,
            storage_dtype=torch.float16,
            epsilon=1.0e-8,
            audit_dense_kl=True,
        )

        self.assertEqual(positions.tolist(), [2, 3, 4, 5, 6])
        self.assertEqual(ids.shape, (5, 8))
        self.assertEqual(probabilities.shape, ids.shape)
        self.assertEqual(tail.shape, (5,))
        self.assertTrue(torch.allclose(probabilities.sum(dim=-1) + tail, torch.ones(5)))
        self.assertEqual(int(audit[0].item()), 5)
        self.assertEqual(int(audit[2].item()), 10)
        self.assertGreaterEqual(float(audit[3].item()), float(audit[4].item()) - 1e-4)
        self.assertEqual(int(audit[6].item()), 0)


if __name__ == "__main__":
    unittest.main()
