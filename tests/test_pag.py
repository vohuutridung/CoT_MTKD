from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F
from transformers import Qwen2Config, Qwen2ForCausalLM

from cot_mtkd.data.schema import PreparedRecord, TokenRegion
from cot_mtkd.signals.pag import score_reference_solution_pag


class _SolutionTokenizer:
    def __call__(
        self,
        text: str,
        add_special_tokens: bool = False,
        return_offsets_mapping: bool = False,
    ):
        del add_special_tokens, return_offsets_mapping
        boundary = text.index("gold")
        return {
            "input_ids": [7, 9, 10],
            "offset_mapping": [
                (0, boundary),
                (boundary, boundary + 2),
                (boundary + 2, len(text)),
            ],
        }


class PAGTest(unittest.TestCase):
    def test_incremental_cache_matches_full_context_scoring(self) -> None:
        torch.manual_seed(31)
        model = Qwen2ForCausalLM(
            Qwen2Config(
                vocab_size=32,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=1,
                max_position_embeddings=64,
            )
        )
        prefix = [1, 2, 3]
        first_step = [4, 5]
        second_step = [6]
        answer_prefix = [7]
        attempted_answer = [8]
        input_ids = prefix + first_step + second_step + answer_prefix + attempted_answer
        regions = [
            int(TokenRegion.PROMPT),
            int(TokenRegion.PROMPT),
            int(TokenRegion.ASSISTANT_CONTROL),
            int(TokenRegion.REASONING),
            int(TokenRegion.DELIMITER),
            int(TokenRegion.REASONING),
            int(TokenRegion.ANSWER_MARKER),
            int(TokenRegion.ANSWER),
        ]
        steps = [-1, -1, -1, 0, 0, 1, -1, -1]
        record = PreparedRecord(
            sample_id="pag",
            input_ids=input_ids,
            labels=[-100, -100, 3, 4, 5, 6, 7, 8],
            attention_mask=[1] * len(input_ids),
            offset_mapping=[(index, index + 1) for index in range(len(input_ids))],
            region_ids=regions,
            step_ids=steps,
            question="q",
            thinking="r",
            attempt="a",
            solution="gold",
            deepseek_grade="Yes",
            original_length=len(input_ids),
            kept_length=len(input_ids),
            original_steps=2,
            kept_steps=2,
            truncated=False,
            answer_start=6,
            reasoning_start=3,
            tokenizer_fingerprint="toy",
        )
        actual = score_reference_solution_pag(
            model,
            record,
            _SolutionTokenizer(),
            torch.device("cpu"),
            chunk_tokens=1,
        )

        solution = [9, 10]
        nlls = []
        for context in (prefix, prefix + first_step, prefix + first_step + second_step):
            sequence = torch.tensor([context + answer_prefix + solution])
            hidden = model.model(
                sequence, use_cache=False, return_dict=True
            ).last_hidden_state
            start = len(context) + len(answer_prefix) - 1
            selected = hidden[0, start : start + len(solution)]
            logits = model.get_output_embeddings()(selected).float()
            nlls.append(
                F.cross_entropy(
                    logits, torch.tensor(solution), reduction="mean"
                ).detach()
            )
        expected = torch.stack(nlls[:-1]) - torch.stack(nlls[1:])
        self.assertTrue(torch.allclose(actual, expected, atol=1.0e-6, rtol=1.0e-5))


if __name__ == "__main__":
    unittest.main()
