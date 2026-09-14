from __future__ import annotations

import unittest

from cot_mtkd.data.prepare import prepare_one
from cot_mtkd.data.schema import CharacterSegment, TokenRegion
from cot_mtkd.data.serialize import serialize_record
from cot_mtkd.data.token_spans import assign_token_regions, complete_step_truncate


class CharacterTokenizer:
    name_or_path = "character-tokenizer"
    special_tokens_map = {}

    def __len__(self) -> int:
        return 256

    def get_added_vocab(self):
        return {}

    def __call__(self, text, **kwargs):
        return {
            "input_ids": [ord(character) % 256 for character in text],
            "offset_mapping": [(index, index + 1) for index in range(len(text))],
        }


class DataContractTest(unittest.TestCase):
    def test_zero_offset_special_token_uses_structural_gap(self) -> None:
        segments = (
            CharacterSegment(0, 5, TokenRegion.PROMPT),
            CharacterSegment(5, 10, TokenRegion.ASSISTANT_CONTROL),
            CharacterSegment(10, 12, TokenRegion.REASONING, 0),
        )
        regions, steps = assign_token_regions([(0, 5), (0, 0), (10, 12)], segments)
        self.assertEqual(
            regions,
            [
                int(TokenRegion.PROMPT),
                int(TokenRegion.ASSISTANT_CONTROL),
                int(TokenRegion.REASONING),
            ],
        )
        self.assertEqual(steps, [-1, -1, 0])

    def test_structural_answer_boundary_and_delimiter_ownership(self) -> None:
        serialized = serialize_record(
            "Question?",
            "First step.\n\nFinal Answer: \\boxed{7}",
            "7",
        )
        reasoning = [
            segment
            for segment in serialized.segments
            if segment.region == TokenRegion.REASONING
        ]
        delimiters = [
            segment
            for segment in serialized.segments
            if segment.region == TokenRegion.DELIMITER
        ]
        answers = [
            segment
            for segment in serialized.segments
            if segment.region == TokenRegion.ANSWER
        ]
        self.assertEqual(len(reasoning), 2)
        self.assertEqual(delimiters[0].step_id, 0)
        self.assertEqual(serialized.text[answers[0].start : answers[0].end], "7")
        self.assertIn(
            "Final Answer", serialized.text[reasoning[1].start : reasoning[1].end]
        )

    def test_prepared_masks(self) -> None:
        record = prepare_one(
            {
                "question": "Q",
                "deepseek_thinking_trajectory": "one\r\n\r\ntwo",
                "deepseek_attempt": "2",
                "solution": "2",
                "deepseek_grade": "Yes",
            },
            CharacterTokenizer(),
            sample_index=0,
            max_length=1000,
            system_prompt="system",
            step_pattern=r"\r?\n[ \t]*\r?\n+",
            tokenizer_hash="test",
        )
        for label, region in zip(record.labels, record.region_ids, strict=True):
            if region == int(TokenRegion.PROMPT):
                self.assertEqual(label, -100)
            else:
                self.assertNotEqual(label, -100)
        self.assertEqual(len(record.offset_mapping), len(record.input_ids))
        self.assertEqual(record.thinking, "one\n\ntwo")
        delimiter_steps = [
            step
            for step, region in zip(record.step_ids, record.region_ids, strict=True)
            if region == int(TokenRegion.DELIMITER)
        ]
        self.assertTrue(delimiter_steps)
        self.assertEqual(delimiter_steps[0], 0)

    def test_complete_step_truncation_keeps_answer(self) -> None:
        ids = list(range(12))
        labels = [-100, -100] + list(range(2, 12))
        regions = [
            int(TokenRegion.PROMPT),
            int(TokenRegion.ASSISTANT_CONTROL),
            int(TokenRegion.REASONING),
            int(TokenRegion.DELIMITER),
            int(TokenRegion.REASONING),
            int(TokenRegion.DELIMITER),
            int(TokenRegion.REASONING),
            int(TokenRegion.DELIMITER),
            int(TokenRegion.ANSWER_MARKER),
            int(TokenRegion.ANSWER),
            int(TokenRegion.ANSWER),
            int(TokenRegion.EOS),
        ]
        steps = [-1, -1, 0, 0, 1, 1, 2, 2, -1, -1, -1, -1]
        kept_ids, _, _, kept_steps, stats = complete_step_truncate(
            ids, labels, regions, steps, max_length=8
        )
        self.assertEqual(kept_ids[-4:], [8, 9, 10, 11])
        self.assertEqual({step for step in kept_steps if step >= 0}, {0})
        self.assertTrue(stats["truncated"])


if __name__ == "__main__":
    unittest.main()
