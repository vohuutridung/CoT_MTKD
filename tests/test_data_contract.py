from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

from cot_mtkd.data.dataset import JsonlRecordDataset
from cot_mtkd.data.prepare import prepare_one, write_prepared_dataset
from cot_mtkd.data.schema import CharacterSegment, PreparedRecord, TokenRegion
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

    def test_cot_final_answer_is_reasoning_and_no_separate_answer_is_appended(self) -> None:
        serialized = serialize_record(
            "Question?",
            "First step.\n\nFinal Answer: \\boxed{7}",
        )
        reasoning = [
            segment for segment in serialized.segments if segment.region == TokenRegion.REASONING
        ]
        delimiters = [
            segment for segment in serialized.segments if segment.region == TokenRegion.DELIMITER
        ]
        answers = [
            segment for segment in serialized.segments if segment.region == TokenRegion.ANSWER
        ]
        self.assertEqual(len(reasoning), 2)
        self.assertEqual(delimiters[0].step_id, 0)
        self.assertEqual(answers, [])
        self.assertFalse(any(s.region == TokenRegion.ANSWER_MARKER for s in serialized.segments))
        self.assertTrue(serialized.text.endswith("Final Answer: \\boxed{7}\n<|im_end|>"))
        self.assertIn("Final Answer", serialized.text[reasoning[1].start : reasoning[1].end])

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
        self.assertNotIn("attempt", record.to_dict())
        self.assertNotIn(int(TokenRegion.ANSWER), record.region_ids)
        self.assertNotIn(int(TokenRegion.ANSWER_MARKER), record.region_ids)
        self.assertEqual(record.region_ids[record.answer_start], int(TokenRegion.EOS))
        delimiter_steps = [
            step
            for step, region in zip(record.step_ids, record.region_ids, strict=True)
            if region == int(TokenRegion.DELIMITER)
        ]
        self.assertTrue(delimiter_steps)
        self.assertEqual(delimiter_steps[0], 0)

    def test_attempt_is_never_read_and_does_not_affect_any_prepared_field(self) -> None:
        raw = {"question": "Q", "deepseek_thinking_trajectory": "Answer: 7", "solution": "7"}

        def prepare(value):
            return prepare_one(value, CharacterTokenizer(), 0, 1000, "system", r"\n\n+", "test")

        expected = prepare(raw).to_dict()

        class UnreadableAttempt:
            def __str__(self):
                raise AssertionError("attempt must never be accessed")

        self.assertEqual(
            prepare(dict(raw, deepseek_attempt=UnreadableAttempt())).to_dict(), expected
        )

    def test_oversized_cot_is_rejected_instead_of_losing_its_final_answer(self) -> None:
        with self.assertRaisesRegex(ValueError, "refusing to truncate its final answer"):
            prepare_one(
                {
                    "question": "Q",
                    "deepseek_thinking_trajectory": "step\n\nAnswer: 7",
                    "solution": "7",
                },
                CharacterTokenizer(),
                0,
                20,
                "system",
                r"\n\n+",
                "test",
            )

    def test_old_attempt_containing_artifact_requires_repreparation(self) -> None:
        with self.assertRaisesRegex(ValueError, "rerun prepare"):
            PreparedRecord.from_dict({"attempt": "obsolete"})

    def test_filter_retains_996_original_ids_and_records_provenance(self) -> None:
        excluded = ["s1k-0135", "s1k-0324", "s1k-0392", "s1k-0438"]
        with tempfile.TemporaryDirectory() as directory:
            config = {
                "_project_root": directory,
                "output_dir": directory,
                "dataset": {
                    "keep_all_grades": True,
                    "expected_records": 1000,
                    "expected_prepared_records": 996,
                    "exclude_sample_ids": excluded,
                },
                "tokenization": {
                    "padding_side": "right",
                    "packing": False,
                    "add_special_tokens": False,
                    "truncation": "reject_overlength",
                    "max_length": 1000,
                },
                "serialization": {
                    "response_field": "deepseek_thinking_trajectory",
                    "system_prompt": "system",
                    "step_pattern": r"\n\n+",
                },
            }
            rows = [
                {
                    "question": "Q",
                    "deepseek_thinking_trajectory": "step\n\nAnswer: 7",
                    "solution": "7",
                    "deepseek_attempt": "NEVER TRAIN ON THIS",
                }
                for _ in range(1000)
            ]
            manifest = write_prepared_dataset(rows, CharacterTokenizer(), config)
            records = JsonlRecordDataset(directory)
            ids = [record.sample_id for record in records]
            self.assertEqual(len(records), 996)
            self.assertEqual(
                ids, [f"s1k-{i:04d}" for i in range(1000) if f"s1k-{i:04d}" not in excluded]
            )
            self.assertEqual(manifest["source_records"], 1000)
            self.assertEqual(manifest["excluded_sample_ids"], excluded)
            self.assertEqual(manifest["training_target"], "deepseek_thinking_trajectory")
            contents = (Path(directory) / "data.jsonl").read_text()
            self.assertNotIn("NEVER TRAIN ON THIS", contents)
            self.assertNotIn('"attempt"', contents)
            config["output_dir"] = str(Path(directory) / "missing-source")
            config["dataset"]["exclude_sample_ids"].append("absent-id")
            with self.assertRaisesRegex(RuntimeError, "missing from source"):
                write_prepared_dataset(rows, CharacterTokenizer(), config)

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
