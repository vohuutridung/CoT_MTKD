from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from cot_mtkd.stage2.step_logging import (
    create_reasoning_logger,
    log_record_steps,
    reasoning_log_filename,
    trim_jsonl_to_checkpoint,
)


class Stage2StepLoggingTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.output_dir = Path(self.directory.name) / "output"
        self.output_dir.mkdir()
        self.run = "test-output-space-run"

    def write_rows(self, path, rows, trailing=""):
        contents = "".join(json.dumps(row) + "\n" for row in rows) + trailing
        path.write_text(contents, encoding="utf-8")
        return contents

    def read_rows(self, path):
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    def row(self, cursor, before_update=True, **values):
        key = "data_step_before" if before_update else "data_step"
        return {"run_fingerprint": self.run, key: cursor, **values}

    def test_single_rank_and_distributed_files_are_distinct(self):
        self.assertEqual(reasoning_log_filename(0, 1), "reasoning_steps.jsonl")
        self.assertEqual(reasoning_log_filename(0, 16), "reasoning_steps.rank00000.jsonl")
        self.assertEqual(reasoning_log_filename(12, 16), "reasoning_steps.rank00012.jsonl")
        first = create_reasoning_logger(self.output_dir, 0, 2, self.run, 0, False)
        second = create_reasoning_logger(self.output_dir, 1, 2, self.run, 0, False)
        first.log("rank-local", rank=0)
        second.log("rank-local", rank=1)
        self.assertNotEqual(first.path, second.path)
        self.assertEqual([row["rank"] for row in self.read_rows(first.path)], [0])
        self.assertEqual([row["rank"] for row in self.read_rows(second.path)], [1])

    def test_fresh_run_truncates_existing_file_and_creates_missing_directories(self):
        path = self.output_dir / "reasoning_steps.jsonl"
        path.write_text("stale content from an earlier run\n", encoding="utf-8")
        logger = create_reasoning_logger(self.output_dir, 0, 1, self.run, 0, False)
        self.assertEqual(path.read_text(encoding="utf-8"), "")
        logger.log("fresh-run", run_fingerprint=self.run)
        self.assertEqual(len(self.read_rows(path)), 1)
        nested = self.output_dir / "new" / "nested"
        nested_logger = create_reasoning_logger(nested, 0, 1, self.run, 0, False)
        self.assertTrue(nested_logger.path.is_file())
        self.assertEqual(nested_logger.path.read_text(encoding="utf-8"), "")

    def test_each_step_keeps_original_id_values_and_sample_training_metadata(self):
        logger = create_reasoning_logger(self.output_dir, 1, 2, self.run, 0, False)
        step_values = [
            {
                "step_index": 0,
                "step_id": 3,
                "n_tokens": 2,
                "token_start": 7,
                "token_end": 9,
                "token_js_mean": 0.1,
                "step_disagreement": 0.2,
                "disagreement_pooling_power": 4.0,
                "tau": 0.3,
                "tau_quantile": 0.75,
                "js_units": "nats",
                "rho": 0.2 / 0.5,
                "step_kd_loss": 0.7,
                "js_temperature": 1.0,
                "kd_temperature": 1.0,
                "teacher_count": 3,
            },
            {
                "step_index": 1,
                "step_id": 8,
                "n_tokens": 4,
                "token_start": 10,
                "token_end": 14,
                "token_js_mean": 0.15,
                "step_disagreement": 0.4,
                "disagreement_pooling_power": 4.0,
                "tau": 0.3,
                "tau_quantile": 0.75,
                "js_units": "nats",
                "rho": 0.4 / 0.7,
                "step_kd_loss": 0.9,
                "js_temperature": 1.0,
                "kd_temperature": 1.0,
                "teacher_count": 3,
            },
        ]
        result = SimpleNamespace(
            loss=0.8,
            steps=2,
            discarded_steps=1,
            step_metrics=step_values,
            metrics={"kd_loss": 0.8, "sft_loss": 0.0},
        )
        metadata = {
            "run_fingerprint": self.run,
            "rank": 1,
            "epoch": 2,
            "batch_in_epoch": 19,
            "sample_in_batch": 3,
            "data_step_before": 7,
            "global_step_before": 5,
        }
        log_record_steps(logger, result, "sample-đặc-biệt", **metadata)
        rows = self.read_rows(logger.path)
        self.assertEqual(len(rows), 2)
        for row, expected_step in zip(rows, step_values, strict=True):
            self.assertEqual(row["event"], "stage2_reasoning_step")
            self.assertIsInstance(row["time"], (int, float))
            self.assertTrue(math.isfinite(row["time"]))
            self.assertEqual(row["sample_id"], "sample-đặc-biệt")
            self.assertEqual(row["sample_kd_loss"], 0.8)
            self.assertEqual(row["retained_steps"], 2)
            self.assertEqual(row["discarded_steps"], 1)
            for key, value in {**metadata, **expected_step}.items():
                self.assertEqual(row[key], value)
            self.assertEqual(
                set(row),
                {
                    "event",
                    "time",
                    "sample_id",
                    "sample_kd_loss",
                    "sample_sft_loss",
                    "sample_total_loss",
                    "retained_steps",
                    "discarded_steps",
                    *metadata.keys(),
                    *expected_step.keys(),
                },
            )
        self.assertEqual([row["step_id"] for row in rows], [3, 8])
        empty = SimpleNamespace(loss=0.0, steps=0, discarded_steps=3, step_metrics=[])
        log_record_steps(logger, empty, "sample-with-no-retained-steps", **metadata)
        self.assertEqual(self.read_rows(logger.path), rows)

    def test_resume_step_logs_keep_only_windows_before_checkpoint_and_append(self):
        path = self.output_dir / reasoning_log_filename(0, 1)
        rows = [self.row(cursor, global_step_before=1) for cursor in (0, 1, 2, 3, 4)]
        self.write_rows(path, rows)
        logger = create_reasoning_logger(self.output_dir, 0, 1, self.run, 3, True)
        self.assertEqual(self.read_rows(path), rows[:3])
        logger.log("stage2_reasoning_step", **self.row(3, global_step_before=1))
        self.assertEqual([row["data_step_before"] for row in self.read_rows(path)], [0, 1, 2, 3])
        self.assertFalse(path.with_suffix(".jsonl.recovering").exists())

    def test_resume_aggregate_metrics_keep_checkpoint_window_inclusively(self):
        path = self.output_dir / "metrics.jsonl"
        rows = [self.row(cursor, before_update=False, step=1) for cursor in (1, 2, 3, 4)]
        self.write_rows(path, rows)
        trim_jsonl_to_checkpoint(path, self.run, 3, before_update=False)
        self.assertEqual(self.read_rows(path), rows[:3])
        self.assertFalse(path.with_suffix(".jsonl.recovering").exists())

    def test_zero_checkpoint_cursor_removes_preupdate_rows_but_keeps_metric_zero(self):
        for before_update in (True, False):
            with self.subTest(before_update=before_update):
                path = self.output_dir / f"zero-{before_update}.jsonl"
                rows = [self.row(0, before_update), self.row(1, before_update)]
                self.write_rows(path, rows)
                trim_jsonl_to_checkpoint(path, self.run, 0, before_update=before_update)
                self.assertEqual(self.read_rows(path), [] if before_update else rows[:1])

    def test_resume_discards_interrupted_trailing_write_without_losing_complete_rows(self):
        path = self.output_dir / "partial.jsonl"
        rows = [self.row(0), self.row(1), self.row(2)]
        self.write_rows(path, rows, trailing='{"run_fingerprint": "test-output-space-run",')
        trim_jsonl_to_checkpoint(path, self.run, 2, before_update=True)
        self.assertEqual(self.read_rows(path), rows[:2])
        self.assertTrue(path.read_text(encoding="utf-8").endswith("\n"))
        self.assertFalse(path.with_suffix(".jsonl.recovering").exists())

    def test_missing_file_needs_no_recovery_and_resume_can_start_new_log(self):
        path = self.output_dir / "missing.jsonl"
        trim_jsonl_to_checkpoint(path, self.run, 3, before_update=True)
        self.assertFalse(path.exists())
        self.assertFalse(path.with_suffix(".jsonl.recovering").exists())
        logger = create_reasoning_logger(self.output_dir, 0, 1, self.run, 3, True)
        logger.log("resumed-run", **self.row(3))
        self.assertEqual(self.read_rows(logger.path)[0]["data_step_before"], 3)

    def test_corrupt_complete_rows_fail_without_modifying_original_and_clean_temporary(self):
        invalid_lines = [
            "{broken}\n",
            "[]\n",
            "null\n",
            json.dumps({"run_fingerprint": self.run}) + "\n",
            json.dumps({"data_step_before": 0}) + "\n",
        ]
        invalid_lines += [
            json.dumps(self.row(cursor)) + "\n" for cursor in (-1, True, 1.0, "1", None)
        ]
        for index, invalid in enumerate(invalid_lines):
            with self.subTest(invalid=invalid):
                path = self.output_dir / f"corrupt-{index}.jsonl"
                original = self.write_rows(path, [self.row(0)], trailing=invalid)
                with self.assertRaisesRegex(RuntimeError, "Invalid log record.*:2"):
                    trim_jsonl_to_checkpoint(path, self.run, 2, before_update=True)
                self.assertEqual(path.read_text(encoding="utf-8"), original)
                self.assertFalse(path.with_suffix(".jsonl.recovering").exists())

    def test_foreign_run_is_rejected_even_beyond_cursor_and_original_is_preserved(self):
        path = self.output_dir / "foreign.jsonl"
        rows = [self.row(0), {**self.row(100), "run_fingerprint": "another-run"}]
        original = self.write_rows(path, rows)
        with self.assertRaisesRegex(RuntimeError, "another run"):
            trim_jsonl_to_checkpoint(path, self.run, 1, before_update=True)
        self.assertEqual(path.read_text(encoding="utf-8"), original)
        self.assertFalse(path.with_suffix(".jsonl.recovering").exists())


if __name__ == "__main__":
    unittest.main()
