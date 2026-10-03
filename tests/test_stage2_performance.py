from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from cot_mtkd.stage2.performance import TrainingPerformanceLogger, performance_log_filename


def sample_result():
    return SimpleNamespace(
        steps=2,
        discarded_steps=1,
        metrics={
            "prefix_tokens": 10, "reasoning_tokens": 5,
            "head_chunks": 3, "cached_steps": 1, "recomputed_steps": 1,
            "teacher_head_chunk_sweeps": 5,
        },
    )


def allocator_stats(allocated, reserved, peak_allocated, peak_reserved):
    return {
        "allocated_bytes.all.current": allocated,
        "reserved_bytes.all.current": reserved,
        "allocated_bytes.all.peak": peak_allocated,
        "reserved_bytes.all.peak": peak_reserved,
    }


class Stage2PerformanceTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.output = Path(self.directory.name)

    def logger(self, device="cpu", rank=0, world_size=1, data_step=0, resume=False):
        return TrainingPerformanceLogger(
            self.output, torch.device(device), rank, world_size, "run", "config",
            {"runtime": {"lm_head_chunk_tokens": 4096}}, data_step, resume,
        )

    def rows(self, name="performance.jsonl"):
        return [json.loads(line) for line in (self.output / name).read_text().splitlines()]

    def test_cpu_timing_and_metadata_never_call_cuda(self):
        with (
            patch("torch.cuda.synchronize", side_effect=AssertionError("No synchronization")),
            patch("torch.cuda.memory_stats", side_effect=AssertionError("No CUDA on CPU")),
            patch("torch.cuda.reset_peak_memory_stats", side_effect=AssertionError("No CUDA on CPU")),
            patch("torch.cuda.get_device_properties", side_effect=AssertionError("No CUDA on CPU")),
            patch("cot_mtkd.stage2.performance.time.perf_counter", side_effect=[10, 20, 24]),
        ):
            logger = self.logger()
            self.assertEqual(logger.begin_sample(), 20)
            logger.log_sample("sample", 2.5, sample_result(), data_step_before=0, epoch=0)
            summary = logger.log_update(1, 1, 3, global_examples=1, skipped_update=False)
        session, sample, update = self.rows()
        self.assertEqual(session["timing_mode"], "host_wall_no_cuda_sync")
        self.assertEqual(session["config_fingerprint"], "config")
        self.assertEqual(sample["sample_id"], "sample")
        self.assertEqual(sample["record_gradient_wall_seconds"], 2.5)
        self.assertEqual(sample["cached_steps"], 1)
        self.assertEqual(sample["recomputed_steps"], 1)
        self.assertEqual(summary["update_window_wall_seconds"], 4)
        self.assertEqual(summary["session_elapsed_wall_seconds"], 14)
        self.assertEqual(summary["local_prefix_tokens_per_second"], 2.5)
        self.assertIsNone(summary["peak_allocated_gib"])
        self.assertIsNone(summary["eta_remaining_wall_seconds_estimate"])
        self.assertEqual(update["data_step_before"], 0)
        self.assertEqual(update["data_step"], 1)

    def test_gpu_update_keeps_largest_sample_peak_and_optimizer_peak_without_sync(self):
        # Large first sample, smaller second sample, then larger optimizer peak.
        stats = [
            allocator_stats(2, 4, 2, 4),
            allocator_stats(3, 8, 10, 12),
            allocator_stats(2, 8, 10, 12),
            allocator_stats(3, 8, 5, 9),
            allocator_stats(4, 10, 6, 14),
        ]
        with (
            patch("torch.cuda.synchronize", side_effect=AssertionError("No synchronization")),
            patch("torch.cuda.Event", side_effect=AssertionError("No CUDA event")),
            patch("torch.cuda.memory_stats", side_effect=stats) as reads,
            patch("torch.cuda.reset_peak_memory_stats") as resets,
            patch("torch.cuda.get_device_properties", return_value=SimpleNamespace(
                name="test GPU", total_memory=100,
            )),
        ):
            logger = self.logger("cuda:0")
            for index in range(2):
                logger.begin_sample()
                logger.log_sample(str(index), 1.0, sample_result(), data_step_before=0)
            summary = logger.log_update(1, 1, 3)
        self.assertEqual(summary["peak_allocated_bytes"], 10)
        self.assertEqual(summary["peak_reserved_bytes"], 14)
        self.assertEqual(summary["session_peak_allocated_gib"], 10 / 2**30)
        self.assertEqual(reads.call_count, 5)
        self.assertEqual(resets.call_count, 4)  # Session, each sample, next update.
        self.assertEqual(self.rows()[0]["gpu_name"], "test GPU")

    def test_eta_ignores_two_warmup_windows_and_uses_completed_data_windows(self):
        # 100s and 200s warmup must not skew 10s steady-state ETA.
        with patch("cot_mtkd.stage2.performance.time.perf_counter", side_effect=[
            0, 1, 101, 102, 302, 303, 313, 314, 334,
        ]):
            logger = self.logger()
            summaries = []
            for index in range(4):
                logger.begin_sample()
                logger.log_sample("sample", 1.0, sample_result(), data_step_before=index)
                summaries.append(logger.log_update(index + 1, index, 5, skipped_update=True))
        self.assertIsNone(summaries[1]["eta_remaining_wall_seconds_estimate"])
        self.assertEqual(summaries[2]["mean_update_wall_seconds_after_warmup"], 10)
        self.assertEqual(summaries[2]["eta_remaining_wall_seconds_estimate"], 20)
        self.assertEqual(summaries[3]["mean_update_wall_seconds_after_warmup"], 15)
        self.assertEqual(summaries[3]["eta_remaining_wall_seconds_estimate"], 15)
        self.assertTrue(all(value["local_examples"] == 1 for value in summaries))

    def test_resume_trims_sample_update_and_session_rows_to_checkpoint(self):
        logger = self.logger()
        for index in range(2):
            logger.begin_sample()
            logger.log_sample(str(index), 1.0, sample_result(), data_step_before=index)
            logger.log_update(index + 1, index + 1, 3)
        self.logger(data_step=1, resume=True)
        rows = self.rows()
        self.assertEqual([row["sample_id"] for row in rows if "sample_id" in row], ["0"])
        updates = [row for row in rows if row["event"] == "stage2_update_performance"]
        self.assertEqual([row["data_step"] for row in updates], [1])
        self.assertEqual(rows[-1]["event"], "stage2_performance_session")
        self.assertTrue(rows[-1]["resumed"])
        self.assertEqual(rows[-1]["data_step_before"], 1)

    def test_distributed_files_are_per_rank_and_fresh_runs_clear_old_rows(self):
        self.assertEqual(performance_log_filename(0, 1), "performance.jsonl")
        self.assertEqual(performance_log_filename(7, 8), "performance.rank00007.jsonl")
        self.logger(rank=0, world_size=2)
        self.logger(rank=1, world_size=2)
        self.assertEqual(self.rows("performance.rank00000.jsonl")[0]["rank"], 0)
        self.assertEqual(self.rows("performance.rank00001.jsonl")[0]["rank"], 1)
        self.logger(rank=0, world_size=2)
        self.assertEqual(len(self.rows("performance.rank00000.jsonl")), 1)

    def test_update_without_samples_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "without any samples"):
            self.logger().log_update(1, 1, 1)


if __name__ == "__main__":
    unittest.main()
