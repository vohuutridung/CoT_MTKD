from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch

from ..utils.local_logging import JsonlLogger
from .step_logging import trim_jsonl_to_checkpoint


def performance_log_filename(rank: int, world_size: int) -> str:
    if world_size == 1:
        return "performance.jsonl"
    return f"performance.rank{rank:05d}.jsonl"


class TrainingPerformanceLogger:
    """Host wall timing and allocator counters, with no CUDA synchronization.

    Sample memory peaks cover record gradients and gradient accumulation. Update
    peaks also cover the existing all-reduce and optimizer calls. All throughput
    and timing values are rank-local; GPU kernels are not separately timed.
    """

    def __init__(
        self,
        output_dir: Path,
        device: torch.device,
        rank: int,
        world_size: int,
        run_fingerprint: str,
        config_fingerprint: str,
        config: dict[str, Any],
        data_step: int,
        resume: bool,
    ) -> None:
        path = output_dir / performance_log_filename(rank, world_size)
        if resume:
            trim_jsonl_to_checkpoint(path, run_fingerprint, data_step, before_update=True)
        self.logger = JsonlLogger(path, truncate=not resume)
        self.device = device
        self.common = {"run_fingerprint": run_fingerprint, "rank": rank}
        self.started = time.perf_counter()
        self.update_started: float | None = None
        self.local_examples = self.prefix_tokens = self.reasoning_tokens = 0
        self.update_peak_allocated = self.update_peak_reserved = 0
        self.session_peak_allocated = self.session_peak_reserved = 0
        self.completed_windows = self.timed_windows = 0
        self.timed_seconds = 0.0
        properties = torch.cuda.get_device_properties(device) if device.type == "cuda" else None
        self.logger.log(
            "stage2_performance_session",
            **self.common,
            data_step_before=data_step,
            config_fingerprint=config_fingerprint,
            config=config,
            resumed=resume,
            world_size=world_size,
            device=str(device),
            gpu_name=properties.name if properties is not None else None,
            gpu_total_memory_bytes=properties.total_memory if properties is not None else None,
            torch_version=str(torch.__version__),
            timing_mode="host_wall_no_cuda_sync",
            memory_scope="pytorch_allocator_on_this_rank_excluding_model_load",
            eta_warmup_windows=2,
        )
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

    def begin_sample(self) -> float:
        # Carry peaks from optimizer/cleanup allocations between sample windows
        # before resetting the allocator's high-water marks for this sample.
        self._memory()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        started = time.perf_counter()
        if self.update_started is None:
            self.update_started = started
        return started

    def _memory(self) -> dict[str, int | float | None]:
        names = {
            "allocated": "allocated_bytes.all.current",
            "reserved": "reserved_bytes.all.current",
            "peak_allocated": "allocated_bytes.all.peak",
            "peak_reserved": "reserved_bytes.all.peak",
        }
        if self.device.type != "cuda":
            return {f"{name}_{unit}": None for name in names for unit in ("bytes", "gib")}
        # One host-side allocator query, not a CUDA event, profiler or NVML poll.
        stats = torch.cuda.memory_stats(self.device)
        values = {name: int(stats[key]) for name, key in names.items()}
        self.update_peak_allocated = max(self.update_peak_allocated, values["peak_allocated"])
        self.update_peak_reserved = max(self.update_peak_reserved, values["peak_reserved"])
        self.session_peak_allocated = max(self.session_peak_allocated, values["peak_allocated"])
        self.session_peak_reserved = max(self.session_peak_reserved, values["peak_reserved"])
        return {
            f"{name}_{unit}": value if unit == "bytes" else value / 2**30
            for name, value in values.items()
            for unit in ("bytes", "gib")
        }

    def log_sample(
        self,
        sample_id: str,
        record_gradient_wall_seconds: float,
        result: Any,
        **cursor: Any,
    ) -> None:
        memory = self._memory()
        prefix = int(result.metrics["prefix_tokens"])
        reasoning = int(result.metrics["reasoning_tokens"])
        self.local_examples += 1
        self.prefix_tokens += prefix
        self.reasoning_tokens += reasoning
        self.logger.log(
            "stage2_sample_performance",
            **self.common,
            **cursor,
            sample_id=sample_id,
            record_gradient_wall_seconds=record_gradient_wall_seconds,
            prefix_tokens=prefix,
            reasoning_tokens=reasoning,
            retained_steps=result.steps,
            discarded_steps=result.discarded_steps,
            head_chunks=int(result.metrics["head_chunks"]),
            cached_steps=int(result.metrics["cached_steps"]),
            recomputed_steps=int(result.metrics["recomputed_steps"]),
            teacher_head_chunk_sweeps=int(result.metrics["teacher_head_chunk_sweeps"]),
            **memory,
        )

    def log_update(self, data_step: int, global_step: int, total_steps: int, **values: Any) -> dict:
        if self.update_started is None:
            raise RuntimeError("Cannot log a performance update without any samples")
        now = time.perf_counter()
        elapsed = now - self.update_started
        memory = self._memory()
        if self.device.type == "cuda":
            memory.update(
                peak_allocated_bytes=self.update_peak_allocated,
                peak_reserved_bytes=self.update_peak_reserved,
                peak_allocated_gib=self.update_peak_allocated / 2**30,
                peak_reserved_gib=self.update_peak_reserved / 2**30,
            )
        self.completed_windows += 1
        warmup = self.completed_windows <= 2
        if not warmup:
            self.timed_windows += 1
            self.timed_seconds += elapsed
        mean = self.timed_seconds / self.timed_windows if self.timed_windows else None
        remaining = max(0, total_steps - data_step)
        eta = 0.0 if not remaining else remaining * mean if mean is not None else None
        summary = {
            "update_window_wall_seconds": elapsed,
            "session_elapsed_wall_seconds": now - self.started,
            "local_examples": self.local_examples,
            "local_prefix_tokens": self.prefix_tokens,
            "local_reasoning_tokens": self.reasoning_tokens,
            "local_prefix_tokens_per_second": self.prefix_tokens / elapsed if elapsed > 0 else None,
            "local_examples_per_second": self.local_examples / elapsed if elapsed > 0 else None,
            "timing_warmup": warmup,
            "mean_update_wall_seconds_after_warmup": mean,
            "eta_remaining_wall_seconds_estimate": eta,
            "session_peak_allocated_gib": (
                self.session_peak_allocated / 2**30 if self.device.type == "cuda" else None
            ),
            "session_peak_reserved_gib": (
                self.session_peak_reserved / 2**30 if self.device.type == "cuda" else None
            ),
            **memory,
        }
        self.logger.log(
            "stage2_update_performance",
            **self.common,
            data_step_before=data_step - 1,
            data_step=data_step,
            global_step=global_step,
            **values,
            **summary,
        )
        self.update_started = None
        self.local_examples = self.prefix_tokens = self.reasoning_tokens = 0
        self.update_peak_allocated = self.update_peak_reserved = 0
        # Keep the next window independent of the previous sample/optimizer peak.
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        return summary
