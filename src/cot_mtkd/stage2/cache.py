from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file


class SparseCacheWriter:
    def __init__(
        self,
        output_dir: str | Path,
        rank: int,
        tokens_per_shard: int,
        probability_dtype: torch.dtype,
        world_size: int = 1,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rank = rank
        self.world_size = int(world_size)
        self.tokens_per_shard = int(tokens_per_shard)
        self.probability_dtype = probability_dtype
        self.shard_number = 0
        self.ids: list[torch.Tensor] = []
        self.probabilities: list[torch.Tensor] = []
        self.tail: list[torch.Tensor] = []
        self.positions: list[torch.Tensor] = []
        self.pending: list[tuple[str, int, int, str]] = []
        self.index: dict[str, dict[str, Any]] = {}
        self.token_count = 0

    def add(
        self,
        sample_id: str,
        token_positions: torch.Tensor,
        top_ids: torch.Tensor,
        top_probabilities: torch.Tensor,
        tail_mass: torch.Tensor,
    ) -> None:
        count = int(token_positions.numel())
        if self.token_count and self.token_count + count > self.tokens_per_shard:
            self.flush()
        if (
            top_ids.shape[0] != count
            or top_probabilities.shape != top_ids.shape
            or tail_mass.numel() != count
        ):
            raise ValueError("Sparse teacher tensors have inconsistent shapes")
        start = self.token_count
        end = start + count
        ids_cpu = top_ids.detach().to("cpu", dtype=torch.int32).contiguous()
        checksum = hashlib.sha256(ids_cpu.numpy().tobytes()).hexdigest()
        self.ids.append(ids_cpu)
        self.probabilities.append(
            top_probabilities.detach()
            .to("cpu", dtype=self.probability_dtype)
            .contiguous()
        )
        self.tail.append(tail_mass.detach().to("cpu", dtype=torch.float32).contiguous())
        self.positions.append(
            token_positions.detach().to("cpu", dtype=torch.int32).contiguous()
        )
        self.pending.append((sample_id, start, end, checksum))
        self.token_count = end

    def flush(self) -> None:
        if not self.ids:
            return
        filename = (
            f"cache-rank{self.rank:05d}-of{self.world_size:05d}-"
            f"shard{self.shard_number:05d}.safetensors"
        )
        path = self.output_dir / filename
        save_file(
            {
                "top_ids": torch.cat(self.ids),
                "top_probabilities": torch.cat(self.probabilities),
                "tail_mass": torch.cat(self.tail),
                "token_positions": torch.cat(self.positions),
            },
            path,
            metadata={"format": "gac-cot-mtkd-topk-tail-v1"},
        )
        for sample_id, start, end, checksum in self.pending:
            if sample_id in self.index:
                raise ValueError(f"Duplicate cache sample id: {sample_id}")
            self.index[sample_id] = {
                "file": filename,
                "start": start,
                "end": end,
                "support_checksum": checksum,
            }
        self.shard_number += 1
        self.ids.clear()
        self.probabilities.clear()
        self.tail.clear()
        self.positions.clear()
        self.pending.clear()
        self.token_count = 0

    def close(self) -> Path:
        self.flush()
        index_path = self.output_dir / (
            f"index-rank{self.rank:05d}-of{self.world_size:05d}.json"
        )
        with index_path.open("w", encoding="utf-8") as handle:
            json.dump(self.index, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        return index_path


class SparseTeacherCache:
    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = Path(cache_dir)
        with (self.cache_dir / "index.json").open("r", encoding="utf-8") as handle:
            self.index: dict[str, dict[str, Any]] = json.load(handle)
        self.current_file: str | None = None
        self.current_tensors: dict[str, torch.Tensor] | None = None

    def get(self, sample_id: str) -> dict[str, torch.Tensor]:
        entry = self.index[sample_id]
        if entry["file"] != self.current_file:
            self.current_tensors = load_file(
                self.cache_dir / entry["file"], device="cpu"
            )
            self.current_file = entry["file"]
        assert self.current_tensors is not None
        selection = slice(int(entry["start"]), int(entry["end"]))
        value = {key: tensor[selection] for key, tensor in self.current_tensors.items()}
        ids = value["top_ids"].to(dtype=torch.int32).contiguous()
        checksum = hashlib.sha256(ids.numpy().tobytes()).hexdigest()
        if checksum != entry["support_checksum"]:
            raise RuntimeError(
                f"Sparse support checksum mismatch for sample {sample_id}"
            )
        return value


def probability_dtype(name: str) -> torch.dtype:
    normalized = name.lower()
    if normalized in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if normalized in {"float16", "fp16"}:
        return torch.float16
    raise ValueError(f"Sparse cache probability dtype must be BF16 or FP16, got {name}")
