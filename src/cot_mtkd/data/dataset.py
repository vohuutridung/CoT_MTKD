from __future__ import annotations

import glob
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from .schema import PreparedRecord


class JsonlRecordDataset(Dataset[PreparedRecord]):
    """Lazy random-access JSONL dataset with byte-offset indexing."""

    def __init__(self, path_or_directory: str | Path, pattern: str = "*.jsonl") -> None:
        source = Path(path_or_directory)
        if source.is_dir():
            preferred = source / "data.jsonl"
            files = (
                [preferred]
                if preferred.exists()
                else [Path(item) for item in sorted(glob.glob(str(source / pattern)))]
            )
        else:
            files = [source]
        if not files:
            raise FileNotFoundError(f"No JSONL files found at {source}")
        self.files = files
        self.index: list[tuple[int, int]] = []
        for file_index, path in enumerate(self.files):
            with path.open("rb") as handle:
                while True:
                    offset = handle.tell()
                    line = handle.readline()
                    if not line:
                        break
                    if line.strip():
                        self.index.append((file_index, offset))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int) -> PreparedRecord:
        file_index, offset = self.index[index]
        with self.files[file_index].open("rb") as handle:
            handle.seek(offset)
            value = json.loads(handle.readline())
        return PreparedRecord.from_dict(value)


def load_jsonl_by_id(
    path_or_directory: str | Path, pattern: str = "*.jsonl"
) -> dict[str, dict[str, Any]]:
    source = Path(path_or_directory)
    files = sorted(source.glob(pattern)) if source.is_dir() else [source]
    return load_jsonl_files(files)


def load_jsonl_files(paths: Iterable[str | Path]) -> dict[str, dict[str, Any]]:
    values: dict[str, dict[str, Any]] = {}
    for item in paths:
        path = Path(item)
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    record = json.loads(line)
                    sample_id = str(record["sample_id"])
                    if sample_id in values:
                        raise RuntimeError(
                            f"Duplicate sample id across JSONL shards: {sample_id}"
                        )
                    values[sample_id] = record
    return values
