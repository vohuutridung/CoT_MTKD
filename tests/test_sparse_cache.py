from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from cot_mtkd.stage2.cache import SparseCacheWriter, SparseTeacherCache


class SparseCacheTest(unittest.TestCase):
    def test_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            writer = SparseCacheWriter(
                directory, rank=0, tokens_per_shard=4, probability_dtype=torch.float16
            )
            writer.add(
                "a",
                torch.tensor([2, 3]),
                torch.tensor([[1, 2], [2, 3]]),
                torch.tensor([[0.4, 0.3], [0.5, 0.2]]),
                torch.tensor([0.3, 0.3]),
            )
            index_path = writer.close()
            Path(directory, "index.json").write_text(
                index_path.read_text(), encoding="utf-8"
            )
            cache = SparseTeacherCache(directory)
            value = cache.get("a")
            self.assertTrue(
                torch.equal(
                    value["token_positions"], torch.tensor([2, 3], dtype=torch.int32)
                )
            )
            self.assertEqual(value["top_ids"].shape, (2, 2))

    def test_support_checksum_is_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            writer = SparseCacheWriter(
                directory, rank=0, tokens_per_shard=4, probability_dtype=torch.float16
            )
            writer.add(
                "a",
                torch.tensor([2]),
                torch.tensor([[1, 2]]),
                torch.tensor([[0.4, 0.3]]),
                torch.tensor([0.3]),
            )
            index_path = writer.close()
            index = json.loads(index_path.read_text(encoding="utf-8"))
            index["a"]["support_checksum"] = "0" * 64
            Path(directory, "index.json").write_text(
                json.dumps(index), encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "checksum mismatch"):
                SparseTeacherCache(directory).get("a")


if __name__ == "__main__":
    unittest.main()
