from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from cot_mtkd.config import project_root
from cot_mtkd.evaluation.adapter import adapter_rank, resolve_adapter
from cot_mtkd.evaluation.benchmarks import load_benchmark
from cot_mtkd.evaluation.metrics import benchmark_metrics, macro_average, seed_summary
from cot_mtkd.evaluation.runner import render_prompt


def _record(correct: list[bool]) -> dict:
    return {"correct": correct, "token_counts": [10] * len(correct),
            "truncated": [False] * len(correct)}


class MetricsTest(unittest.TestCase):
    def test_pass_at_1_is_mean_over_samples(self) -> None:
        metrics = benchmark_metrics([_record([True, False, False]), _record([False] * 3)])
        self.assertAlmostEqual(metrics["pass@1"], (1 / 3) / 2)
        self.assertAlmostEqual(metrics["pass@3"], 0.5)
        self.assertEqual(metrics["k"], 3)

    def test_macro_average_and_seed_summary(self) -> None:
        runs = {}
        for seed, value in ((42, 0.3), (43, 0.5)):
            benchmarks = {
                "a": {"pass@1": value, "pass@3": 1.0},
                "b": {"pass@1": value + 0.2, "pass@3": 0.0},
            }
            runs[seed] = {"benchmarks": benchmarks, **macro_average(benchmarks)}
        self.assertAlmostEqual(runs[42]["avg"], 0.4)
        self.assertAlmostEqual(runs[42]["avg_passk"], 0.5)
        summary = seed_summary(runs)
        self.assertAlmostEqual(summary["avg"]["mean"], 0.5)
        self.assertAlmostEqual(summary["benchmarks"]["a"]["pass@1"]["mean"], 0.4)
        self.assertGreater(summary["avg"]["std"], 0.0)


class BenchmarkTest(unittest.TestCase):
    def test_vendored_benchmarks(self) -> None:
        sizes = {"math500": 500, "aime24": 30, "aime25": 30, "amc": 83}
        for name, size in sizes.items():
            items = load_benchmark(
                {"name": name, "local_json": f"data/eval/{name}.jsonl",
                 "expected_records": size},
                project_root(),
            )
            self.assertEqual(len(items), size)
            self.assertFalse(items[0].question.startswith("Please reason"))
        amc = load_benchmark({"name": "amc", "local_json": "data/eval/amc.jsonl"}, project_root())
        self.assertEqual(amc[0].answer, "142")


class _Tokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        return f"<user>{messages[0]['content']}</user><assistant>"


class PromptTest(unittest.TestCase):
    def test_instruction_joined_with_a_space(self) -> None:
        prompt = render_prompt(_Tokenizer(), "Reason.", "1+1?")
        self.assertEqual(prompt, "<user>Reason. 1+1?</user><assistant>")


class AdapterTest(unittest.TestCase):
    def _write_adapter(self, path: Path, rank: int = 16) -> None:
        path.mkdir(parents=True)
        (path / "adapter_config.json").write_text(json.dumps({"r": rank}))

    def test_resolves_stage2_dir_packed_dir_and_peft_dir(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            student = root / "run" / "final" / "adapters" / "student"
            self._write_adapter(student, rank=32)
            (root / "run" / "manifest.json").write_text(
                json.dumps({"artifact": "stage2_checkpoint", "global_step": 7})
            )
            for source in (root / "run", student):
                resolved, manifest = resolve_adapter({"path": str(source)}, root)
                self.assertEqual(resolved, student)
                self.assertEqual(manifest["global_step"], 7)
            self.assertEqual(adapter_rank(student), 32)

    def test_base_model_and_missing_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(resolve_adapter({"path": "base"}, root), (None, None))
            (root / "empty").mkdir()
            with self.assertRaises(FileNotFoundError):
                resolve_adapter({"path": "empty"}, root)


if __name__ == "__main__":
    unittest.main()
