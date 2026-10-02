import json
import tempfile
import unittest
from pathlib import Path

from cot_mtkd.evaluation.stage1_runner import (
    benchmark_metrics,
    evaluate_stage1_experts,
    resolve_expert_adapters,
)
from cot_mtkd.utils.manifest import file_sha256

MODEL = {"name_or_path": "tiny/model", "revision": "abc", "dtype": "bfloat16"}
EXPERTS = ["expert_0", "expert_1", "expert_2"]


class FakeTokenizer:
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": list(range(len(text.split())))}


class FakeEngine:
    """expert_k answers correctly on its first k+1 samples (out of three)."""

    def __init__(self):
        self.tokenizer = FakeTokenizer()
        self.calls = []

    def generate(self, prompt_token_ids, seeds, adapter_name, adapter_path):
        self.calls.append((adapter_name, Path(adapter_path).name, list(seeds)))
        correct = int(adapter_name.rsplit("_", 1)[1]) + 1
        return [
            [
                (r"\boxed{4}" if sample < correct else r"\boxed{5}", "stop")
                for sample in range(3)
            ]
            for _ in prompt_token_ids
        ]


def write_stage1(root: Path) -> Path:
    stage1 = root / "stage1"
    expert_files = {}
    for index, name in enumerate(EXPERTS):
        adapter = stage1 / "final" / "adapters" / name
        adapter.mkdir(parents=True)
        (adapter / "adapter_config.json").write_text(json.dumps({"r": 16}))
        (adapter / "adapter_model.safetensors").write_bytes(bytes([index]) * 8)
        expert_files[name] = {
            "adapter_config": f"final/adapters/{name}/adapter_config.json",
            "adapter_config_sha256": file_sha256(adapter / "adapter_config.json"),
            "adapter_weights": f"final/adapters/{name}/adapter_model.safetensors",
            "adapter_weights_sha256": file_sha256(adapter / "adapter_model.safetensors"),
        }
    manifest = {
        "adapter_names": EXPERTS,
        "expert_files": expert_files,
        "config": {"model": MODEL, "lora": {"rank": 16}},
    }
    (stage1 / "manifest.json").write_text(json.dumps(manifest))
    return stage1


def make_config(root: Path, stage1: Path) -> dict:
    benchmark = root / "bench.jsonl"
    benchmark.write_text(
        "".join(
            json.dumps({"problem": f"What is 2+2? ({index})", "answer": "4"}) + "\n"
            for index in range(2)
        )
    )
    return {
        "seed": 42,
        "model": MODEL,
        "stage1_checkpoint": "final",
        "vllm": {
            "max_model_len": 4096,
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": 0.8,
            "max_lora_rank": 16,
        },
        "generation": {
            "prompt_prefix": "Reason step by step.",
            "n": 3,
            "temperature": 0.6,
            "top_p": 0.9,
            "repetition_penalty": 1.05,
            "max_tokens": 4096,
        },
        "benchmarks": [
            {"name": "toy", "local_json": str(benchmark), "split": "train", "expected_records": 2}
        ],
        "grading": {"timeout_seconds": 5, "prefer_math_verify": False},
        "paths": {"stage1": str(stage1), "output": str(root / "evaluation")},
        "_project_root": str(root),
    }


class Stage1EvaluationTests(unittest.TestCase):
    def test_experts_are_evaluated_in_order_with_pass_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = make_config(root, write_stage1(root))
            engine = FakeEngine()
            manifest = evaluate_stage1_experts(config, engine=engine)

            self.assertEqual([call[0] for call in engine.calls], EXPERTS)
            self.assertEqual([call[1] for call in engine.calls], EXPERTS)
            seeds = [call[2] for call in engine.calls]
            self.assertEqual(seeds[0], seeds[1])
            self.assertEqual(seeds[1], seeds[2])
            for index, name in enumerate(EXPERTS):
                toy = manifest["experts"][name]["benchmarks"]["toy"]
                self.assertEqual(toy["problems"], 2)
                self.assertAlmostEqual(toy["pass_at_1"], (index + 1) / 3)
                self.assertAlmostEqual(toy["pass_at_3"], 1.0)
                self.assertEqual(toy["length_capped_fraction"], 0.0)
                self.assertTrue((root / "evaluation" / name / "toy.jsonl").is_file())
            self.assertTrue((root / "evaluation" / "summary.md").is_file())

            rerun = FakeEngine()
            evaluate_stage1_experts(config, engine=rerun)
            self.assertEqual(rerun.calls, [])

            evaluate_stage1_experts(config, experts=["expert_1"], force=True, engine=rerun)
            self.assertEqual([call[0] for call in rerun.calls], ["expert_1"])

    def test_tampered_adapter_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            stage1 = write_stage1(Path(directory))
            manifest = json.loads((stage1 / "manifest.json").read_text())
            (stage1 / "final/adapters/expert_2/adapter_model.safetensors").write_bytes(b"x")
            with self.assertRaises(RuntimeError):
                resolve_expert_adapters(stage1, manifest, "final")
            with self.assertRaises(ValueError):
                resolve_expert_adapters(stage1, manifest, "final", ["expert_9"])

    def test_length_capped_fraction(self) -> None:
        records = [
            {"correct": [True, False, False], "finish_reasons": ["stop", "length", "stop"]},
            {"correct": [False, False, False], "finish_reasons": ["length", "length", "stop"]},
        ]
        metrics = benchmark_metrics(records)
        self.assertAlmostEqual(metrics["pass_at_1"], 1 / 6)
        self.assertAlmostEqual(metrics["pass_at_3"], 0.5)
        self.assertAlmostEqual(metrics["length_capped_fraction"], 3 / 6)


if __name__ == "__main__":
    unittest.main()
