from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from cot_mtkd.data.schema import PreparedRecord, TokenRegion
from cot_mtkd.stage1.publish import publish_stage1
from cot_mtkd.stage1.trainer import train_stage1
from cot_mtkd.utils.distributed import DistributedContext
from cot_mtkd.utils.manifest import file_sha256, write_json


class FakeHub:
    def create_repo(self, **kwargs) -> None:
        self.repo = kwargs

    def create_commit(self, **kwargs) -> SimpleNamespace:
        self.files = {operation.path_in_repo for operation in kwargs["operations"]}
        return SimpleNamespace(oid="tiny-stage1-commit")

    def list_repo_files(self, **kwargs) -> list[str]:
        assert kwargs["revision"] == "tiny-stage1-commit"
        return sorted(self.files)


def _prepared_record() -> PreparedRecord:
    ids = [1, 2, 3, 4, 5, 6]
    return PreparedRecord(
        sample_id="tiny-0",
        input_ids=ids,
        labels=[-100, -100, 3, 4, 5, 6],
        attention_mask=[1] * len(ids),
        offset_mapping=[(index, index + 1) for index in range(len(ids))],
        region_ids=[
            int(TokenRegion.PROMPT),
            int(TokenRegion.PROMPT),
            int(TokenRegion.REASONING),
            int(TokenRegion.REASONING),
            int(TokenRegion.ANSWER),
            int(TokenRegion.EOS),
        ],
        step_ids=[-1, -1, 0, 0, -1, -1],
        question="tiny question",
        thinking="two tokens",
        attempt="5",
        solution="5",
        deepseek_grade=None,
        original_length=len(ids),
        kept_length=len(ids),
        original_steps=1,
        kept_steps=1,
        truncated=False,
        answer_start=4,
        reasoning_start=2,
        tokenizer_fingerprint="tiny-tokenizer",
    )


def _stage1_config(prepared: Path, output: Path) -> dict:
    model = {"name_or_path": "Qwen/tiny", "revision": "test-revision"}
    return {
        "schema_version": 1,
        "seed": 42,
        "model": model,
        "lora": {
            "rank": 2,
            "alpha": 2,
            "dropout": 0.0,
            "target_modules": ["q_proj"],
        },
        "stage1": {
            "num_experts": 2,
            "epochs": 1,
            "learning_rate": 1.0e-3,
            "micro_batch_size": 1,
            "global_batch_size": 1,
            "gradient_accumulation_steps": None,
            "step_drop_probability": 0.0,
            "diversity_weight": 0.2,
            "repulsion_weight": 0.1,
            "max_grad_norm": 1.0,
            "checkpoint_every_steps": 1,
            "log_every_steps": 1,
            "resume_from": None,
        },
        "dpp": {"jitter": 1.0e-4, "max_jitter": 1.0e-2},
        "grassmann": {
            "rank_epsilon": 1.0e-3,
            "angle_epsilon": 1.0e-6,
            "bandwidth_floor": 1.0e-8,
        },
        "optimizer": {
            "name": "adamw",
            "betas": [0.9, 0.999],
            "eps": 1.0e-8,
            "weight_decay": 0.0,
        },
        "scheduler": {"name": "cosine", "warmup_ratio": 0.0, "min_lr_ratio": 0.0},
        "runtime": {
            "lm_head_chunk_tokens": 4,
            "council_chunk_tokens": 2,
            "dataloader_workers": 0,
            "pin_memory": False,
            "probe_hidden_device": "cpu",
        },
        "paths": {"prepared": str(prepared), "output": str(output)},
        "_project_root": str(Path(__file__).resolve().parents[1]),
    }


class Stage1PublicationIntegrationTest(unittest.TestCase):
    def test_tiny_training_artifacts_are_publishable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepared = root / "prepared"
            output = root / "stage1"
            prepared.mkdir()
            data = prepared / "data.jsonl"
            data.write_text(json.dumps(_prepared_record().to_dict()) + "\n", encoding="utf-8")
            prepared_config = prepared / "config.yaml"
            prepared_config.write_text("model: Qwen/tiny\n", encoding="utf-8")
            config = _stage1_config(prepared, output)
            write_json(
                prepared / "manifest.json",
                {
                    "tokenizer_fingerprint": "tiny-tokenizer",
                    "data_file": "data.jsonl",
                    "data_file_sha256": file_sha256(data),
                    "config_file": "config.yaml",
                    "config_file_sha256": file_sha256(prepared_config),
                    "config": {"model": config["model"]},
                },
            )
            base = Qwen2ForCausalLM(
                Qwen2Config(
                    vocab_size=32,
                    hidden_size=16,
                    intermediate_size=32,
                    num_hidden_layers=1,
                    num_attention_heads=2,
                    num_key_value_heads=1,
                    max_position_embeddings=32,
                )
            )
            distributed = DistributedContext(0, 0, 1, torch.device("cpu"))
            with (
                patch("cot_mtkd.stage1.trainer.load_tokenizer", return_value=SimpleNamespace(pad_token_id=0)),
                patch("cot_mtkd.stage1.trainer.tokenizer_fingerprint", return_value="tiny-tokenizer"),
                patch("cot_mtkd.models.multi_adapter.load_base_causal_lm", return_value=base),
            ):
                manifest = train_stage1(config, distributed)

            self.assertEqual(manifest["global_step"], 1)
            self.assertEqual(manifest["adapter_names"], ["expert_0", "expert_1"])
            for name, specification in manifest["expert_files"].items():
                for file_key, hash_key in (
                    ("adapter_config", "adapter_config_sha256"),
                    ("adapter_weights", "adapter_weights_sha256"),
                ):
                    self.assertEqual(
                        file_sha256(output / specification[file_key]),
                        specification[hash_key],
                        msg=f"{name}/{file_key}",
                    )
            hub = FakeHub()
            published = publish_stage1(output, "user/tiny-experts", api=hub)
            self.assertEqual(published["revision"], "tiny-stage1-commit")
            self.assertEqual(len(published["files"]), 5)
            self.assertNotIn("checkpoint.pt", published["files"])


if __name__ == "__main__":
    unittest.main()
