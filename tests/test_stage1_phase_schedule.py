from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from cot_mtkd.data.schema import PreparedRecord
from cot_mtkd.models.chunked_head import full_vocab_probe
from cot_mtkd.stage1.rbf import effective_update_distances
from cot_mtkd.stage1.trainer import train_stage1
from cot_mtkd.utils.distributed import DistributedContext
from cot_mtkd.utils.manifest import read_json as disk_read_json


def prepared_record(index: int) -> PreparedRecord:
    ids = list(range(1, 15))
    regions = [0, 0, 1] + [2] * 8 + [4, 5, 6]
    steps = [-1, -1, -1] + [0] * 4 + [1] * 4 + [-1] * 3
    if index % 2:
        keep = [position for position in range(len(ids)) if position not in (5, 6)]
        ids = [ids[position] for position in keep]
        regions = [regions[position] for position in keep]
        steps = [steps[position] for position in keep]
    return PreparedRecord(
        sample_id=f"sample-{index}",
        input_ids=ids,
        labels=[-100 if region == 0 else token for token, region in zip(ids, regions)],
        attention_mask=[1] * len(ids),
        offset_mapping=[(0, 0)] * len(ids),
        region_ids=regions,
        step_ids=steps,
        question="",
        thinking="",
        attempt="",
        solution="",
        deepseek_grade=None,
        original_length=len(ids),
        kept_length=len(ids),
        original_steps=2,
        kept_steps=2,
        truncated=False,
        answer_start=regions.index(4),
        reasoning_start=regions.index(2),
        tokenizer_fingerprint="test-tokenizer",
    )


class Stage1PhaseScheduleTest(unittest.TestCase):
    def test_train_uses_sft_ramp_and_full_paths(self) -> None:
        torch.manual_seed(41)
        base = Qwen2ForCausalLM(
            Qwen2Config(
                vocab_size=64,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=1,
                max_position_embeddings=32,
            )
        )
        base.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        base.enable_input_require_grads()
        base.config.use_cache = False
        records = [prepared_record(index) for index in range(12)]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = {
                "_project_root": str(root),
                "seed": 42,
                "model": {
                    "name_or_path": "tiny",
                    "revision": "test",
                    "dtype": "float32",
                },
                "lora": {
                    "rank": 2,
                    "alpha": 2,
                    "dropout": 0.2,
                    "target_modules": [
                        "q_proj",
                        "k_proj",
                        "v_proj",
                        "o_proj",
                        "gate_proj",
                        "up_proj",
                        "down_proj",
                    ],
                },
                "stage1": {
                    "num_experts": 3,
                    "forward_mode": "one_pass",
                    "epochs": 1,
                    "learning_rate": 1.0e-3,
                    "micro_batch_size": 1,
                    "global_batch_size": 2,
                    "gradient_accumulation_steps": None,
                    "dpp_weight": 0.2,
                    "rbf_weight": 1.0,
                    "interaction_off_until": 0.1,
                    "interaction_ramp_until": 0.3,
                    "max_grad_norm": 1.0,
                    "checkpoint_every_steps": 100,
                    "log_every_steps": 1,
                    "resume_from": None,
                },
                "kneedle": {"probe_k": 6, "min_k": 3, "max_k": 4},
                "dpp": {"jitter": 1.0e-4, "max_jitter": 1.0e-2},
                "rbf": {"bandwidth_ema": 0.9, "bandwidth_floor": 1.0e-12},
                "optimizer": {
                    "name": "adamw",
                    "betas": [0.9, 0.999],
                    "eps": 1.0e-8,
                    "weight_decay": 0.0,
                },
                "scheduler": {
                    "name": "cosine",
                    "warmup_ratio": 0.1,
                    "min_lr_ratio": 0.0,
                },
                "runtime": {
                    "lm_head_chunk_tokens": 4,
                    "dataloader_workers": 0,
                    "pin_memory": False,
                    "probe_hidden_device": "cpu",
                },
                "paths": {
                    "prepared": str(root / "prepared"),
                    "output": str(root / "output"),
                },
            }
            manifest = {
                "tokenizer_fingerprint": "test-tokenizer",
                "config": {"model": config["model"]},
            }
            distributed = DistributedContext(0, 0, 1, torch.device("cpu"))
            with (
                patch(
                    "cot_mtkd.models.multi_adapter.load_base_causal_lm",
                    return_value=base,
                ),
                patch(
                    "cot_mtkd.stage1.trainer.read_json",
                    side_effect=lambda path: (
                        manifest
                        if Path(path).parent == root / "prepared"
                        else disk_read_json(path)
                    ),
                ),
                patch("cot_mtkd.stage1.trainer.require_file_sha256"),
                patch(
                    "cot_mtkd.stage1.trainer.load_tokenizer",
                    return_value=SimpleNamespace(pad_token_id=0),
                ),
                patch(
                    "cot_mtkd.stage1.trainer.tokenizer_fingerprint",
                    return_value="test-tokenizer",
                ),
                patch(
                    "cot_mtkd.stage1.trainer.JsonlRecordDataset", return_value=records
                ),
                patch(
                    "cot_mtkd.stage1.trainer.full_vocab_probe", wraps=full_vocab_probe
                ) as probe_calls,
                patch(
                    "cot_mtkd.stage1.trainer.effective_update_distances",
                    wraps=effective_update_distances,
                ) as rbf_calls,
            ):
                result = train_stage1(config, distributed)
            self.assertEqual(probe_calls.call_count, 30)
            self.assertEqual(rbf_calls.call_count, 5)
            self.assertEqual(result["global_step"], 6)
            with (root / "output" / "metrics.jsonl").open(encoding="utf-8") as handle:
                steps = [
                    json.loads(line)
                    for line in handle
                    if '"event": "stage1_step"' in line
                ]
            self.assertEqual(
                [item["interaction_phase"] for item in steps],
                ["sft", "ramp", "full", "full", "full", "full"],
            )
            self.assertEqual(steps[0]["dpp_loss"], 0.0)
            self.assertIsNone(steps[0]["bandwidth"])
            self.assertFalse(steps[0]["probe_computed"])
            self.assertAlmostEqual(steps[1]["interaction"], 0.5)
            self.assertEqual(steps[2]["interaction"], 1.0)
            self.assertEqual(steps[2]["task_gradient_kind"], "sft_plus_weighted_dpp")


if __name__ == "__main__":
    unittest.main()
