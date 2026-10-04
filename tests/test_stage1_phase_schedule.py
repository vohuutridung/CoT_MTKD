from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.utils.data import DistributedSampler
from transformers import Qwen2Config, Qwen2ForCausalLM

from cot_mtkd.data.schema import PreparedRecord
from cot_mtkd.models.chunked_head import full_vocab_probe
from cot_mtkd.stage1.rbf import effective_update_distances
from cot_mtkd.stage1.trainer import (
    one_pass_expert_gradients,
    replay_expert_gradients,
    compose_objective_gradients,
    train_stage1,
)
from cot_mtkd.utils.distributed import DistributedContext
from cot_mtkd.utils.manifest import read_json as disk_read_json


def prepared_record(index: int) -> PreparedRecord:
    ids = list(range(1, 13))
    regions = [0, 0, 1] + [2] * 8 + [6]
    steps = [-1, -1, -1] + [0] * 4 + [1] * 4 + [-1]
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
        solution="",
        deepseek_grade=None,
        original_length=len(ids),
        kept_length=len(ids),
        original_steps=2,
        kept_steps=2,
        truncated=False,
        answer_start=regions.index(6),
        reasoning_start=regions.index(2),
        tokenizer_fingerprint="test-tokenizer",
    )


class Stage1PhaseScheduleTest(unittest.TestCase):
    def test_every_update_is_full_in_both_forward_modes(self) -> None:
        for mode in ("one_pass", "two_pass"):
            with self.subTest(mode=mode):
                self._assert_full_training(mode)

    def test_single_update_starts_with_full_interaction(self) -> None:
        self._assert_full_training("one_pass", record_count=2)

    def test_full_interaction_handles_windows_without_reasoning(self) -> None:
        self._assert_full_training("one_pass", record_count=2, no_reasoning=True)

    def _assert_full_training(
        self, mode: str, record_count: int = 12, no_reasoning: bool = False
    ) -> None:
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
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        base.enable_input_require_grads()
        base.config.use_cache = False
        records = [prepared_record(index) for index in range(record_count)]
        if no_reasoning:
            for record in records:
                record.region_ids = [1 if region == 2 else region for region in record.region_ids]
                record.step_ids = [-1] * len(record.step_ids)
                record.original_steps = record.kept_steps = 0
        total_steps = record_count // 2
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
                    "forward_mode": mode,
                    "epochs": 1,
                    "learning_rate": 1.0e-3,
                    "micro_batch_size": 1,
                    "global_batch_size": 2,
                    "gradient_accumulation_steps": None,
                    "dpp_weight": 0.2,
                    "rbf_weight": 0.01,
                    "interaction_mode": "full",
                    "max_grad_norm": 1.0,
                    "checkpoint_every_steps": 100,
                    "log_every_steps": 1,
                    "resume_from": None,
                },
                "kneedle": {"search_k": 512, "k_min": 8},
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
                        manifest if Path(path).parent == root / "prepared" else disk_read_json(path)
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
                patch("cot_mtkd.stage1.trainer.JsonlRecordDataset", return_value=records),
                patch(
                    "cot_mtkd.stage1.trainer.full_vocab_probe", wraps=full_vocab_probe
                ) as probe_calls,
                patch(
                    "cot_mtkd.stage1.trainer.effective_update_distances",
                    wraps=effective_update_distances,
                ) as rbf_calls,
                patch(
                    "cot_mtkd.stage1.trainer.one_pass_expert_gradients",
                    wraps=one_pass_expert_gradients,
                ) as one_pass,
                patch(
                    "cot_mtkd.stage1.trainer.replay_expert_gradients",
                    wraps=replay_expert_gradients,
                ) as replay,
                patch(
                    "cot_mtkd.stage1.trainer.compose_objective_gradients",
                    wraps=compose_objective_gradients,
                ) as objective,
            ):
                result = train_stage1(config, distributed)
            self.assertEqual(probe_calls.call_count, 0 if no_reasoning else record_count * 3)
            self.assertEqual(rbf_calls.call_count, total_steps)
            self.assertEqual(objective.call_count, total_steps)
            for call in objective.call_args_list:
                self.assertEqual(len(call.args[1]), 3)
                self.assertEqual(call.args[2], config["stage1"]["rbf_weight"])
            active_calls = one_pass if mode == "one_pass" else replay
            self.assertEqual(
                active_calls.call_count, record_count * (1 if mode == "one_pass" else 3)
            )
            for call in active_calls.call_args_list:
                self.assertIsNotNone(call.kwargs["combined_dpp_scale"])
                if no_reasoning:
                    self.assertEqual(call.kwargs["combined_dpp_scale"], 0.0)
                else:
                    self.assertGreater(call.kwargs["combined_dpp_scale"], 0.0)
            self.assertEqual(result["global_step"], total_steps)
            with (root / "output" / "metrics.jsonl").open(encoding="utf-8") as handle:
                steps = [json.loads(line) for line in handle if '"event": "stage1_step"' in line]
            self.assertEqual(
                [item["interaction_phase"] for item in steps],
                ["full"] * total_steps,
            )
            self.assertEqual(len(steps), total_steps)
            for step in steps:
                self.assertEqual(step["interaction"], 1.0)
                self.assertEqual(step["task_gradient_kind"], "sft_plus_weighted_dpp")
                self.assertIsNotNone(step["bandwidth"])
                self.assertTrue(step["probe_computed"])
                self.assertTrue(step["rbf_computed"])
                self.assertEqual(step["objective"], "sft_dpp_rbf")
                self.assertNotIn("repulsion_cap_factors", step)
                self.assertEqual(len(step["rbf_task_gradient_ratios"]), 3)
                self.assertGreaterEqual(step["rbf_loss"], 0.0)
                self.assertLessEqual(step["rbf_loss"], 1.0)
                self.assertAlmostEqual(step["phase1_loss"], 3 * step["sft_nll"] + 0.2 * step["dpp_loss"] + 0.01 * step["rbf_loss"])
                if no_reasoning:
                    self.assertEqual(step["dpp_loss"], 0.0)
                else:
                    self.assertGreater(step["dpp_loss"], 0.0)
            order = list(DistributedSampler(records, num_replicas=1, rank=0, seed=42))
            for window, step in enumerate(steps):
                # Count the actual shuffled samples in this accumulation window;
                # unequal trace lengths must not become a mean of batch means.
                expected_count = 3 * sum(
                    records[index].region_ids.count(2)
                    for index in order[2 * window : 2 * window + 2]
                )
                self.assertEqual(step["support_selection_count"], expected_count)
                for field, mean in (
                    ("raw_k_histogram", "mean_raw_k"),
                    ("selected_k_histogram", "mean_selected_k"),
                ):
                    histogram = step[field]
                    self.assertEqual(len(histogram), 513)
                    self.assertEqual(sum(histogram), expected_count)
                    self.assertEqual(histogram[0], 0)
                    # This tiny model has only 63 non-target candidates even
                    # though search_k is 512; no bin may exceed that window.
                    self.assertEqual(sum(histogram[64:]), 0)
                    self.assertAlmostEqual(
                        step[mean],
                        sum(k * count for k, count in enumerate(histogram))
                        / max(expected_count, 1),
                        places=5,
                    )
                self.assertEqual(sum(step["selected_k_histogram"][:8]), 0)


if __name__ == "__main__":
    unittest.main()
