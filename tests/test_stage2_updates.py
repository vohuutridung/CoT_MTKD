from __future__ import annotations

import copy
import json
import math
import random
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from test_stage2_online import LORA, CharacterTokenizer, tiny_online_council, two_step_record
from transformers import Qwen2Config, Qwen2ForCausalLM

from cot_mtkd.data.prepare import tokenizer_fingerprint
from cot_mtkd.models.multi_adapter import adapter_parameter_map, extract_adapter_state
from cot_mtkd.stage2.council_cache import build_council_cache
from cot_mtkd.stage2.trainer import (
    METHOD,
    OUTPUT_SPACE_METHOD,
    _load_checkpoint,
    _save_checkpoint,
    apply_accumulated_update,
    train_stage2,
)
from cot_mtkd.utils.distributed import DistributedContext
from cot_mtkd.utils.manifest import file_sha256, fingerprint, write_config_snapshot, write_json


class TinyAdapterModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base = torch.nn.Parameter(torch.tensor([0.75]), requires_grad=False)
        self.lora_A = torch.nn.ModuleDict(
            {
                "student": torch.nn.Linear(2, 1, bias=False),
                "expert_0": torch.nn.Linear(2, 1, bias=False),
            }
        )
        self.lora_B = torch.nn.ModuleDict(
            {
                "student": torch.nn.Linear(1, 2, bias=False),
                "expert_0": torch.nn.Linear(1, 2, bias=False),
            }
        )
        with torch.no_grad():
            self.lora_A["student"].weight.copy_(torch.tensor([[0.6, -0.2]]))
            self.lora_B["student"].weight.copy_(torch.tensor([[0.3], [-0.4]]))
            self.lora_A["expert_0"].weight.fill_(0.4)
            self.lora_B["expert_0"].weight.fill_(-0.1)
        for name, value in self.named_parameters():
            value.requires_grad_(".student." in name)


def optimizer_and_scheduler(parameters):
    optimizer = torch.optim.AdamW(parameters, lr=0.1, weight_decay=0.2, eps=1.0e-8)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 0.8**step)
    return optimizer, scheduler


class Stage2UpdateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.thread_count = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.thread_count)

    def assert_state_equal(self, actual, expected):
        if isinstance(actual, torch.Tensor):
            torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)
        elif isinstance(actual, dict):
            self.assertEqual(actual.keys(), expected.keys())
            for key in actual:
                self.assert_state_equal(actual[key], expected[key])
        elif isinstance(actual, (list, tuple)):
            self.assertEqual(len(actual), len(expected))
            for a, b in zip(actual, expected, strict=True):
                self.assert_state_equal(a, b)
        else:
            self.assertEqual(actual, expected)

    def test_zero_signal_window_keeps_existing_adam_momentum_and_scheduler_unchanged(self):
        parameter = torch.nn.Parameter(torch.tensor([0.4, -0.8]))
        optimizer, scheduler = optimizer_and_scheduler([parameter])
        updated, _ = apply_accumulated_update(
            [parameter],
            [torch.tensor([2.0, -3.0])],
            optimizer,
            scheduler,
            sample_count=1,
            active_steps=1,
            max_grad_norm=100.0,
        )
        self.assertTrue(updated)
        before_parameter = parameter.detach().clone()
        before_optimizer = copy.deepcopy(optimizer.state_dict())
        before_scheduler = copy.deepcopy(scheduler.state_dict())
        self.assertGreater(optimizer.state[parameter]["exp_avg"].abs().sum().item(), 0.0)
        updated, norm = apply_accumulated_update(
            [parameter],
            [torch.zeros_like(parameter)],
            optimizer,
            scheduler,
            sample_count=4,
            active_steps=0,
            max_grad_norm=100.0,
        )
        self.assertFalse(updated)
        self.assertEqual(norm, 0.0)
        self.assert_state_equal(parameter, before_parameter)
        self.assert_state_equal(optimizer.state_dict(), before_optimizer)
        self.assert_state_equal(scheduler.state_dict(), before_scheduler)
        self.assertIsNone(parameter.grad)

    def test_sample_mean_includes_zero_signal_examples_before_global_clipping(self):
        parameter = torch.nn.Parameter(torch.tensor([0.4, -0.8]))
        reference_parameter = torch.nn.Parameter(parameter.detach().clone())
        optimizer, scheduler = optimizer_and_scheduler([parameter])
        reference_optimizer, reference_scheduler = optimizer_and_scheduler([reference_parameter])
        gradient_sum = torch.tensor([4.0, -8.0])
        expected_mean = gradient_sum / 4
        expected_norm = torch.linalg.vector_norm(expected_mean).item()
        expected_clipped = expected_mean * (0.5 / (expected_norm + 1.0e-12))
        reference_parameter.grad = expected_clipped
        reference_optimizer.step()
        reference_scheduler.step()
        reference_optimizer.zero_grad(set_to_none=True)
        updated, norm = apply_accumulated_update(
            [parameter],
            [gradient_sum],
            optimizer,
            scheduler,
            sample_count=4,
            active_steps=1,
            max_grad_norm=0.5,
        )
        self.assertTrue(updated)
        self.assertAlmostEqual(norm, math.sqrt(5.0), places=6)
        self.assert_state_equal(parameter, reference_parameter)
        self.assert_state_equal(optimizer.state_dict(), reference_optimizer.state_dict())
        self.assert_state_equal(scheduler.state_dict(), reference_scheduler.state_dict())
        torch.testing.assert_close(gradient_sum, expected_clipped, atol=0.0, rtol=0.0)

    def test_nonfinite_batch_is_rejected_before_any_optimizer_change(self):
        parameter = torch.nn.Parameter(torch.tensor([0.4, -0.8]))
        optimizer, scheduler = optimizer_and_scheduler([parameter])
        before_parameter = parameter.detach().clone()
        before_optimizer = copy.deepcopy(optimizer.state_dict())
        before_scheduler = copy.deepcopy(scheduler.state_dict())
        with self.assertRaisesRegex(FloatingPointError, "Non-finite accumulated"):
            apply_accumulated_update(
                [parameter],
                [torch.tensor([float("nan"), 1.0])],
                optimizer,
                scheduler,
                sample_count=2,
                active_steps=1,
                max_grad_norm=1.0,
            )
        self.assert_state_equal(parameter, before_parameter)
        self.assert_state_equal(optimizer.state_dict(), before_optimizer)
        self.assert_state_equal(scheduler.state_dict(), before_scheduler)

    def test_checkpoint_resume_matches_uninterrupted_updates_and_restores_data_cursor_and_RNG(self):
        random.seed(881)
        np.random.seed(881)
        torch.manual_seed(881)
        model = TinyAdapterModel()
        parameters = list(adapter_parameter_map(model, "student").values())
        optimizer, scheduler = optimizer_and_scheduler(parameters)
        frozen_before = {
            name: value.detach().clone()
            for name, value in model.named_parameters()
            if not value.requires_grad
        }
        first = [torch.ones_like(value) * 0.2 for value in parameters]
        apply_accumulated_update(parameters, first, optimizer, scheduler, 2, 1, 1.0)
        apply_accumulated_update(
            parameters,
            [torch.zeros_like(value) for value in parameters],
            optimizer,
            scheduler,
            2,
            0,
            1.0,
        )
        metrics = {"loss": 0.25, "active_steps": 1}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            _save_checkpoint(
                path,
                model,
                optimizer,
                scheduler,
                global_step=1,
                data_step=2,
                epoch=0,
                batch_in_epoch=4,
                run_fingerprint="test-run",
                base_seed=42,
                world_size=1,
                metrics=metrics,
            )
            expected_draws = (random.random(), np.random.rand(), torch.rand(3))
            second = [torch.ones_like(value) * -0.3 for value in parameters]
            apply_accumulated_update(parameters, second, optimizer, scheduler, 4, 2, 1.0)
            apply_accumulated_update(
                parameters,
                [torch.zeros_like(value) for value in parameters],
                optimizer,
                scheduler,
                1,
                0,
                1.0,
            )
            expected_student = extract_adapter_state(model, "student")
            expected_optimizer = copy.deepcopy(optimizer.state_dict())
            expected_scheduler = copy.deepcopy(scheduler.state_dict())

            resumed_model = TinyAdapterModel()
            resumed_parameters = list(adapter_parameter_map(resumed_model, "student").values())
            resumed_optimizer, resumed_scheduler = optimizer_and_scheduler(resumed_parameters)
            cursor = _load_checkpoint(
                path, resumed_model, resumed_optimizer, resumed_scheduler, "test-run"
            )
            self.assertEqual(cursor, (1, 2, 0, 4, metrics))
            actual_draws = (random.random(), np.random.rand(), torch.rand(3))
            self.assertEqual(actual_draws[:2], expected_draws[:2])
            self.assert_state_equal(actual_draws[2], expected_draws[2])
            resumed_second = [torch.ones_like(value) * -0.3 for value in resumed_parameters]
            apply_accumulated_update(
                resumed_parameters, resumed_second, resumed_optimizer, resumed_scheduler, 4, 2, 1.0
            )
            apply_accumulated_update(
                resumed_parameters,
                [torch.zeros_like(value) for value in resumed_parameters],
                resumed_optimizer,
                resumed_scheduler,
                1,
                0,
                1.0,
            )
            self.assert_state_equal(
                extract_adapter_state(resumed_model, "student"), expected_student
            )
            self.assert_state_equal(resumed_optimizer.state_dict(), expected_optimizer)
            self.assert_state_equal(resumed_scheduler.state_dict(), expected_scheduler)
            for name, value in resumed_model.named_parameters():
                if name in frozen_before:
                    self.assert_state_equal(value, frozen_before[name])

    def test_checkpoint_from_another_method_is_rejected_without_loading_student(self):
        model = TinyAdapterModel()
        parameters = list(adapter_parameter_map(model, "student").values())
        optimizer, scheduler = optimizer_and_scheduler(parameters)
        before = extract_adapter_state(model, "student")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old-checkpoint.pt"
            torch.save({"method": "old_dual_source", "run_fingerprint": "test-run"}, path)
            with self.assertRaisesRegex(RuntimeError, "different method or configuration"):
                _load_checkpoint(path, model, optimizer, scheduler, "test-run")
        self.assert_state_equal(extract_adapter_state(model, "student"), before)
        self.assertEqual(optimizer.state, {})

    def test_complete_council_training_export_and_completed_resume_on_tiny_Qwen(self):
        self.check_council_training_export_and_resume(METHOD)

    def test_output_space_council_training_export_and_completed_resume_on_tiny_Qwen(self):
        self.check_council_training_export_and_resume(OUTPUT_SPACE_METHOD)

    def check_council_training_export_and_resume(self, method):
        tokenizer = CharacterTokenizer()
        context = DistributedContext(0, 0, 1, torch.device("cpu"))
        initial_model, names, _ = tiny_online_council(checkpointing=True)

        def load_fixture_backbone(model_config, device):
            # Match the immutable backbone used to create the source council.
            with torch.random.fork_rng():
                torch.manual_seed(17)
                model = Qwen2ForCausalLM(
                    Qwen2Config(
                        vocab_size=64,
                        hidden_size=8,
                        intermediate_size=16,
                        num_hidden_layers=1,
                        num_attention_heads=2,
                        num_key_value_heads=1,
                        max_position_embeddings=128,
                        attention_dropout=0.0,
                    )
                )
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            model.enable_input_require_grads()
            return model.to(device)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared, stage1, council, output = [
                root / name for name in ("prepared", "stage1", "council", "stage2")
            ]
            prepared.mkdir()
            stage1.mkdir()
            records = [two_step_record(), two_step_record()]
            records[1].sample_id = "tiny-second-example"
            records[1].solution = "17"
            data_path = prepared / "verified-records.jsonl"
            data_path.write_text("".join(json.dumps(asdict(record)) + "\n" for record in records))
            # A directory-based loader prefers data.jsonl. This unverified decoy
            # must never override the explicitly owned and checked manifest file.
            decoy_path = prepared / "data.jsonl"
            decoy_path.write_text("This is not the manifest-owned training corpus.\n")
            decoy_hash = file_sha256(decoy_path)
            model_config = {
                "name_or_path": "fixture/model",
                "revision": "immutable-test-revision",
                "dtype": "float32",
                "gradient_checkpointing": True,
            }
            prepared_config = {"model": model_config, "tokenization": {"max_length": 128}}
            write_config_snapshot(prepared / "config.yaml", prepared_config)
            prepared_manifest = {
                "artifact": "prepared_dataset",
                "records": 2,
                "data_file": data_path.name,
                "data_file_sha256": file_sha256(data_path),
                "config_file": "config.yaml",
                "config_file_sha256": file_sha256(prepared / "config.yaml"),
                "tokenizer_fingerprint": tokenizer_fingerprint(tokenizer),
                "config": prepared_config,
            }
            write_json(prepared / "manifest.json", prepared_manifest)
            stage1_config = {"seed": 42, "model": model_config, "lora": LORA}
            write_config_snapshot(stage1 / "config.yaml", stage1_config)
            bundle_path = stage1 / "adapters.pt"
            torch.save(
                {name: extract_adapter_state(initial_model, name) for name in names}, bundle_path
            )
            stage1_manifest = {
                "artifact": "stage1_checkpoint",
                "adapter_names": names,
                "adapter_bundle": bundle_path.name,
                "adapter_bundle_sha256": file_sha256(bundle_path),
                "config_file": "config.yaml",
                "config_file_sha256": file_sha256(stage1 / "config.yaml"),
                "prepared_manifest_fingerprint": fingerprint(prepared_manifest),
                "tokenizer_fingerprint": tokenizer_fingerprint(tokenizer),
                "config": stage1_config,
            }
            write_json(stage1 / "manifest.json", stage1_manifest)
            config = {
                "method": method,
                "seed": 42,
                "model": model_config,
                "lora": LORA,
                "geometry": {
                    "temperature": 2.0,
                    "epsilon_a": 1.0e-12,
                    "epsilon_u": 1.0e-12,
                    "teacher_execution": "online_full_vocab",
                },
                "stage2": {
                    "epochs": 1,
                    "micro_batch_size": 1,
                    "global_batch_size": 2,
                    "gradient_accumulation_steps": 2,
                    "learning_rate": 2.0e-4,
                    "max_length": 128,
                    "max_grad_norm": 1.0,
                    "checkpoint_every_steps": 1,
                    "log_every_steps": 1,
                    "resume_from": None,
                },
                "optimizer": {
                    "name": "adamw",
                    "betas": [0.9, 0.999],
                    "eps": 1.0e-8,
                    "weight_decay": 0.0,
                },
                "scheduler": {"name": "cosine", "warmup_ratio": 0.1, "min_lr_ratio": 0.0},
                "runtime": {"lm_head_chunk_tokens": 2, "dataloader_workers": 0},
                "paths": {
                    "prepared": str(prepared),
                    "stage1": str(stage1),
                    "teacher_cache_dir": str(council),
                    "output": str(output),
                },
                "_project_root": str(root),
            }
            config["aggregation"] = {
                "js_temperature": 1.0,
                "kd_temperature": 2.0,
                "sft_weight": 0.25,
                "search_k": 512,
                "k_min": 8,
                "teacher_execution": "precomputed_support_tail",
            }
            if method == OUTPUT_SPACE_METHOD:
                del config["geometry"]
            with (
                patch(
                    "cot_mtkd.models.multi_adapter.load_base_causal_lm",
                    side_effect=load_fixture_backbone,
                ),
                patch("cot_mtkd.stage2.council_cache.load_tokenizer", return_value=tokenizer),
                patch("cot_mtkd.stage2.trainer.load_tokenizer", return_value=tokenizer),
            ):
                council_manifest = build_council_cache(config, context)
                with patch(
                    "cot_mtkd.stage2.council_cache.create_multi_adapter_model",
                    side_effect=AssertionError("Cache hit must not load teachers"),
                ):
                    hit = build_council_cache(config, context)
                self.assertEqual(hit, council_manifest)
                if method == OUTPUT_SPACE_METHOD:
                    with (
                        patch(
                            "cot_mtkd.stage2.council_cache.compile_record",
                            side_effect=AssertionError("Training must not preprocess teachers"),
                        ),
                        patch(
                            "cot_mtkd.stage2.online.create_multi_adapter_model",
                            side_effect=AssertionError("Training must load only student"),
                        ),
                    ):
                        manifest = train_stage2(config, context)
                else:
                    manifest = train_stage2(config, context)
                self.assertEqual(manifest["method"], method)
                self.assertEqual(manifest["data_step"], 1)
                self.assertEqual(manifest["global_step"], 1)
                self.assertEqual(manifest["loss_scalars"]["examples"], 2)
                self.assertEqual(manifest["loss_scalars"]["reasoning_steps"], 4)
                self.assertEqual(
                    manifest["initial_expert_adapter"],
                    None if method == OUTPUT_SPACE_METHOD else council_manifest["selected_expert"],
                )
                bundle = torch.load(output / manifest["adapter_bundle"], weights_only=True)
                self.assertEqual(set(bundle), {"student"})
                checkpoint = torch.load(output / "checkpoint.pt", weights_only=False)
                self.assertEqual(checkpoint["method"], method)
                if method == OUTPUT_SPACE_METHOD:
                    self.assertEqual(manifest["loss_scalars"]["active_steps"], 4)
                    self.assertIn("disagreement_mean", manifest["loss_scalars"])
                    self.assertNotIn("anchor_ce_mean", manifest["loss_scalars"])
                    step_log = output / "reasoning_steps.jsonl"
                    rows = [json.loads(line) for line in step_log.read_text().splitlines()]
                    self.assertEqual(len(rows), 4)
                    self.assertEqual(
                        {row["sample_id"] for row in rows}, {record.sample_id for record in records}
                    )
                    self.assertEqual({row["step_id"] for row in rows}, {0, 1})
                    for row in rows:
                        self.assertEqual(row["run_fingerprint"], manifest["run_fingerprint"])
                        self.assertEqual(row["data_step_before"], 0)
                        self.assertEqual(row["global_step_before"], 0)
                        self.assertAlmostEqual(row["js_mean"], row["js_normalized"] * math.log(3))
                        # ECDF rho is the step's rank among the corpus step JS values.
                        reference = council_manifest["diagnostics"]["js"]
                        self.assertGreater(row["rho"], 0.0)
                        self.assertLessEqual(row["rho"], 1.0)
                        self.assertEqual(reference["count"], 4)
                    for record in records:
                        sample_rows = [row for row in rows if row["sample_id"] == record.sample_id]
                        self.assertEqual(len(sample_rows), 2)
                        # Token normalization: the sample KD is the token-weighted step mean.
                        self.assertAlmostEqual(
                            sum(row["step_kd_loss"] * row["n_tokens"] for row in sample_rows)
                            / sum(row["n_tokens"] for row in sample_rows),
                            sample_rows[0]["sample_kd_loss"],
                        )
                    self.assertEqual(
                        manifest["reasoning_step_logs"],
                        [
                            {
                                "rank": 0,
                                "file": "reasoning_steps.jsonl",
                                "sha256": file_sha256(step_log),
                            }
                        ],
                    )
                    performance_log = output / "performance.jsonl"
                    perf_rows = [
                        json.loads(line) for line in performance_log.read_text().splitlines()
                    ]
                    self.assertEqual(perf_rows[0]["event"], "stage2_performance_session")
                    self.assertEqual(
                        perf_rows[0]["config_fingerprint"], manifest["config_fingerprint"]
                    )
                    samples = [
                        row for row in perf_rows if row["event"] == "stage2_sample_performance"
                    ]
                    updates = [
                        row for row in perf_rows if row["event"] == "stage2_update_performance"
                    ]
                    self.assertEqual(len(samples), 2)
                    self.assertEqual(len(updates), 1)
                    self.assertEqual(
                        {row["sample_id"] for row in samples}, {r.sample_id for r in records}
                    )
                    for row in samples:
                        self.assertEqual(row["prefix_tokens"], 10)
                        self.assertEqual(row["reasoning_tokens"], 5)
                        self.assertEqual(row["cached_steps"], 2)
                        self.assertEqual(row["recomputed_steps"], 0)
                        self.assertEqual(row["head_chunks"], 3)
                        self.assertEqual(row["teacher_head_chunk_sweeps"], 0)
                        self.assertIsNone(row["peak_allocated_gib"])
                        self.assertGreaterEqual(row["record_gradient_wall_seconds"], 0)
                    self.assertEqual(updates[0]["local_examples"], 2)
                    self.assertEqual(updates[0]["local_prefix_tokens"], 20)
                    self.assertEqual(updates[0]["eta_remaining_wall_seconds_estimate"], 0)
                    self.assertEqual(
                        manifest["performance_logs"],
                        [
                            {
                                "rank": 0,
                                "file": "performance.jsonl",
                                "sha256": file_sha256(performance_log),
                            }
                        ],
                    )
                self.assert_state_equal(bundle["student"], checkpoint["student_state"])
                self.assertTrue(
                    (output / "final" / "adapters" / "student" / "adapter_config.json").is_file()
                )
                self.assertEqual(file_sha256(data_path), prepared_manifest["data_file_sha256"])
                self.assertEqual(file_sha256(decoy_path), decoy_hash)
                self.assertEqual(file_sha256(bundle_path), stage1_manifest["adapter_bundle_sha256"])
                checkpoint_hash = file_sha256(output / "checkpoint.pt")
                metrics_hash = file_sha256(output / "metrics.jsonl")
                resume_config = copy.deepcopy(config)
                resume_config["stage2"]["resume_from"] = str(output / "checkpoint.pt")
                gradient_path = (
                    "cot_mtkd.stage2.output_space.compute_record_gradient"
                    if method == OUTPUT_SPACE_METHOD
                    else "cot_mtkd.stage2.trainer.compute_record_gradient"
                )
                with patch(
                    gradient_path,
                    side_effect=AssertionError("A completed resume must not retrain examples"),
                ):
                    resumed_manifest = train_stage2(resume_config, context)
                self.assertEqual(resumed_manifest, manifest)
                self.assertEqual(file_sha256(output / "checkpoint.pt"), checkpoint_hash)
                self.assertEqual(file_sha256(output / "metrics.jsonl"), metrics_hash)
                if method == OUTPUT_SPACE_METHOD:
                    self.assertEqual(
                        file_sha256(step_log), manifest["reasoning_step_logs"][0]["sha256"]
                    )
                    self.assertEqual(
                        file_sha256(performance_log), manifest["performance_logs"][0]["sha256"]
                    )


if __name__ == "__main__":
    unittest.main()
