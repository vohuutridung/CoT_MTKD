from __future__ import annotations

import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from cot_mtkd.cli.stress_stage2_memory import (
    GIB,
    _h200_requirement,
    _optimizer_and_scheduler,
    _record_provenance,
    _run_microbatch,
    longest_eligible_record,
    main,
    make_synthetic_record,
    memory_recommendation,
    run_case,
)
from cot_mtkd.data.schema import PreparedRecord
from cot_mtkd.data.serialize import pag_answer_prefix
from cot_mtkd.models.multi_adapter import (
    adapter_parameter_map,
    create_multi_adapter_model,
    extract_adapter_state,
    load_adapter_state,
    lora_config,
    set_active_adapter,
)
from cot_mtkd.stage2.online import RecordGradientResult, compute_record_gradient, plan_record
from cot_mtkd.stage2.trainer import METHOD, _stable_config
from cot_mtkd.utils.manifest import file_sha256, fingerprint, read_json


class TinyTokenizer:
    def __call__(self, text: str, **kwargs) -> dict:
        prefix = pag_answer_prefix()
        assert text.startswith(prefix)
        gold = text[len(prefix) :]
        return {
            "input_ids": [20] + [21 + index % 8 for index in range(len(gold))],
            "offset_mapping": [(0, len(prefix))]
            + [(len(prefix) + index, len(prefix) + index + 1) for index in range(len(gold))],
        }


def stress_record() -> PreparedRecord:
    ids = list(range(1, 15))
    regions = [0, 0, 1] + [2] * 3 + [3] * 2 + [2] * 2 + [3, 4, 5, 6]
    return PreparedRecord(
        sample_id="tiny-phase2-stress",
        input_ids=ids,
        labels=[-100 if region == 0 else token for token, region in zip(ids, regions)],
        attention_mask=[1] * len(ids),
        offset_mapping=[(0, 0)] * len(ids),
        region_ids=regions,
        step_ids=[-1] * 3 + [0] * 5 + [1] * 3 + [-1] * 3,
        question="question",
        thinking="first\n\nsecond",
        attempt="non-gold attempt",
        solution="gold",
        deepseek_grade=None,
        original_length=len(ids),
        kept_length=len(ids),
        original_steps=2,
        kept_steps=2,
        truncated=False,
        answer_start=11,
        reasoning_start=3,
        tokenizer_fingerprint="tiny-test",
    )


def stress_config() -> dict:
    return {
        "seed": 42,
        "model": {"gradient_checkpointing": True},
        "lora": {"dropout": 0.0},
        "geometry": {"temperature": 2.0, "epsilon_a": 1.0e-12, "epsilon_u": 1.0e-12},
        "stage2": {
            "epochs": 3,
            "max_length": 32,
            "learning_rate": 2.0e-5,
            "micro_batch_size": 1,
            "global_batch_size": 32,
            "max_grad_norm": 1.0,
        },
        "runtime": {"lm_head_chunk_tokens": 4},
        "optimizer": {"name": "adamw", "betas": [0.9, 0.999], "eps": 1.0e-8, "weight_decay": 0.1},
        "scheduler": {"name": "cosine", "warmup_ratio": 0.1, "min_lr_ratio": 0.0},
    }


def result_with_signal(active: bool = True) -> RecordGradientResult:
    return RecordGradientResult(
        gradients=[torch.tensor([0.5 if active else 0.0])],
        loss=0.1 if active else 0.0,
        steps=2,
        active_steps=1 if active else 0,
        discarded_steps=0,
        metrics={"weight_sum": 0.4 if active else 0.0},
        timings={"teacher_forward": 0.25, "answer_anchor_gradients": 0.5},
    )


class Stage2StressTest(unittest.TestCase):
    def test_tiny_checkpointed_council_runs_real_online_method_and_optimizer(self) -> None:
        thread_count = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, thread_count)
        torch.manual_seed(17)
        base = Qwen2ForCausalLM(
            Qwen2Config(
                vocab_size=64,
                hidden_size=8,
                intermediate_size=16,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=1,
                max_position_embeddings=32,
                attention_dropout=0.0,
            )
        )
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        base.enable_input_require_grads()
        config = stress_config()
        config["lora"].update(
            rank=2,
            alpha=2,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )
        with patch("cot_mtkd.models.multi_adapter.load_base_causal_lm", return_value=base):
            model, names = create_multi_adapter_model(
                {}, config["lora"], 3, torch.device("cpu"), 42
            )
        model.add_adapter("student", lora_config(config["lora"]))
        generator = torch.Generator().manual_seed(703)
        with torch.no_grad():
            for name in names:
                for key, parameter in adapter_parameter_map(model, name).items():
                    if "lora_B" in key:
                        parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.13)
            load_adapter_state(model, "student", extract_adapter_state(model, names[0]))
        set_active_adapter(model, "student")
        parameters = list(adapter_parameter_map(model, "student").values())
        before_student = [parameter.detach().clone() for parameter in parameters]
        before_teachers = {name: extract_adapter_state(model, name) for name in names}
        optimizer, scheduler = _optimizer_and_scheduler(parameters, config, 10)
        with (
            patch("torch.cuda.synchronize"),
            patch(
                "cot_mtkd.cli.stress_stage2_memory.compute_record_gradient",
                wraps=compute_record_gradient,
            ) as compute,
        ):
            details = _run_microbatch(
                stress_record(),
                TinyTokenizer(),
                model,
                names,
                parameters,
                optimizer,
                scheduler,
                config,
                torch.device("cpu"),
            )
        self.assertEqual(compute.call_count, 1)
        self.assertEqual(details["steps"], 2)
        self.assertGreater(details["active_steps"], 0)
        self.assertTrue(details["optimizer_update_applied"])
        self.assertEqual(len(optimizer.state), len(parameters))
        self.assertTrue(
            any(not torch.equal(a, b) for a, b in zip(before_student, parameters, strict=True))
        )
        self.assertTrue(all(parameter.grad is None for parameter in parameters))
        for name in names:
            for key, value in extract_adapter_state(model, name).items():
                torch.testing.assert_close(value, before_teachers[name][key], rtol=0.0, atol=0.0)
        for component in (
            "teacher_forward",
            "teacher_kd_gradients",
            "answer_anchor_gradients",
            "geometry",
            "final_kd_gradient",
        ):
            self.assertIn(component, details["component_seconds"])

    def test_synthetic_extends_only_final_step_to_gold_anchor_limit(self) -> None:
        record, tokenizer = stress_record(), TinyTokenizer()
        original = plan_record(record, tokenizer, 32)
        synthetic, plan = make_synthetic_record(record, tokenizer, 32)
        self.assertEqual(synthetic.solution, record.solution)
        self.assertEqual(synthetic.attempt, record.attempt)
        self.assertEqual(plan.solution, original.solution)
        self.assertEqual(plan.num_steps, 2)
        self.assertEqual(plan.discarded_steps, 0)
        self.assertEqual(plan.max_anchor_length, 32)
        self.assertEqual(plan.step_positions[0], original.step_positions[0])
        extra = 32 - original.max_anchor_length
        self.assertEqual(len(plan.step_positions[-1]), len(original.step_positions[-1]) + extra)
        self.assertEqual(
            synthetic.input_ids[synthetic.answer_start :], record.input_ids[record.answer_start :]
        )
        self.assertEqual(synthetic.region_ids[plan.prefix_ends[-1] - 1], 3)
        self.assertTrue(
            all(
                synthetic.region_ids[position] == 2
                for positions in plan.step_positions
                for position in positions
            )
        )

    def test_synthetic_removes_discarded_steps_before_extension(self) -> None:
        record, tokenizer = stress_record(), TinyTokenizer()
        before = plan_record(record, tokenizer, 15)
        self.assertEqual(before.num_steps, 1)
        self.assertEqual(before.discarded_steps, 1)
        synthetic, plan = make_synthetic_record(record, tokenizer, 15)
        self.assertEqual(plan.num_steps, 1)
        self.assertEqual(plan.max_anchor_length, 15)
        self.assertEqual(plan.discarded_steps, 0)
        self.assertEqual(synthetic.solution, "gold")
        self.assertNotIn(1, synthetic.step_ids)

    def test_longest_selection_includes_whole_gold_solution_length(self) -> None:
        record = stress_record()
        longer_gold = replace(record, sample_id="longer-gold", solution="entire gold solution")
        selected, plan, counts = longest_eligible_record([record, longer_gold], TinyTokenizer(), 64)
        self.assertEqual(selected.sample_id, "longer-gold")
        self.assertEqual(plan.max_anchor_length, 11 + 1 + len(longer_gold.solution))
        self.assertEqual(counts["eligible_examples"], 2)

    def test_full_result_is_applied_by_real_adamw_update(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        config = stress_config()
        optimizer, scheduler = _optimizer_and_scheduler([parameter], config, 10)
        before = parameter.detach().clone()
        with (
            patch("torch.cuda.synchronize"),
            patch(
                "cot_mtkd.cli.stress_stage2_memory.compute_record_gradient",
                return_value=result_with_signal(),
            ) as compute,
        ):
            details = _run_microbatch(
                stress_record(),
                TinyTokenizer(),
                None,
                ["a", "b", "c"],
                [parameter],
                optimizer,
                scheduler,
                config,
                torch.device("cpu"),
            )
        self.assertTrue(compute.call_args.kwargs["profile"])
        self.assertEqual(compute.call_args.args[2], [parameter])
        self.assertEqual(compute.call_args.args[3].solution, "gold")
        self.assertTrue(details["optimizer_update_applied"])
        self.assertFalse(torch.equal(before, parameter))
        self.assertIn(parameter, optimizer.state)
        self.assertIsNone(parameter.grad)
        self.assertIn("optimizer_update_seconds", details["component_seconds"])

    def test_zero_signal_does_not_apply_weight_decay_or_advance_scheduler(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        config = stress_config()
        optimizer, scheduler = _optimizer_and_scheduler([parameter], config, 10)
        epoch_before = scheduler.last_epoch
        with patch(
            "cot_mtkd.cli.stress_stage2_memory.compute_record_gradient",
            return_value=result_with_signal(False),
        ):
            details = _run_microbatch(
                stress_record(),
                TinyTokenizer(),
                None,
                ["a", "b", "c"],
                [parameter],
                optimizer,
                scheduler,
                config,
                torch.device("cpu"),
            )
        self.assertFalse(details["optimizer_update_applied"])
        self.assertEqual(parameter.item(), 1.0)
        self.assertEqual(scheduler.last_epoch, epoch_before)
        self.assertEqual(len(optimizer.state), 0)

    def test_warmup_component_times_are_separate_and_peak_includes_warmup(self) -> None:
        record = stress_record()
        plan = plan_record(record, TinyTokenizer(), 32)
        details = {
            "steps": 2,
            "optimizer_update_applied": True,
            "component_seconds": {"teacher_forward": 0.5},
        }
        with (
            patch("torch.cuda.synchronize"),
            patch("torch.cuda.reset_peak_memory_stats") as reset,
            patch("torch.cuda.max_memory_allocated", return_value=40 * GIB),
            patch("torch.cuda.max_memory_reserved", return_value=45 * GIB),
            patch(
                "cot_mtkd.cli.stress_stage2_memory._run_microbatch", return_value=details
            ) as compute,
            patch(
                "cot_mtkd.cli.stress_stage2_memory.time.perf_counter",
                side_effect=[0, 100, 100, 102, 102, 106],
            ),
        ):
            report = run_case(
                "longest_real",
                record,
                plan,
                None,
                None,
                [],
                [],
                None,
                None,
                stress_config(),
                torch.device("cpu"),
                warmup=1,
                repetitions=2,
            )
        self.assertEqual(reset.call_count, 1)
        self.assertEqual(compute.call_count, 3)
        self.assertEqual(report["warmup_elapsed_seconds"], [100])
        self.assertEqual(report["measured_elapsed_seconds"], [2, 4])
        self.assertEqual(report["mean_measured_elapsed_seconds"], 3)
        self.assertEqual(report["mean_measured_component_seconds"]["teacher_forward"], 0.5)
        self.assertEqual(report["max_memory_reserved_bytes"], 45 * GIB)
        self.assertTrue(report["memory_includes_warmup"])
        self.assertTrue(report["all_retained_steps_evaluated"])
        self.assertEqual(report["optimizer_updates"], 3)
        self.assertEqual(report["measured_optimizer_updates"], 2)
        self.assertFalse(report["timing_representative_of_training"])

    def test_warmup_oom_preserves_failure_and_has_no_measured_timing(self) -> None:
        record = stress_record()
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer, scheduler = _optimizer_and_scheduler([parameter], stress_config(), 10)
        with (
            patch("torch.cuda.synchronize"),
            patch("torch.cuda.reset_peak_memory_stats"),
            patch("torch.cuda.max_memory_allocated", return_value=130 * GIB),
            patch("torch.cuda.max_memory_reserved", return_value=139 * GIB),
            patch(
                "cot_mtkd.cli.stress_stage2_memory._run_microbatch",
                side_effect=torch.cuda.OutOfMemoryError("test"),
            ),
        ):
            report = run_case(
                "longest_real",
                record,
                plan_record(record, TinyTokenizer(), 32),
                None,
                None,
                [],
                [parameter],
                optimizer,
                scheduler,
                stress_config(),
                torch.device("cpu"),
                warmup=1,
                repetitions=1,
            )
        self.assertEqual(report["status"], "cuda_oom")
        self.assertEqual(report["failed_iteration_kind"], "warmup")
        self.assertEqual(report["completed_measured_iterations"], 0)
        self.assertIsNone(report["mean_measured_elapsed_seconds"])
        self.assertFalse(report["all_retained_steps_evaluated"])
        self.assertEqual(
            memory_recommendation([report], 141 * GIB, 12)[0], "direct_teacher_not_ready"
        )

    def test_memory_readiness_requires_two_complete_cases_and_actual_updates(self) -> None:
        case = {
            "status": "ok",
            "max_memory_reserved_bytes": 120 * GIB,
            "optimizer_updates": 1,
            "measured_optimizer_updates": 1,
            "all_retained_steps_evaluated": True,
        }
        self.assertEqual(memory_recommendation([], 141 * GIB, 12), ("not_measured", None))
        self.assertEqual(memory_recommendation([case], 141 * GIB, 12)[0], "incomplete_measurement")
        self.assertEqual(
            memory_recommendation([case, case], 141 * GIB, 12)[0], "direct_teacher_ready"
        )
        high = {**case, "max_memory_reserved_bytes": 135 * GIB}
        self.assertEqual(memory_recommendation([case, high], 141 * GIB, 12)[0], "review_headroom")
        skipped = {**case, "optimizer_updates": 1, "measured_optimizer_updates": 0}
        self.assertEqual(
            memory_recommendation([case, skipped], 141 * GIB, 12)[0], "inconclusive_zero_signal"
        )
        self.assertEqual(
            memory_recommendation(
                [case, {**case, "all_retained_steps_evaluated": False}], 141 * GIB, 12
            )[0],
            "incomplete_measurement",
        )

    def test_h200_requirement_rejects_absent_multiple_and_wrong_devices(self) -> None:
        with patch("torch.cuda.is_available", return_value=False):
            self.assertIsNone(_h200_requirement()[0])
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.device_count", return_value=2),
        ):
            self.assertIn("exactly one", _h200_requirement()[2])
        with (
            patch.dict("os.environ", {"WORLD_SIZE": "1"}),
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.device_count", return_value=1),
            patch("torch.cuda.get_device_name", return_value="NVIDIA A100"),
        ):
            self.assertIn("Expected H200", _h200_requirement()[2])

    def test_provenance_matches_training_and_rejects_changed_input_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = stress_config()
            config["paths"] = {
                key: str(Path(temporary) / key) for key in ("prepared", "stage1", "medoid")
            }
            manifests = {}
            for key, files in (
                (
                    "prepared",
                    [
                        ("data_file", "data_file_sha256", "data.jsonl"),
                        ("config_file", "config_file_sha256", "config.yaml"),
                    ],
                ),
                (
                    "stage1",
                    [
                        ("adapter_bundle", "adapter_bundle_sha256", "adapters.pt"),
                        ("config_file", "config_file_sha256", "config.yaml"),
                    ],
                ),
                ("medoid", [("config_file", "config_file_sha256", "config.yaml")]),
            ):
                root = Path(config["paths"][key])
                root.mkdir()
                manifests[key] = {}
                for file_key, hash_key, filename in files:
                    path = root / filename
                    path.write_text(f"{key} {file_key}", encoding="utf-8")
                    manifests[key].update({file_key: filename, hash_key: file_sha256(path)})
            manifests["medoid"].update(
                artifact="stage2_medoid",
                prepared_manifest_fingerprint=fingerprint(manifests["prepared"]),
                stage1_manifest_fingerprint=fingerprint(manifests["stage1"]),
            )
            report = {"config_fingerprint": fingerprint(_stable_config(config))}
            _record_provenance(config, report, manifests)
            self.assertEqual(report["source_provenance_status"], "verified")
            self.assertEqual(len(report["verified_source_files"]), 5)
            expected = fingerprint(
                {
                    "method": METHOD,
                    "config_fingerprint": report["config_fingerprint"],
                    "prepared_manifest_fingerprint": fingerprint(manifests["prepared"]),
                    "stage1_manifest_fingerprint": fingerprint(manifests["stage1"]),
                    "medoid_manifest_fingerprint": fingerprint(manifests["medoid"]),
                    "world_size": 1,
                }
            )
            self.assertEqual(report["run_fingerprint"], expected)
            (Path(config["paths"]["stage1"]) / "adapters.pt").write_text(
                "changed bytes", encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "content hash mismatch"):
                _record_provenance(config, {}, manifests)

    def test_missing_cuda_writes_requirement_report_without_loading_model(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "stress.json"
            with (
                patch.object(
                    sys, "argv", ["stress", "--config", "test.yaml", "--output", str(destination)]
                ),
                patch(
                    "cot_mtkd.cli.stress_stage2_memory.load_config", return_value=stress_config()
                ),
                patch("cot_mtkd.cli.stress_stage2_memory._validate_stage2_config") as validate,
                patch("cot_mtkd.cli.stress_stage2_memory.runtime_metadata", return_value={}),
                patch("torch.cuda.is_available", return_value=False),
                patch("cot_mtkd.cli.stress_stage2_memory.create_online_model") as create,
                self.assertRaises(SystemExit) as raised,
            ):
                main()
            report = read_json(destination)
        self.assertEqual(raised.exception.code, 2)
        validate.assert_called_once()
        create.assert_not_called()
        self.assertEqual(report["status"], "requirement_not_met")
        self.assertEqual(report["recommendation"], "not_measured")
        self.assertEqual(report["cases"], [])
        self.assertEqual(report["cases_completed"], 0)
        self.assertFalse(report["trained_artifact_written"])


if __name__ == "__main__":
    unittest.main()
