from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from cot_mtkd.cli.stress_stage1_memory import (
    GIB,
    PHASES,
    _run_microbatch,
    memory_recommendation,
    run_case,
)
from cot_mtkd.data.collator import LongCoTCollator, shifted_token_views
from cot_mtkd.data.schema import PreparedRecord
from cot_mtkd.models.multi_adapter import adapter_parameter_groups, create_multi_adapter_model
from cot_mtkd.stage1.gac_gradient import stable_gac_gradients
from cot_mtkd.stage1.rbf import BandwidthEMA
from cot_mtkd.stage1.trainer import (
    _optimizer_and_scheduler,
    one_pass_expert_gradients,
)


def stress_record() -> PreparedRecord:
    ids = list(range(1, 15))
    regions = [0, 0, 1] + [2] * 8 + [4, 5, 6]
    return PreparedRecord(
        sample_id="tiny-stress",
        input_ids=ids,
        labels=[-100 if region == 0 else token for token, region in zip(ids, regions)],
        attention_mask=[1] * len(ids),
        offset_mapping=[(0, 0)] * len(ids),
        region_ids=regions,
        step_ids=[-1, -1, -1] + [0] * 4 + [1] * 4 + [-1] * 3,
        question="",
        thinking="",
        solution="",
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
            "learning_rate": 1.0e-3,
            "global_batch_size": 16,
            "dpp_weight": 0.1,
            "rbf_weight": 0.5,
            "gac_beta": 0.5,
            "gac_bandwidth_scale": 0.5,
            "rbf_bandwidth_scale": 1.0,
            "max_grad_norm": 1.0,
        },
        "runtime": {"lm_head_chunk_tokens": 32768, "probe_hidden_device": "cpu"},
        "kneedle": {"search_k": 6, "k_min": 8},
        "dpp": {"jitter": 1.0e-4, "max_jitter": 1.0e-2},
        "optimizer": {"name": "adamw", "betas": [0.9, 0.999], "eps": 1.0e-8, "weight_decay": 0.0},
        "scheduler": {"name": "cosine", "warmup_ratio": 0.1, "min_lr_ratio": 0.0},
    }


class Stage1StressTest(unittest.TestCase):
    def test_tiny_qwen_runs_sft_only_and_full_interaction_with_checkpointing_and_dropout(self) -> None:
        torch.manual_seed(17)
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
        config = stress_config()
        with patch("cot_mtkd.models.multi_adapter.load_base_causal_lm", return_value=base):
            model, names = create_multi_adapter_model({}, config["lora"], 3, torch.device("cpu"))
        groups = adapter_parameter_groups(model, names)
        parameters = [list(group.values()) for group in groups]
        pairs = [_optimizer_and_scheduler(current, config, 6) for current in parameters]
        optimizers = [pair[0] for pair in pairs]
        schedulers = [pair[1] for pair in pairs]
        initial = [[value.detach().clone() for value in current] for current in parameters]
        record = stress_record()
        with (
            patch("torch.cuda.synchronize"),
            patch("torch.cuda.reset_peak_memory_stats") as resets,
            patch("torch.cuda.max_memory_allocated", return_value=42 * GIB),
            patch("torch.cuda.max_memory_reserved", return_value=45 * GIB),
            patch(
                "cot_mtkd.cli.stress_stage1_memory.one_pass_expert_gradients",
                wraps=one_pass_expert_gradients,
            ) as one_pass,
            patch(
                "cot_mtkd.cli.stress_stage1_memory.stable_gac_gradients", wraps=stable_gac_gradients
            ) as objective,
        ):
            reports = [
                run_case(
                    "tiny",
                    phase,
                    record,
                    LongCoTCollator(0),
                    model,
                    names,
                    groups,
                    parameters,
                    optimizers,
                    schedulers,
                    BandwidthEMA(decay=0.9, floor=1.0e-12),
                    config,
                    torch.device("cpu"),
                    first_iteration=index * 2,
                )
                for index, phase in enumerate(PHASES)
            ]
        self.assertEqual(PHASES, ("sft_only", "full_interaction"))
        self.assertEqual(resets.call_count, 2)
        self.assertEqual(one_pass.call_count, 2)
        self.assertEqual(objective.call_count, 2)
        for call in one_pass.call_args_list:
            self.assertAlmostEqual(call.kwargs["combined_dpp_scale"], 0.1 * 12)
        for call in objective.call_args_list:
            self.assertEqual(len(call.args[1]), 3)
            self.assertEqual(call.kwargs["rbf_weight"], 0.5)
            self.assertEqual(call.kwargs["beta"], 0.5)
        for report in reports:
            self.assertEqual(report["status"], "ok")
            self.assertEqual(report["completed_warmup_iterations"], 1)
            self.assertEqual(report["completed_measured_iterations"], 1)
            self.assertEqual(len(report["measured_elapsed_seconds"]), 1)
            self.assertEqual(report["max_memory_allocated_bytes"], 42 * GIB)
            self.assertEqual(report["max_memory_reserved_bytes"], 45 * GIB)
            self.assertEqual(report["timing_unit"], "one_microbatch_with_optimizer_step")
            self.assertEqual(report["response_tokens"], 12)
            self.assertTrue(report["memory_includes_warmup"])
        self.assertEqual([report["dpp_samples"] for report in reports], [0, 1])
        for old, current in zip(initial, parameters, strict=True):
            self.assertTrue(any(not torch.equal(a, b) for a, b in zip(old, current, strict=True)))
            self.assertTrue(all(torch.isfinite(value).all() for value in current))

    def test_full_gradients_use_training_normalizers(self) -> None:
        config = stress_config()
        record = stress_record()
        collator = LongCoTCollator(0)
        tokens = shifted_token_views(collator([record]))["response_targets"].numel()
        parameters = [[torch.nn.Parameter(torch.tensor([1.0]))] for _ in range(3)]
        optimizers = [torch.optim.SGD(current, lr=0.0) for current in parameters]
        schedulers = [
            torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0) for optimizer in optimizers
        ]

        def fake_one_pass(*args, combined_dpp_scale=None, **kwargs):
            sft = tokens if combined_dpp_scale is None else tokens + 2 * combined_dpp_scale
            return SimpleNamespace(dpp_sample_count=1), [
                (
                    [torch.tensor([float(sft)])],
                    [torch.tensor([2.0])] if combined_dpp_scale is None else [],
                    0.0,
                    tokens,
                )
                for _ in parameters
            ]

        def inspect_objective(task, repulsion, kernel, *, beta, rbf_weight):
            expected = 1.0 + 2.0 * config["stage1"]["dpp_weight"]
            for values in task:
                self.assertTrue(torch.allclose(values[0], torch.tensor([expected])))
            return stable_gac_gradients(task, repulsion, kernel, beta=beta, rbf_weight=rbf_weight)

        with (
            patch(
                "cot_mtkd.cli.stress_stage1_memory.one_pass_expert_gradients",
                side_effect=fake_one_pass,
            ),
            patch(
                "cot_mtkd.cli.stress_stage1_memory.sft_only_expert_gradients",
                return_value=([torch.tensor([float(tokens)])], [], 0.0, tokens),
            ),
            patch("cot_mtkd.cli.stress_stage1_memory.set_all_adapters_trainable"),
            patch(
                "cot_mtkd.cli.stress_stage1_memory.effective_update_distances",
                return_value=torch.ones(3, 3),
            ),
            patch(
                "cot_mtkd.cli.stress_stage1_memory.repulsion_updates",
                return_value=([[torch.zeros(1)]] * 3, torch.eye(3), torch.ones(3, 3)),
            ),
            patch(
                "cot_mtkd.cli.stress_stage1_memory.stable_gac_gradients", side_effect=inspect_objective
            ) as objective,
        ):
            for index, phase in enumerate(PHASES):
                _run_microbatch(
                    phase,
                    record,
                    collator,
                    None,
                    ["a", "b", "c"],
                    [],
                    parameters,
                    optimizers,
                    schedulers,
                    BandwidthEMA(decay=0.9, floor=1.0e-12),
                    config,
                    torch.device("cpu"),
                    index,
                )
        self.assertEqual(objective.call_count, 1)

    def test_warmup_is_excluded_from_timing_and_included_in_peak(self) -> None:
        with (
            patch("torch.cuda.synchronize"),
            patch("torch.cuda.reset_peak_memory_stats") as resets,
            patch("torch.cuda.max_memory_allocated", return_value=100 * GIB),
            patch("torch.cuda.max_memory_reserved", return_value=115 * GIB),
            patch(
                "cot_mtkd.cli.stress_stage1_memory._run_microbatch", return_value={"dpp_samples": 1}
            ) as iteration,
            patch(
                "cot_mtkd.cli.stress_stage1_memory.time.perf_counter",
                side_effect=[0, 100, 100, 102, 102, 106],
            ),
        ):
            result = run_case(
                "mock",
                "full_interaction",
                stress_record(),
                None,
                None,
                [],
                [],
                [],
                [],
                [],
                BandwidthEMA(0.9, 1.0e-12),
                stress_config(),
                torch.device("cpu"),
                warmup=1,
                repetitions=2,
                first_iteration=7,
            )
        self.assertEqual(resets.call_count, 1)
        self.assertEqual(iteration.call_count, 3)
        self.assertEqual([call.args[-1] for call in iteration.call_args_list], [7, 8, 9])
        self.assertEqual(result["measured_elapsed_seconds"], [2.0, 4.0])
        self.assertEqual(result["elapsed_seconds"], 3.0)
        self.assertEqual(result["max_memory_reserved_bytes"], 115 * GIB)
        self.assertEqual(result["completed_warmup_iterations"], 1)
        self.assertEqual(result["completed_measured_iterations"], 2)

    def test_oom_counters_preserve_partial_measurements(self) -> None:
        for warmup in (0, 1):
            with (
                self.subTest(warmup=warmup),
                patch("torch.cuda.synchronize"),
                patch("torch.cuda.reset_peak_memory_stats"),
                patch("torch.cuda.max_memory_allocated", return_value=100 * GIB),
                patch("torch.cuda.max_memory_reserved", return_value=125 * GIB),
                patch(
                    "cot_mtkd.cli.stress_stage1_memory._run_microbatch",
                    side_effect=[{"dpp_samples": 1}, torch.cuda.OutOfMemoryError("test")],
                ),
            ):
                result = run_case(
                    "mock",
                    "full_interaction",
                    stress_record(),
                    None,
                    None,
                    [],
                    [],
                    [],
                    [],
                    [],
                    BandwidthEMA(0.9, 1.0e-12),
                    stress_config(),
                    torch.device("cpu"),
                    warmup=warmup,
                    repetitions=2,
                )
            self.assertEqual(result["status"], "cuda_oom")
            self.assertEqual(result["completed_warmup_iterations"], warmup)
            self.assertEqual(result["completed_measured_iterations"], 1 - warmup)
            self.assertEqual(result["failed_iteration_kind"], "measured")
            self.assertEqual(memory_recommendation([result])[0], "two_pass_fallback")

    def test_memory_thresholds_and_oom_are_conservative(self) -> None:
        for peak, status, expected in (
            (109, "ok", "one_pass_ready"),
            (110, "ok", "review_headroom"),
            (119, "ok", "review_headroom"),
            (120, "ok", "two_pass_fallback"),
            (90, "cuda_oom", "two_pass_fallback"),
        ):
            with self.subTest(peak=peak, status=status):
                recommendation, worst = memory_recommendation(
                    [{"status": status, "max_memory_reserved_bytes": peak * GIB}]
                )
                self.assertEqual(recommendation, expected)
                self.assertEqual(worst, peak)

    def test_warmup_oom_is_reported_before_any_measurement(self) -> None:
        with (
            patch("torch.cuda.synchronize"),
            patch("torch.cuda.reset_peak_memory_stats"),
            patch("torch.cuda.max_memory_allocated", return_value=90 * GIB),
            patch("torch.cuda.max_memory_reserved", return_value=125 * GIB),
            patch(
                "cot_mtkd.cli.stress_stage1_memory._run_microbatch",
                side_effect=torch.cuda.OutOfMemoryError("warmup test"),
            ) as iteration,
        ):
            result = run_case(
                "mock",
                "full_interaction",
                stress_record(),
                None,
                None,
                [],
                [],
                [],
                [],
                [],
                BandwidthEMA(0.9, 1.0e-12),
                stress_config(),
                torch.device("cpu"),
                warmup=1,
                repetitions=2,
            )
        self.assertEqual(iteration.call_count, 1)
        self.assertEqual(result["status"], "cuda_oom")
        self.assertEqual(result["failed_iteration_kind"], "warmup")
        self.assertEqual(result["completed_warmup_iterations"], 0)
        self.assertEqual(result["completed_measured_iterations"], 0)
        self.assertEqual(result["measured_elapsed_seconds"], [])
        self.assertIsNone(result["elapsed_seconds"])
        self.assertEqual(result["max_memory_reserved_bytes"], 125 * GIB)

    def test_invalid_phase_and_repetition_counts_fail_before_cuda(self) -> None:
        for phase, warmup, repetitions in (("unknown", 1, 1), ("full_interaction", -1, 1), ("full_interaction", 1, 0)):
            with self.subTest(phase=phase, warmup=warmup, repetitions=repetitions):
                with self.assertRaises(ValueError), patch("torch.cuda.synchronize") as synchronize:
                    run_case(
                        "mock",
                        phase,
                        stress_record(),
                        None,
                        None,
                        [],
                        [],
                        [],
                        [],
                        [],
                        BandwidthEMA(0.9, 1.0e-12),
                        stress_config(),
                        torch.device("cpu"),
                        warmup=warmup,
                        repetitions=repetitions,
                    )
                synchronize.assert_not_called()


if __name__ == "__main__":
    unittest.main()
