from __future__ import annotations

import copy
import math
import unittest
from dataclasses import replace
from unittest.mock import patch

import torch
import torch.nn.functional as F
from test_stage2_online import CharacterTokenizer, tiny_online_council, two_step_record

from cot_mtkd.cli.stress_stage2_memory import (
    _optimizer_and_scheduler,
    _run_microbatch,
    longest_eligible_record,
    make_synthetic_record,
)
from cot_mtkd.models.multi_adapter import extract_adapter_state, set_active_adapter
from cot_mtkd.stage2.output_space import compute_record_gradient, plan_record, runtime_options
from cot_mtkd.stage2.trainer import OUTPUT_SPACE_METHOD


def dense_reference(model, names, parameters, record, temperature):
    """Direct full-model PDF reference on the two complete fixture steps."""
    ids = torch.tensor([record.input_ids[:10]])
    steps = [[3, 4], [6, 7, 8]]
    indices = torch.tensor([p for step in steps for p in step]) - 1
    logps = []
    for name in names:
        model.set_adapter(name)
        model.requires_grad_(False)
        model.eval()
        with torch.no_grad():
            logits = model(input_ids=ids, use_cache=False).logits[0, indices]
            logps.append(F.log_softmax(logits.double() / temperature, -1))
    logps = torch.stack(logps)
    set_active_adapter(model, "student")
    model.train()
    logits = model(input_ids=ids, use_cache=False).logits[0, indices]
    student_logp = F.log_softmax(logits.float() / temperature, -1)
    cursor, losses, disagreements = 0, [], []
    for step in steps:
        end = cursor + len(step)
        values = logps[:, cursor:end]
        probs = values.exp()
        mixture = probs.mean(0)
        ds = float((probs * (values - mixture.log())).sum(-1).mean() / math.log(len(names)))
        raw = probs.pow(ds).mean(0).pow(1 / ds)
        target = raw / raw.sum(-1, keepdim=True)
        losses.append(
            temperature**2 * (target * (target.log() - student_logp[cursor:end])).sum(-1).mean()
        )
        disagreements.append(ds)
        cursor = end
    loss = torch.stack(losses).mean()
    gradients = torch.autograd.grad(loss, parameters)
    return (
        float(loss.detach()),
        [g.detach() for g in gradients],
        sum(disagreements),
        disagreements,
        [float(value.detach()) for value in losses],
    )


class OutputSpaceOnlineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.thread_count = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.thread_count)

    def setUp(self):
        self.record, self.tokenizer = two_step_record(), CharacterTokenizer()
        self.config = {
            "method": OUTPUT_SPACE_METHOD,
            "aggregation": {"temperature": 2.0, "teacher_execution": "online_full_vocab"},
            "stage2": {
                "max_length": 128,
                "learning_rate": 2.0e-4,
                "max_grad_norm": 1.0,
                "global_batch_size": 32,
            },
            "runtime": {"lm_head_chunk_tokens": 2},
            "optimizer": {"betas": [0.9, 0.999], "eps": 1.0e-8, "weight_decay": 0.0},
            "scheduler": {"warmup_ratio": 0.1, "min_lr_ratio": 0.0},
        }

    def test_parameter_loss_gradient_and_frozen_council_match_dense_reference(self):
        model, names, parameters = tiny_online_council(checkpointing=True)
        before = {name: p.detach().clone() for name, p in model.named_parameters()}
        expected_loss, expected_gradients, disagreement, per_step_js, per_step_loss = (
            dense_reference(model, names, parameters, self.record, 2.0)
        )
        with patch("torch.autograd.grad", wraps=torch.autograd.grad) as grad:
            actual = compute_record_gradient(
                model,
                names,
                parameters,
                self.record,
                self.tokenizer,
                self.config,
                torch.device("cpu"),
            )
        parameter_vjps = [call for call in grad.call_args_list if isinstance(call.args[1], list)]
        self.assertEqual(len(parameter_vjps), 1)
        self.assertEqual((actual.steps, actual.active_steps, actual.discarded_steps), (2, 2, 0))
        self.assertAlmostEqual(actual.loss, expected_loss, delta=3.0e-7)
        self.assertAlmostEqual(actual.metrics["disagreement_sum"], disagreement, delta=1.0e-8)
        self.assertEqual(len(actual.step_metrics), 2)
        for index, row in enumerate(actual.step_metrics):
            self.assertEqual(row["step_id"], index)
            self.assertEqual(row["n_tokens"], [2, 3][index])
            self.assertAlmostEqual(row["js_normalized"], per_step_js[index], delta=1.0e-8)
            self.assertAlmostEqual(row["js_mean"], per_step_js[index] * math.log(3), delta=1.0e-8)
            self.assertEqual(row["rho"], row["js_normalized"])
            self.assertAlmostEqual(row["step_kd_loss"], per_step_loss[index], delta=3.0e-7)
        for value, expected in zip(actual.gradients, expected_gradients, strict=True):
            torch.testing.assert_close(value, expected, atol=3.0e-7, rtol=3.0e-4)
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(parameter, before[name], atol=0, rtol=0)
            self.assertIsNone(parameter.grad)
            self.assertEqual(parameter.requires_grad, "lora_" in name and ".student." in name)

    def test_planner_retains_complete_steps_without_gold_and_masks_structure(self):
        long_gold = replace(self.record, solution="gold" * 10000)
        plan = plan_record(long_gold, None, 10)
        self.assertEqual(plan.input_ids, self.record.input_ids[:10])
        self.assertEqual(plan.step_positions, [[3, 4], [6, 7, 8]])
        self.assertEqual(plan.solution, [])
        first = plan_record(self.record, None, 9)
        self.assertEqual(first.step_positions, [[3, 4]])
        self.assertEqual((first.num_steps, first.discarded_steps), (1, 1))
        self.assertEqual(plan_record(self.record, None, 5).num_steps, 0)
        with self.assertRaisesRegex(ValueError, "prompt/control"):
            plan_record(self.record, None, 2)

    def test_device_hidden_storage_and_cached_probabilities_match_memory_path(self):
        model, names, parameters = tiny_online_council(checkpointing=True)
        reference = compute_record_gradient(
            model, names, parameters, self.record, None, self.config, torch.device("cpu")
        )
        config = copy.deepcopy(self.config)
        config["runtime"].update(
            teacher_hidden_storage="device", teacher_probability_cache_gib=1.0,
            lm_head_chunk_tokens=1024,
        )
        actual = compute_record_gradient(
            model, names, parameters, self.record, None, config, torch.device("cpu")
        )
        self.assertAlmostEqual(actual.loss, reference.loss, delta=3.0e-7)
        self.assertAlmostEqual(
            actual.metrics["disagreement_sum"], reference.metrics["disagreement_sum"],
            delta=1.0e-8,
        )
        for a, b in zip(actual.gradients, reference.gradients, strict=True):
            torch.testing.assert_close(a, b, atol=3.0e-7, rtol=3.0e-4)
        self.assertEqual(len(actual.step_metrics), len(reference.step_metrics))

    def test_runtime_options_reject_invalid_storage_and_cache_limits(self):
        self.assertEqual(runtime_options({}), ("cpu", 0))
        self.assertEqual(
            runtime_options({"teacher_hidden_storage": "device", "teacher_probability_cache_gib": 8}),
            ("device", 8 * 2**30),
        )
        for value in (-1, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "finite and nonnegative"):
                runtime_options({"teacher_probability_cache_gib": value})
        with self.assertRaisesRegex(ValueError, "cpu or device"):
            runtime_options({"teacher_hidden_storage": "disk"})

    def test_gold_metadata_and_legacy_answer_suffix_do_not_change_target_or_gradient(self):
        model, names, parameters = tiny_online_council(checkpointing=False)
        original = compute_record_gradient(
            model, names, parameters, self.record, None, self.config, torch.device("cpu")
        )
        altered = copy.deepcopy(self.record)
        altered.solution = "unrelated gold"
        altered.input_ids[10:] = [22, 23, 24]
        actual = compute_record_gradient(
            model, names, parameters, altered, None, self.config, torch.device("cpu")
        )
        self.assertEqual(actual.loss, original.loss)
        self.assertEqual(actual.metrics, original.metrics)
        self.assertEqual(actual.step_metrics, original.step_metrics)
        for a, b in zip(actual.gradients, original.gradients, strict=True):
            torch.testing.assert_close(a, b, atol=0, rtol=0)

    def test_empty_plan_is_zero_and_synthetic_stress_reaches_reasoning_limit(self):
        config = copy.deepcopy(self.config)
        config["stage2"]["max_length"] = 5
        result = compute_record_gradient(
            None,
            ["a", "b"],
            [torch.nn.Parameter(torch.ones(2))],
            self.record,
            None,
            config,
            torch.device("cpu"),
        )
        self.assertEqual((result.steps, result.active_steps, result.loss), (0, 0, 0.0))
        self.assertEqual(result.step_metrics, [])
        torch.testing.assert_close(result.gradients[0], torch.zeros(2))
        synthetic, plan = make_synthetic_record(self.record, None, 32, OUTPUT_SPACE_METHOD)
        self.assertEqual((plan.num_steps, plan.discarded_steps), (2, 0))
        self.assertEqual(len(plan.input_ids), 32)
        self.assertEqual(synthetic.solution, self.record.solution)
        selected, selected_plan, counts = longest_eligible_record(
            [self.record, replace(self.record, solution="gold" * 1000)],
            None,
            32,
            OUTPUT_SPACE_METHOD,
        )
        self.assertIs(selected, self.record)
        self.assertEqual(len(selected_plan.input_ids), 10)
        self.assertEqual(counts["eligible_examples"], 2)

    def test_stress_uses_output_space_path_and_updates_only_student(self):
        model, names, parameters = tiny_online_council(checkpointing=True)
        before_student = [p.detach().clone() for p in parameters]
        before_teachers = {name: extract_adapter_state(model, name) for name in names}
        optimizer, scheduler = _optimizer_and_scheduler(parameters, self.config, 10)
        with patch("torch.cuda.synchronize"):
            result = _run_microbatch(
                self.record,
                None,
                model,
                names,
                parameters,
                optimizer,
                scheduler,
                self.config,
                torch.device("cpu"),
            )
        self.assertTrue(result["optimizer_update_applied"])
        self.assertEqual(result["active_steps"], 2)
        self.assertIn("final_kd_gradient", result["component_seconds"])
        self.assertNotIn("teacher_kd_gradients", result["component_seconds"])
        self.assertNotIn("answer_anchor_gradients", result["component_seconds"])
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before_student, parameters)))
        for name in names:
            for key, value in extract_adapter_state(model, name).items():
                torch.testing.assert_close(value, before_teachers[name][key], atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
