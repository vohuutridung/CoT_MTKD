from __future__ import annotations

import unittest
from dataclasses import replace
from unittest.mock import patch

import torch
import torch.nn.functional as F
from test_stage2_online import tiny_online_council, two_step_record

from cot_mtkd.cli.stress_stage2_memory import (
    _optimizer_and_scheduler,
    _run_microbatch,
    make_synthetic_record,
)
from cot_mtkd.models.multi_adapter import set_active_adapter
from cot_mtkd.stage1.kneedle import build_union_support, local_k_from_probe
from cot_mtkd.stage2.council_cache import compile_record, finalize_record
from cot_mtkd.stage2.disagreement import fit_disagreement_calibration
from cot_mtkd.stage2.output_space import compute_record_gradient, plan_record
from cot_mtkd.stage2.trainer import OUTPUT_SPACE_METHOD


def output_config():
    return {
        "method": OUTPUT_SPACE_METHOD,
        "aggregation": {
            "temperature": 1.0,
            "disagreement_pooling_power": 4.0,
            "tau_quantile": 0.75,
            "sft_weight": 0.01,
            "search_k": 512,
            "k_min": 8,
            "teacher_execution": "precomputed_support_tail",
        },
        "stage2": {
            "max_length": 128,
            "learning_rate": 2e-4,
            "max_grad_norm": 1.0,
            "global_batch_size": 16,
        },
        "runtime": {"lm_head_chunk_tokens": 2, "preprocessing_hidden_storage": "cpu"},
        "optimizer": {"betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0},
        "scheduler": {"warmup_ratio": 0.1, "min_lr_ratio": 0},
    }


def compile_fixture_target(model, names, record, config, device):
    """Tiny-test corpus calibration; production fits all records jointly."""
    pending, scores, stats = compile_record(model, names, record, config, device)
    values = pending["step_disagreement"]
    # An empty record receives the calibration of another valid fixture step.
    calibration = fit_disagreement_calibration(
        values if len(values) else torch.tensor([0.1]),
        config["aggregation"]["disagreement_pooling_power"],
        config["aggregation"]["tau_quantile"],
    )
    return finalize_record(pending, calibration, config), scores, stats


def dense_reference(model, names, parameters, record, config):
    """Independent dense JSD, corpus quantile, power target, loss and LoRA VJP."""
    plan = plan_record(record, None, config["stage2"]["max_length"])
    positions = [p for step in plan.step_positions for p in step]
    indices = torch.tensor(positions) - 1
    gold = torch.tensor([record.input_ids[p] for p in positions])
    ids = torch.tensor([plan.input_ids])
    teachers = []
    for name in names:
        model.set_adapter(name)
        model.requires_grad_(False)
        model.eval()
        with torch.no_grad():
            teachers.append(model(input_ids=ids, use_cache=False).logits[0, indices].double())
    z = torch.stack(teachers)
    non_gold = z.clone()
    non_gold.scatter_(2, gold[None, :, None].expand(3, -1, -1), -torch.inf)
    values, top = non_gold.topk(min(512, z.shape[-1] - 1), -1)
    k = torch.stack([local_k_from_probe(v, 8)[1] for v in values])
    union, mask = build_union_support(top, k)
    support = [sorted(set(union[t, mask[t]].tolist()) | {int(gold[t])}) for t in range(len(gold))]
    a = config["aggregation"]
    temperature, power = a["temperature"], a["disagreement_pooling_power"]
    cursor = 0
    blocks, means, deltas = [], [], []
    for step in plan.step_positions:
        js_tokens, reduced = [], []
        for t in range(cursor, cursor + len(step)):
            v = support[t]
            pi = F.softmax(z[:, t, v] / temperature, -1)
            mixture = pi.mean(0)
            js_tokens.append((pi * (pi.log() - mixture.log())).sum(-1).mean())
            p = F.softmax(z[:, t] / temperature, -1)
            outside = p.clone()
            outside[:, v] = 0
            reduced.append(torch.cat([p[:, v], outside.sum(-1, keepdim=True)], -1))
        js = torch.stack(js_tokens)
        means.append(float(js.mean()))
        deltas.append(float(js.pow(power).mean().pow(1 / power)))
        blocks.append((cursor, len(step), reduced))
        cursor += len(step)
    tau = float(torch.quantile(torch.tensor(deltas, dtype=torch.float64), a["tau_quantile"]))
    rhos = [delta / (delta + tau) for delta in deltas]
    set_active_adapter(model, "student")
    model.train()
    student = model(input_ids=ids, use_cache=False).logits[0, indices].float()
    kd_steps, sft_steps, init_steps = [], [], []
    for (cursor, length, reduced), rho in zip(blocks, rhos, strict=True):
        kd_tokens = []
        for t, r in zip(range(cursor, cursor + length), reduced, strict=True):
            q = r.log().mean(0).exp() if rho == 0 else r.pow(rho).mean(0).pow(1 / rho)
            q = q / q.sum()
            p = F.softmax(student[t] / temperature, -1)
            outside = torch.ones_like(p, dtype=torch.bool)
            outside[support[t]] = False
            rs = torch.cat([p[support[t]], p[outside].sum()[None]])
            kd_tokens.append(temperature**2 * (q * (q.log() - rs.log())).sum())
        kd_steps.append(torch.stack(kd_tokens).mean())
        sft_steps.append(
            F.cross_entropy(student[cursor : cursor + length], gold[cursor : cursor + length])
        )
        init_steps.append(
            -F.log_softmax(z[:, cursor : cursor + length], -1)
            .gather(-1, gold[cursor : cursor + length][None, :, None].expand(3, -1, -1))
            .squeeze(-1)
            .mean(-1)
        )
    kd, sft = torch.stack(kd_steps).mean(), torch.stack(sft_steps).mean()
    loss = kd + a["sft_weight"] * sft
    return (
        float(loss.detach()),
        [g.detach() for g in torch.autograd.grad(loss, parameters)],
        means,
        torch.stack(init_steps).mean(0),
        float(kd.detach()),
        float(sft.detach()),
        deltas,
        tau,
        rhos,
    )


class CachedOutputSpaceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_cache_training_matches_dense_loss_gradient_js_sft_and_init_scores(self):
        record, config = two_step_record(), output_config()
        model, names, parameters = tiny_online_council(checkpointing=True)
        expected = dense_reference(model, names, parameters, record, config)
        target, init_scores, _ = compile_fixture_target(
            model, names, record, config, torch.device("cpu")
        )
        # Any teacher selection during training fails this test.
        original = model.set_adapter

        def student_only(name):
            self.assertEqual(name, "student")
            return original(name)

        with (
            patch.object(model, "set_adapter", side_effect=student_only),
            patch("torch.autograd.grad", wraps=torch.autograd.grad) as grad,
        ):
            result = compute_record_gradient(
                model,
                names,
                parameters,
                record,
                None,
                config,
                torch.device("cpu"),
                cached_target=target,
            )
        self.assertEqual(len([c for c in grad.call_args_list if isinstance(c.args[1], list)]), 1)
        self.assertAlmostEqual(result.loss, expected[0], delta=1e-6)
        torch.testing.assert_close(
            target["token_js_mean"],
            torch.tensor(expected[2], dtype=torch.float64),
            atol=5e-8,
            rtol=1e-5,
        )
        torch.testing.assert_close(
            target["step_disagreement"],
            torch.tensor(expected[6], dtype=torch.float64),
            atol=5e-8,
            rtol=1e-5,
        )
        self.assertAlmostEqual(float(target["tau"]), expected[7], delta=5e-8)
        torch.testing.assert_close(
            target["step_rho"], torch.tensor(expected[8], dtype=torch.float64), atol=1e-5, rtol=1e-5
        )
        torch.testing.assert_close(init_scores, expected[3], atol=5e-7, rtol=1e-6)
        for g, e in zip(result.gradients, expected[1], strict=True):
            torch.testing.assert_close(g, e, atol=5e-7, rtol=3e-4)
        self.assertAlmostEqual(result.metrics["kd_loss"], expected[4], delta=1e-6)
        self.assertAlmostEqual(result.metrics["sft_loss"], expected[5], delta=1e-6)
        self.assertEqual(result.metrics["teacher_head_chunk_sweeps"], 0)
        self.assertAlmostEqual(
            result.loss, result.metrics["kd_loss"] + 0.01 * result.metrics["sft_loss"], delta=1e-12
        )
        for row in result.step_metrics:
            self.assertEqual((row["js_temperature"], row["kd_temperature"]), (1.0, 1.0))
        self.assertEqual(result.metrics["recomputed_steps"], 0)

    def test_zero_sft_weight_is_new_kd_only_and_cache_targets_are_static(self):
        model, names, params = tiny_online_council(False)
        record, config = two_step_record(), output_config()
        target, _, _ = compile_fixture_target(model, names, record, config, torch.device("cpu"))
        config["aggregation"]["sft_weight"] = 0.0
        config["stage2"]["epochs"] = 99
        result = compute_record_gradient(
            model, names, params, record, None, config, torch.device("cpu"), cached_target=target
        )
        self.assertEqual(result.loss, result.metrics["kd_loss"])
        expected = dense_reference(model, names, params, record, config)
        for g, e in zip(result.gradients, expected[1], strict=True):
            torch.testing.assert_close(g, e, atol=5e-7, rtol=3e-4)

    def test_planner_uses_only_content_and_complete_steps(self):
        record = two_step_record()
        plan = plan_record(replace(record, solution="gold" * 10000), None, 10)
        self.assertEqual(plan.step_positions, [[3, 4], [6, 7, 8]])
        self.assertEqual(plan.solution, [])
        self.assertEqual(plan_record(record, None, 9).step_positions, [[3, 4]])
        self.assertEqual(plan_record(record, None, 5).num_steps, 0)
        with self.assertRaisesRegex(ValueError, "prompt/control"):
            plan_record(record, None, 2)
        synthetic, p = make_synthetic_record(record, None, 32, OUTPUT_SPACE_METHOD)
        self.assertEqual(len(p.input_ids), 32)
        self.assertEqual(synthetic.solution, record.solution)

    def test_cache_required_and_mapping_mismatch_rejected(self):
        record, config = two_step_record(), output_config()
        with self.assertRaisesRegex(RuntimeError, "requires a council cache"):
            compute_record_gradient(None, [], [], record, None, config, torch.device("cpu"))
        model, names, params = tiny_online_council(False)
        target, _, _ = compile_fixture_target(model, names, record, config, torch.device("cpu"))
        target["token_positions"][0] += 1
        with self.assertRaisesRegex(RuntimeError, "mapping mismatch"):
            compute_record_gradient(
                model,
                names,
                params,
                record,
                None,
                config,
                torch.device("cpu"),
                cached_target=target,
            )

    def test_stress_measures_only_cached_student_path(self):
        record, config = two_step_record(), output_config()
        model, names, parameters = tiny_online_council(True)
        target, _, _ = compile_fixture_target(model, names, record, config, torch.device("cpu"))
        config["_stress_targets"] = {record.sample_id: target}
        optimizer, scheduler = _optimizer_and_scheduler(parameters, config, 10)
        with patch("torch.cuda.synchronize"):
            result = _run_microbatch(
                record,
                None,
                model,
                names,
                parameters,
                optimizer,
                scheduler,
                config,
                torch.device("cpu"),
            )
        self.assertTrue(result["optimizer_update_applied"])
        self.assertIn("final_kd_sft_gradient", result["component_seconds"])
        self.assertNotIn("teacher_forward", result["component_seconds"])

    def test_empty_plan_has_zero_objective_and_never_forwards_any_model(self):
        record, config = two_step_record(), output_config()
        config["stage2"]["max_length"] = 5
        target, scores, stats = compile_fixture_target(
            None, ["a", "b", "c"], record, config, torch.device("cpu")
        )
        parameter = torch.nn.Parameter(torch.ones(2))
        result = compute_record_gradient(
            None,
            ["a", "b", "c"],
            [parameter],
            record,
            None,
            config,
            torch.device("cpu"),
            cached_target=target,
        )
        self.assertEqual((result.steps, result.active_steps, result.loss), (0, 0, 0.0))
        self.assertEqual(stats["teacher_forward_count"], 0)
        torch.testing.assert_close(scores, torch.zeros(3, dtype=torch.float64))
        torch.testing.assert_close(result.gradients[0], torch.zeros(2))


if __name__ == "__main__":
    unittest.main()
