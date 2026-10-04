from __future__ import annotations

import math
import unittest
from dataclasses import replace
from unittest.mock import patch

import torch
import torch.nn.functional as F

from test_stage2_online import tiny_online_council, two_step_record
from cot_mtkd.cli.stress_stage2_memory import (
    _run_microbatch,
    _optimizer_and_scheduler,
    make_synthetic_record,
)
from cot_mtkd.models.multi_adapter import set_active_adapter
from cot_mtkd.stage1.kneedle import local_k_from_probe, build_union_support
from cot_mtkd.stage2.council_cache import compile_record
from cot_mtkd.stage2.output_space import compute_record_gradient, plan_record
from cot_mtkd.stage2.trainer import OUTPUT_SPACE_METHOD


# Corpus step-JS reference (nats) for the ECDF rho mapping in unit tests.
JS_REFERENCE = torch.tensor([0.0, 1.0e-4, 1.0e-3, 1.0e-2, 5.0e-2, 0.1, 0.3], dtype=torch.float64)


def output_config():
    return {
        "method": OUTPUT_SPACE_METHOD,
        "aggregation": {
            "js_temperature": 1.0,
            "kd_temperature": 2.0,
            "search_k": 512,
            "k_min": 8,
            "teacher_execution": "precomputed_support_tail",
            "kd_weight": 1.0,
            "sft_weight": 0.25,
            "rho_mapping": "ecdf",
            "rho_constant": None,
            "teachers": "all",
            "loss_normalization": "token",
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
        "_council": {"js_reference": JS_REFERENCE},
    }


def reference_rho(ds, aggregation, experts, js_reference):
    mapping = aggregation["rho_mapping"]
    if mapping == "constant":
        return float(aggregation["rho_constant"])
    if mapping == "linear":
        return min(1.0, float(ds / math.log(experts)))
    return float((js_reference <= ds).sum()) / len(js_reference)


def dense_reference(model, names, parameters, record, config):
    """Independent full-model and dense softmax reference for every aggregation mode."""
    aggregation = config["aggregation"]
    council = config.get("_council", {})
    chosen = council.get("teacher_indices", list(range(len(names))))
    temperature = aggregation["kd_temperature"]
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
    non_gold.scatter_(2, gold[None, :, None].expand(len(names), -1, -1), -torch.inf)
    values, top = non_gold.topk(min(512, z.shape[-1] - 1), -1)
    k = torch.stack([local_k_from_probe(v, 8)[1] for v in values])
    union, mask = build_union_support(top, k)
    support = [sorted(set(union[t, mask[t]].tolist()) | {int(gold[t])}) for t in range(len(gold))]
    set_active_adapter(model, "student")
    model.train()
    student = model(input_ids=ids, use_cache=False).logits[0, indices].float()
    cursor = 0
    kd_steps, sft_steps, js_steps, init_steps, rho_steps = [], [], [], [], []
    for step in plan.step_positions:
        js_tokens, reduced = [], []
        for t in range(cursor, cursor + len(step)):
            v = support[t]
            pi = F.softmax(z[:, t, v], -1)
            mixture = pi.mean(0)
            js_tokens.append((pi * (pi.log() - mixture.log())).sum(-1).mean())
            p = F.softmax(z[:, t] / temperature, -1)
            tail = p.clone()
            tail[:, v] = 0
            reduced.append(torch.cat([p[:, v], tail.sum(-1, keepdim=True)], -1))
        ds = torch.stack(js_tokens).mean()
        rho = reference_rho(ds, aggregation, len(names), council.get("js_reference"))
        js_steps.append(float(ds))
        rho_steps.append(rho)
        kd_tokens = []
        for t, r in zip(range(cursor, cursor + len(step)), reduced, strict=True):
            r = r[chosen]
            if len(chosen) == 1:
                q = r[0]
            else:
                q = r.log().mean(0).exp() if rho == 0 else r.pow(rho).mean(0).pow(1 / rho)
            q = q / q.sum()
            p = F.softmax(student[t] / temperature, -1)
            outside = torch.ones_like(p, dtype=torch.bool)
            outside[support[t]] = False
            rs = torch.cat([p[support[t]], p[outside].sum()[None]])
            kd_tokens.append(temperature**2 * (q * (q.log() - rs.log())).sum())
        kd_steps.append(torch.stack(kd_tokens))
        sft_steps.append(
            F.cross_entropy(
                student[cursor : cursor + len(step)],
                gold[cursor : cursor + len(step)],
                reduction="none",
            )
        )
        init_steps.append(
            -F.log_softmax(z[:, cursor : cursor + len(step)], -1)
            .gather(-1, gold[cursor : cursor + len(step)][None, :, None].expand(len(names), -1, -1))
            .squeeze(-1)
            .mean(-1)
        )
        cursor += len(step)
    if aggregation["loss_normalization"] == "token":
        kd, sft = torch.cat(kd_steps).mean(), torch.cat(sft_steps).mean()
    else:
        kd = torch.stack([v.mean() for v in kd_steps]).mean()
        sft = torch.stack([v.mean() for v in sft_steps]).mean()
    loss = aggregation["kd_weight"] * kd + aggregation["sft_weight"] * sft
    return (
        float(loss.detach()),
        [g.detach() for g in torch.autograd.grad(loss, parameters)],
        js_steps,
        torch.stack(init_steps).mean(0),
        float(kd.detach()),
        float(sft.detach()),
        rho_steps,
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
        target, init_scores, _ = compile_record(model, names, record, config, torch.device("cpu"))
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
            target["step_js"], torch.tensor(expected[2], dtype=torch.float64), atol=5e-8, rtol=1e-5
        )
        torch.testing.assert_close(init_scores, expected[3], atol=5e-7, rtol=1e-6)
        for g, e in zip(result.gradients, expected[1], strict=True):
            torch.testing.assert_close(g, e, atol=5e-7, rtol=3e-4)
        self.assertAlmostEqual(result.metrics["kd_loss"], expected[4], delta=1e-6)
        self.assertAlmostEqual(result.metrics["sft_loss"], expected[5], delta=1e-6)
        self.assertEqual(result.metrics["teacher_head_chunk_sweeps"], 0)
        self.assertAlmostEqual(
            result.loss, result.metrics["kd_loss"] + 0.25 * result.metrics["sft_loss"], delta=1e-12
        )
        for row in result.step_metrics:
            self.assertEqual((row["js_temperature"], row["kd_temperature"]), (1.0, 2.0))
        self.assertEqual(result.metrics["recomputed_steps"], 0)

    def test_zero_sft_weight_is_new_kd_only_and_cache_targets_are_static(self):
        model, names, params = tiny_online_council(False)
        record, config = two_step_record(), output_config()
        target, _, _ = compile_record(model, names, record, config, torch.device("cpu"))
        config["aggregation"]["sft_weight"] = 0.0
        config["stage2"]["epochs"] = 99
        result = compute_record_gradient(
            model, names, params, record, None, config, torch.device("cpu"), cached_target=target
        )
        self.assertEqual(result.loss, result.metrics["kd_loss"])
        expected = dense_reference(model, names, params, record, config)
        for g, e in zip(result.gradients, expected[1], strict=True):
            torch.testing.assert_close(g, e, atol=5e-7, rtol=3e-4)

    def test_every_aggregation_mode_matches_dense_reference(self):
        record = two_step_record()
        model, names, params = tiny_online_council(False)
        base = output_config()
        target, _, _ = compile_record(model, names, record, base, torch.device("cpu"))
        variants = {
            "old_linear_step": {"rho_mapping": "linear", "loss_normalization": "step"},
            "geometric": {"rho_mapping": "constant", "rho_constant": 0.0},
            "arithmetic": {"rho_mapping": "constant", "rho_constant": 1.0},
            "single_expert": {"teachers": "expert_1"},
            "sft_only": {"kd_weight": 0.0, "sft_weight": 1.0},
            "kd_temperature_1": {"kd_temperature": 1.0},
        }
        for label, changes in variants.items():
            with self.subTest(label):
                config = output_config()
                config["aggregation"].update(changes)
                if label == "single_expert":
                    config["_council"]["teacher_indices"] = [1]
                cached = target
                if label == "kd_temperature_1":
                    cached, _, _ = compile_record(model, names, record, config, torch.device("cpu"))
                result = compute_record_gradient(
                    model,
                    names,
                    params,
                    record,
                    None,
                    config,
                    torch.device("cpu"),
                    cached_target=cached,
                )
                expected = dense_reference(model, names, params, record, config)
                self.assertAlmostEqual(result.loss, expected[0], delta=1e-6)
                rows = result.step_metrics
                for actual, wanted in zip([row["rho"] for row in rows], expected[6], strict=True):
                    self.assertAlmostEqual(actual, wanted, delta=1e-9)
                for g, e in zip(result.gradients, expected[1], strict=True):
                    torch.testing.assert_close(g, e, atol=5e-7, rtol=3e-4)
                if label == "single_expert":
                    self.assertEqual({row["teacher_count"] for row in result.step_metrics}, {1})

    def test_council_size_is_not_fixed_to_three(self):
        record, config = two_step_record(), output_config()
        model, names, params = tiny_online_council(False)
        pair = names[:2]
        target, scores, _ = compile_record(model, pair, record, config, torch.device("cpu"))
        self.assertEqual(target["expert_support_log_probs"].shape[0], 2)
        self.assertEqual(target["raw_k"].shape[0], 2)
        self.assertEqual(len(scores), 2)
        result = compute_record_gradient(
            model, pair, params, record, None, config, torch.device("cpu"), cached_target=target
        )
        expected = dense_reference(model, pair, params, record, config)
        self.assertAlmostEqual(result.loss, expected[0], delta=1e-6)
        for g, e in zip(result.gradients, expected[1], strict=True):
            torch.testing.assert_close(g, e, atol=5e-7, rtol=3e-4)
        with self.assertRaisesRegex(ValueError, "at least two"):
            compile_record(model, names[:1], record, config, torch.device("cpu"))

    def test_ecdf_rho_is_rank_uniform_whatever_the_js_scale(self):
        from cot_mtkd.stage2.output_space import map_step_rho

        aggregation = {"rho_mapping": "ecdf"}
        reference = torch.tensor([1e-4, 2e-4, 3e-4, 4e-4], dtype=torch.float64)
        queries = torch.tensor([1e-4, 2.5e-4, 4e-4, 1.0], dtype=torch.float64)
        rho = map_step_rho(queries, aggregation, 3, reference)
        torch.testing.assert_close(rho, torch.tensor([0.25, 0.5, 1.0, 1.0], dtype=torch.float64))
        # The same ranks at a 1000x larger scale give the same rho.
        torch.testing.assert_close(
            map_step_rho(queries * 1000, aggregation, 3, reference * 1000), rho
        )
        with self.assertRaisesRegex(RuntimeError, "JS reference"):
            map_step_rho(torch.tensor([0.1]), aggregation, 3, None)

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
        target, _, _ = compile_record(model, names, record, config, torch.device("cpu"))
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
        target, _, _ = compile_record(model, names, record, config, torch.device("cpu"))
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
        target, scores, stats = compile_record(
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
