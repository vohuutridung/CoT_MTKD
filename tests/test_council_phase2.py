from __future__ import annotations

import copy
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from test_stage2_online import LORA, tiny_online_council, two_step_record

from cot_mtkd.data.schema import TokenRegion
from cot_mtkd.models.multi_adapter import extract_adapter_state, set_active_adapter
from cot_mtkd.stage2 import council_cache as council_cache_module
from cot_mtkd.stage2.council import (
    council_signals,
    dynamic_temperature,
    effective_temperature,
    kneedle_support,
    reasoning_token_terms,
    restricted_log_softmax,
    support_mass,
    validate_temperature_config,
)
from cot_mtkd.stage2.council_cache import (
    CACHE_VERSION,
    CouncilCache,
    cache_identity,
    compile_record,
    validate_council_config,
)
from cot_mtkd.stage2.medoid import projection_distance_matrix, select_medoid
from cot_mtkd.stage2.student import compute_record_gradient, plan_record, unpack_cached_support
from cot_mtkd.utils.manifest import file_sha256, fingerprint, write_json


def council_config() -> dict:
    return {
        "method": "council_topk_dynamic_temperature_self_distillation",
        "seed": 42,
        "model": {"name_or_path": "fixture/model", "dtype": "float32"},
        "lora": LORA,
        "medoid": {"epsilon_rel": 1.0e-4, "epsilon_abs": 1.0e-8, "tau_b": 0.0},
        "council": {
            "k_max": None,
            "mask_epsilon": 1.0e-12,
            "tau_min": 0.5,
            "tau_max": 2.0,
            "temperature_schedule": "linear",
            "js_max": None,
            "sigmoid_gamma": 8.0,
            "sigmoid_center": None,
            "step_theta_1": 0.1,
            "step_theta_2": 0.4,
            "alpha": 1.0,
            "beta": 0.25,
            "epsilon_m": 1.0e-6,
        },
        "stage2": {
            "epochs": 1,
            "micro_batch_size": 1,
            "global_batch_size": 1,
            "learning_rate": 2.0e-4,
            "max_length": 128,
            "max_grad_norm": 1.0,
            "checkpoint_every_steps": 1,
            "log_every_steps": 1,
            "resume_from": None,
        },
        "optimizer": {"name": "adamw", "betas": [0.9, 0.999], "eps": 1.0e-8, "weight_decay": 0.0},
        "scheduler": {"name": "cosine", "warmup_ratio": 0.1, "min_lr_ratio": 0.0},
        "runtime": {"lm_head_chunk_tokens": 2, "dataloader_workers": 0},
        "logging": {"reasoning_steps": True, "performance": True},
        "paths": {},
    }


def dense_phase2_reference(model, names, record, config, cached):
    """Full-model implementation of L_SFT + alpha D_KL + beta L_mass for one record."""
    council = config["council"]
    plan = plan_record(record, None, config["stage2"]["max_length"])
    ids = torch.tensor([plan.input_ids])
    reasoning = plan.reasoning_positions
    support_ids, support_mask, variance_mask = unpack_cached_support(cached)
    tau = dynamic_temperature(cached["js"], council, len(names))
    tau_eff = effective_temperature(variance_mask, tau)
    set_active_adapter(model, "student")
    model.train()
    logits = model(input_ids=ids, use_cache=False).logits[0].float()
    units = plan.sft_units
    sft = logits.sum() * 0.0
    for step in plan.step_positions:
        rows = torch.tensor(step) - 1
        targets = torch.tensor([record.input_ids[p] for p in step])
        sft = sft + F.cross_entropy(logits[rows], targets) / units
    for block in (plan.answer_positions, plan.format_positions):
        if block:
            rows = torch.tensor(block) - 1
            targets = torch.tensor([record.input_ids[p] for p in block])
            sft = sft + F.cross_entropy(logits[rows], targets) / units
    kl_terms, mass_terms = [], []
    for index, position in enumerate(reasoning):
        z = logits[position - 1]
        valid = support_mask[index]
        selected = support_ids[index][valid]
        z_support = z[selected]
        log_p = F.log_softmax(z_support, -1)
        q_dist = F.softmax(z_support.detach() / tau_eff[index][valid].float(), -1)
        if int(cached["k"][index]) >= 2:
            kl_terms.append((q_dist * (q_dist.log() - log_p)).sum())
        if bool(cached["gold_in_support"][index]):
            m = torch.logsumexp(z_support, -1) - torch.logsumexp(z, -1)
            m = m.exp().clamp(council["epsilon_m"], 1 - council["epsilon_m"])
            q = float(cached["q"][index])
            mass_terms.append(q * math.log(q / m) + (1 - q) * math.log((1 - q) / (1 - m)))
    kl = torch.stack(kl_terms).mean() if kl_terms else sft * 0.0
    mass = torch.stack(mass_terms).mean() if mass_terms else sft * 0.0
    total = sft + council["alpha"] * kl + council["beta"] * mass
    return total, {"sft": float(sft), "kl": float(kl), "mass": float(mass)}


class CouncilToolTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(5)
        self.experts, self.tokens, self.vocabulary = 3, 6, 40
        self.logits = torch.randn(self.experts, self.tokens, self.vocabulary) * 3
        self.probabilities = F.softmax(self.logits, -1)
        self.mean = self.probabilities.mean(0)

    def test_kneedle_matches_sorted_reference_and_flat_window_keeps_all(self):
        ids, mask, k = kneedle_support(self.mean)
        for t in range(self.tokens):
            sorted_p, order = torch.sort(self.mean[t], descending=True)
            x = torch.arange(1, self.vocabulary + 1) / self.vocabulary
            y = (sorted_p - sorted_p[-1]) / (sorted_p[0] - sorted_p[-1])
            expected = int(torch.argmax((1 - x) - y)) + 1
            self.assertEqual(int(k[t]), expected)
            self.assertEqual(ids[t, :expected].tolist(), order[:expected].tolist())
            self.assertEqual(int(mask[t].sum()), expected)
        _, _, flat = kneedle_support(torch.full((1, 9), 1 / 9))
        self.assertEqual(int(flat[0]), 9)
        _, mask_max, k_max = kneedle_support(self.mean, k_max=5)
        self.assertLessEqual(int(k_max.max()), 5)
        self.assertEqual(mask_max.sum(-1).tolist(), k_max.tolist())
        self.assertFalse(ids.requires_grad)
        with self.assertRaises(ValueError):
            kneedle_support(self.mean, k_max=0)

    def test_restricted_distribution_disagreement_mask_and_mass_match_dense(self):
        ids, mask, _ = kneedle_support(self.mean)
        log_pi = restricted_log_softmax(self.logits, ids, mask)
        signals = council_signals(log_pi, mask)
        q = support_mass(self.mean, ids, mask)
        for t in range(self.tokens):
            selected = ids[t][mask[t]]
            pis = torch.stack(
                [
                    self.probabilities[m, t, selected] / self.probabilities[m, t, selected].sum()
                    for m in range(self.experts)
                ]
            ).double()
            torch.testing.assert_close(
                log_pi[:, t][:, mask[t]].double(), pis.log(), atol=1e-5, rtol=1e-5
            )
            bar = pis.mean(0)
            entropy = lambda d: -(d * d.log()).sum()
            js = entropy(bar) - sum(entropy(d) for d in pis) / self.experts
            self.assertAlmostEqual(float(signals.js[t]), float(js), places=6)
            variance = ((pis - bar) ** 2).mean(0)
            torch.testing.assert_close(
                signals.variance_mask[t][mask[t]],
                variance / (variance.max() + 1e-12),
                atol=1e-5,
                rtol=1e-5,
            )
            self.assertAlmostEqual(float(q[t]), float(self.mean[t, selected].sum()), places=6)
        self.assertLessEqual(float(signals.js.max()), math.log(self.experts))
        self.assertTrue(bool((signals.variance_mask[~mask] == 0).all()))
        self.assertTrue(bool(((signals.variance_mask.max(-1).values - 1).abs() < 1e-6).all()))

    def test_temperature_schedules_and_effective_temperature_branches(self):
        js = torch.tensor([0.0, 0.2, math.log(3), 5.0])
        linear = dynamic_temperature(js, {"tau_min": 0.5, "tau_max": 2.0}, 3)
        torch.testing.assert_close(
            linear, torch.tensor([0.5, 0.5 + 1.5 * 0.2 / math.log(3), 2.0, 2.0]).double()
        )
        percentile = dynamic_temperature(
            torch.tensor([0.0, 0.04, 0.08, 0.2]),
            {"tau_min": 0.8, "tau_max": 1.5, "js_max": "p95"},
            3,
            js_p95=0.08,
        )
        torch.testing.assert_close(percentile, torch.tensor([0.8, 1.15, 1.5, 1.5]).double())
        with self.assertRaises(ValueError):
            dynamic_temperature(
                js, {"tau_min": 0.8, "tau_max": 1.5, "js_max": "p95"}, 3
            )
        step = dynamic_temperature(
            torch.tensor([0.05, 0.1, 0.2, 0.4, 0.9], dtype=torch.float64),
            {
                "tau_min": 0.5,
                "tau_max": 2.0,
                "temperature_schedule": "step",
                "step_theta_1": 0.1,
                "step_theta_2": 0.4,
            },
            3,
        )
        self.assertEqual(step.tolist(), [0.5, 0.5, 1.0, 2.0, 2.0])
        sigmoid = dynamic_temperature(
            torch.tensor([0.3]),
            {"tau_min": 0.5, "tau_max": 2.0, "temperature_schedule": "sigmoid", "sigmoid_gamma": 2},
            3,
            js_median=0.3,
        )
        self.assertAlmostEqual(float(sigmoid[0]), 1.25, places=6)
        with self.assertRaises(ValueError):
            dynamic_temperature(
                js,
                {"tau_min": 0.5, "tau_max": 2.0, "temperature_schedule": "sigmoid"},
                3,
            )
        for bad in ({"tau_min": 1.0, "tau_max": 2.0}, {"tau_min": 0.5, "tau_max": 1.0}):
            with self.assertRaises(ValueError):
                validate_temperature_config(bad)
        mask = torch.tensor([[1.0, 0.5, 0.0], [1.0, 0.5, 0.0]])
        tau = torch.tensor([2.0, 0.5], dtype=torch.float64)
        tau_eff = effective_temperature(mask, tau)
        self.assertEqual(tau_eff[0].tolist(), [2.0, 1.5, 1.0])
        self.assertEqual(tau_eff[1].tolist(), [1.0, 0.75, 0.5])

    def test_student_terms_gradient_is_P_minus_Q_on_support_and_mass_is_binary_kl(self):
        ids, mask, _ = kneedle_support(self.mean)
        log_pi = restricted_log_softmax(self.logits, ids, mask)
        signals = council_signals(log_pi, mask)
        q = support_mass(self.mean, ids, mask)
        tau_eff = effective_temperature(
            signals.variance_mask,
            dynamic_temperature(signals.js, {"tau_min": 0.5, "tau_max": 2}, 3),
        )
        z = torch.randn(self.tokens, self.vocabulary, requires_grad=True)
        targets = torch.randint(0, self.vocabulary, (self.tokens,))
        terms = reasoning_token_terms(z, targets, ids, mask, tau_eff, q, 1e-6)
        gradient = torch.autograd.grad(terms["kl"].sum(), z)[0]
        for t in range(self.tokens):
            selected = ids[t][mask[t]]
            p = F.softmax(z[t, selected], -1)
            q_dist = F.softmax(z[t, selected] / tau_eff[t][mask[t]].float(), -1)
            torch.testing.assert_close(gradient[t, selected], p - q_dist, atol=1e-5, rtol=1e-5)
            outside = torch.ones(self.vocabulary, dtype=torch.bool)
            outside[selected] = False
            self.assertEqual(float(gradient[t, outside].abs().max()), 0.0)
            m = F.softmax(z[t], -1)[selected].sum()
            qq = q[t].float()
            expected = qq * torch.log(qq / m) + (1 - qq) * torch.log((1 - qq) / (1 - m))
            self.assertAlmostEqual(float(terms["mass"][t]), float(expected), places=4)
            self.assertAlmostEqual(
                float(terms["sft"][t]), float(F.cross_entropy(z[t], targets[t])), places=5
            )
            self.assertAlmostEqual(float(terms["student_mass"][t]), float(m), places=5)
        single = reasoning_token_terms(
            z[:1],
            targets[:1],
            torch.tensor([[2]]),
            torch.tensor([[True]]),
            torch.ones(1, 1, dtype=torch.float64),
            torch.tensor([0.9]),
            1e-6,
        )
        self.assertEqual(float(single["kl"][0]), 0.0)
        self.assertEqual(float(single["entropy"][0]), 0.0)

    def test_distribution_summary_quantiles_match_when_torch_quantile_refuses(self):
        values = torch.linspace(-1, 3, 41, dtype=torch.float64)
        direct = council_cache_module.distribution_summary(values)
        self.assertAlmostEqual(direct["p95"], float(torch.quantile(values, 0.95)))
        with patch.object(council_cache_module, "_TORCH_QUANTILE_LIMIT", 4):
            forced = council_cache_module.distribution_summary(values)
        for key in ("median", "p90", "p95", "mean", "min", "max"):
            self.assertAlmostEqual(forced[key], direct[key])


class MedoidTest(unittest.TestCase):
    def test_projection_distance_bounds_invariance_and_medoid_choice(self):
        torch.manual_seed(3)
        rank, width = 4, 32
        x = torch.randn(rank, width)
        transform = torch.randn(rank, rank) + 3 * torch.eye(rank)
        y = torch.randn(rank, width)
        basis, _ = torch.linalg.qr(torch.randn(width, width))
        a, b = basis[:, :rank].T, basis[:, rank : 2 * rank].T
        distances = projection_distance_matrix([x, transform @ x, y, a, b], 1e-4, 1e-8)
        self.assertLess(float(distances[0, 1]), 1e-2)
        self.assertAlmostEqual(float(distances[3, 4]), rank, places=2)
        self.assertTrue(bool((distances >= -1e-6).all()))
        self.assertTrue(bool((distances <= rank + 1e-6).all()))
        torch.testing.assert_close(distances, distances.T)
        zeros = torch.zeros(width, rank)
        states = {
            "e0": {"m.lora_A.e0.weight": x, "m.lora_B.e0.weight": zeros},
            "e1": {"m.lora_A.e1.weight": transform @ x, "m.lora_B.e1.weight": zeros},
            "e2": {"m.lora_A.e2.weight": a, "m.lora_B.e2.weight": zeros},
        }
        result = select_medoid(states, {"epsilon_rel": 1e-4, "epsilon_abs": 1e-8, "tau_b": 0.0})
        self.assertFalse(result.use_b_side)
        self.assertIn(result.medoid_index, (0, 1))
        self.assertEqual(result.module_count, 1)
        self.assertEqual(result.sums.tolist(), result.squared_distances.sum(1).tolist())
        with_b = {
            name: {**state, f"m.lora_B.{name}.weight": torch.randn(width, rank)}
            for name, state in states.items()
        }
        result_b = select_medoid(with_b, {"epsilon_rel": 1e-4, "epsilon_abs": 1e-8, "tau_b": 0.0})
        self.assertTrue(result_b.use_b_side)
        self.assertIsInstance(result_b.to_json()["squared_distances"], list)
        with self.assertRaises(ValueError):
            select_medoid({"e0": states["e0"]}, {"epsilon_rel": 1e-4, "epsilon_abs": 1e-8})

    def test_tiny_council_medoid_minimizes_total_distance(self):
        model, names, _ = tiny_online_council(False)
        states = {name: extract_adapter_state(model, name) for name in names}
        result = select_medoid(states, {"epsilon_rel": 1e-4, "epsilon_abs": 1e-8, "tau_b": 0.0})
        self.assertEqual(result.medoid_index, int(torch.argmin(result.sums)))
        self.assertTrue(result.use_b_side)
        self.assertEqual(result.module_count, 7)


class StudentStepTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_plan_covers_steps_answer_and_format_blocks(self):
        record = two_step_record()
        plan = plan_record(record, None, 128)
        self.assertEqual(plan.step_positions, [[3, 4], [6, 7, 8]])
        self.assertEqual(plan.answer_positions, [11])
        self.assertEqual(plan.format_positions, [2, 5, 9, 10, 12])
        self.assertEqual(plan.sft_units, 4)
        self.assertEqual(plan.reasoning_positions, [3, 4, 6, 7, 8])
        with self.assertRaisesRegex(ValueError, "exceeds stage2.max_length"):
            plan_record(record, None, 12)
        no_answer = copy.deepcopy(record)
        no_answer.region_ids[11] = int(TokenRegion.EOS)
        self.assertEqual(plan_record(no_answer, None, 128).sft_units, 3)

    def test_record_gradient_matches_dense_reference_and_only_moves_student(self):
        for checkpointing in (False, True):
            with self.subTest(checkpointing=checkpointing):
                model, names, parameters = tiny_online_council(checkpointing)
                config = council_config()
                record = two_step_record()
                cached, scores, _ = compile_record(
                    model, names, record, config, torch.device("cpu")
                )
                self.assertEqual(cached["token_positions"].tolist(), [3, 4, 6, 7, 8])
                self.assertEqual(cached["step_offsets"].tolist(), [0, 2, 5])
                self.assertEqual(cached["support_offsets"][-1].item(), int(cached["k"].sum()))
                self.assertEqual(scores.shape, (3,))
                teachers_before = {name: extract_adapter_state(model, name) for name in names}
                result = compute_record_gradient(
                    model,
                    names,
                    parameters,
                    record,
                    None,
                    config,
                    torch.device("cpu"),
                    cached_target=cached,
                )
                reference, parts = dense_phase2_reference(model, names, record, config, cached)
                expected = torch.autograd.grad(reference, parameters, allow_unused=True)
                for actual, want, parameter in zip(result.gradients, expected, parameters):
                    want = torch.zeros_like(parameter) if want is None else want
                    torch.testing.assert_close(actual, want, atol=2e-5, rtol=2e-4)
                self.assertAlmostEqual(result.loss, float(reference), places=5)
                self.assertAlmostEqual(result.metrics["sft_loss"], parts["sft"], places=5)
                self.assertAlmostEqual(result.metrics["kl_loss"], parts["kl"], places=5)
                self.assertAlmostEqual(result.metrics["mass_loss"], parts["mass"], places=4)
                self.assertEqual(result.steps, 2)
                self.assertEqual(result.active_steps, 2)
                self.assertEqual(result.metrics["reasoning_tokens"], 5)
                self.assertEqual(result.metrics["supervised_tokens"], 11)
                self.assertEqual(result.metrics["sequence_tokens"], 13)
                self.assertEqual(len(result.step_metrics), 2)
                self.assertEqual(result.metrics["kl_tokens"], int((cached["k"] >= 2).sum()))
                self.assertEqual(
                    result.metrics["mass_tokens"], int(cached["gold_in_support"].sum())
                )
                for name in names:
                    for key, value in extract_adapter_state(model, name).items():
                        torch.testing.assert_close(
                            value, teachers_before[name][key], atol=0, rtol=0
                        )
                with self.assertRaisesRegex(RuntimeError, "requires a council cache"):
                    compute_record_gradient(
                        model, names, parameters, record, None, config, torch.device("cpu")
                    )
                broken = dict(cached)
                broken["token_positions"] = cached["token_positions"][:-1]
                with self.assertRaisesRegex(RuntimeError, "mapping mismatch"):
                    compute_record_gradient(
                        model,
                        names,
                        parameters,
                        record,
                        None,
                        config,
                        torch.device("cpu"),
                        cached_target=broken,
                    )

    def test_k_max_bounds_support_and_changes_cache_identity(self):
        model, names, _ = tiny_online_council(False)
        config = council_config()
        config["council"]["k_max"] = 3
        cached, _, _ = compile_record(model, names, two_step_record(), config, torch.device("cpu"))
        self.assertLessEqual(int(cached["k"].max()), 3)
        prepared = {"data_file_sha256": "data", "tokenizer_fingerprint": "tok"}
        teachers = {"adapter_bundle_sha256": "weights"}
        base = fingerprint(cache_identity(council_config(), prepared, teachers))
        self.assertNotEqual(base, fingerprint(cache_identity(config, prepared, teachers)))
        for section, key, value in (
            ("council", "tau_min", 0.7),
            ("council", "tau_max", 3.0),
            ("council", "alpha", 0.5),
            ("council", "beta", 0.1),
            ("council", "temperature_schedule", "step"),
            ("council", "js_max", "p95"),
            ("stage2", "epochs", 4),
            ("stage2", "learning_rate", 1e-3),
        ):
            changed = council_config()
            changed[section][key] = value
            self.assertEqual(base, fingerprint(cache_identity(changed, prepared, teachers)))
        for section, key, value in (
            ("stage2", "max_length", 64),
            ("medoid", "epsilon_rel", 1e-3),
            ("model", "name_or_path", "other"),
        ):
            changed = council_config()
            changed[section][key] = value
            self.assertNotEqual(base, fingerprint(cache_identity(changed, prepared, teachers)))
        with patch("cot_mtkd.stage2.council_cache.file_sha256", return_value="changed"):
            self.assertNotEqual(
                base, fingerprint(cache_identity(council_config(), prepared, teachers))
            )

    def test_config_validation_rejects_obsolete_and_invalid_settings(self):
        validate_council_config(council_config())
        for mutate in (
            lambda c: c["council"].update(k_max=1),
            lambda c: c["council"].update(tau_min=1.5),
            lambda c: c["council"].update(alpha=-1.0),
            lambda c: c["council"].update(epsilon_m=0.5),
            lambda c: c["council"].update(temperature_schedule="cubic"),
            lambda c: c["medoid"].update(epsilon_abs=0.0),
            lambda c: c.update(aggregation={}),
        ):
            changed = council_config()
            mutate(changed)
            with self.assertRaises(ValueError):
                validate_council_config(changed)

    def test_cache_roundtrip_checksums_and_medoid_consistency(self):
        model, names, _ = tiny_online_council(False)
        config = council_config()
        value, _, _ = compile_record(model, names, two_step_record(), config, torch.device("cpu"))
        states = {name: extract_adapter_state(model, name) for name in names}
        medoid = select_medoid(states, config["medoid"])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_file(value, str(root / "sample.safetensors"))
            torch.save(states[medoid.medoid_name], root / "student_init.pt")
            write_json(
                root / "index.json",
                {
                    "sample": {
                        "file": "sample.safetensors",
                        "sha256": file_sha256(root / "sample.safetensors"),
                    }
                },
            )
            identity = {"test": 1}
            key = fingerprint(identity)
            manifest = {
                "artifact": "stage2_council_cache",
                "cache_version": CACHE_VERSION,
                "fingerprint": key,
                "identity": identity,
                "records": 1,
                "adapter_names": names,
                "index_file": "index.json",
                "index_file_sha256": file_sha256(root / "index.json"),
                "student_init_file": "student_init.pt",
                "student_init_file_sha256": file_sha256(root / "student_init.pt"),
                "medoid": medoid.to_json(),
                "selected_expert_index": medoid.medoid_index,
                "selected_expert": medoid.medoid_name,
                "diagnostics": {"js": {"median": 0.1}},
            }
            write_json(root / "manifest.json", manifest)
            cache = CouncilCache(root, key)
            loaded = cache.get("sample")
            for name in value:
                torch.testing.assert_close(loaded[name], value[name], atol=0, rtol=0)
            _, mask, variance = unpack_cached_support(loaded)
            self.assertEqual(mask.sum(-1).tolist(), loaded["k"].tolist())
            self.assertTrue(bool((variance[~mask] == 0).all()))
            self.assertEqual(cache.js_median, 0.1)
            from cot_mtkd.stage2.initialization import create_cached_student

            config["model"], config["lora"] = {}, {}
            with patch("cot_mtkd.stage2.initialization.create_student_model", return_value=model):
                student, _, _ = create_cached_student(
                    config, SimpleNamespace(device=torch.device("cpu")), cache
                )
            for key_name, tensor in extract_adapter_state(student, "student").items():
                torch.testing.assert_close(
                    tensor, states[medoid.medoid_name][key_name], atol=0, rtol=0
                )
            wrong = dict(manifest, selected_expert_index=(medoid.medoid_index + 1) % 3)
            write_json(root / "manifest.json", wrong)
            with self.assertRaisesRegex(RuntimeError, "medoid selection mismatch"):
                CouncilCache(root, key)
            write_json(root / "manifest.json", manifest)
            with self.assertRaisesRegex(RuntimeError, "fingerprint mismatch"):
                CouncilCache(root, "wrong")
            (root / "sample.safetensors").write_bytes(b"corrupt")
            with self.assertRaisesRegex(RuntimeError, "checksum mismatch"):
                CouncilCache(root, key).get("sample")


if __name__ == "__main__":
    unittest.main()
