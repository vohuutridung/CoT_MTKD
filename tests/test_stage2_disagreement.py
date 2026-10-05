from __future__ import annotations

import copy
import math
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F
from test_output_space_online import output_config
from test_stage2_online import tiny_online_council, two_step_record

from cot_mtkd.stage2.council_cache import (
    CACHE_VERSION,
    CouncilCache,
    compile_record,
    finalize_record,
)
from cot_mtkd.stage2.disagreement import (
    fit_disagreement_calibration,
    pool_token_disagreement,
    saturation_rho,
)
from cot_mtkd.stage2.output_space import compute_record_gradient
from cot_mtkd.stage2.output_space_losses import normalized_js_disagreement
from cot_mtkd.stage2.trainer import OUTPUT_SPACE_METHOD, _load_checkpoint
from cot_mtkd.utils.manifest import fingerprint, write_json


class Phase2DisagreementTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_token_js_matches_independent_kl_definition(self):
        torch.manual_seed(14)
        logs = F.log_softmax(torch.randn(3, 17, 23, dtype=torch.float64), -1)
        p = logs.exp()
        independent = (p * (logs - p.mean(0).log())).sum(-1).mean(0)
        actual = normalized_js_disagreement(logs) * math.log(3)
        torch.testing.assert_close(actual, independent, atol=1e-14, rtol=1e-13)

    def test_power_four_preserves_pivot_and_power_one_is_arithmetic(self):
        js = torch.tensor([0.001] * 99 + [0.5], dtype=torch.float64)
        expected = js.pow(4).mean().pow(0.25)
        torch.testing.assert_close(pool_token_disagreement(js), expected, atol=1e-15, rtol=1e-14)
        torch.testing.assert_close(pool_token_disagreement(js, 1), js.mean(), atol=0, rtol=0)
        self.assertGreater(float(expected), 20 * float(js.mean()))
        self.assertEqual(float(pool_token_disagreement(torch.zeros(100))), 0)
        for scale in (1e-200, 1e200):
            values = torch.tensor([scale, scale / 2], dtype=torch.float64)
            actual = pool_token_disagreement(values)
            self.assertTrue(bool(torch.isfinite(actual)))
            self.assertGreater(float(actual), 0)
        for power in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                pool_token_disagreement(js, power)

    def test_corpus_quantile_and_saturation_are_in_nats(self):
        steps = torch.tensor([0.0, 0.01, 0.02, 0.03, 0.04], dtype=torch.float64)
        calibration = fit_disagreement_calibration(steps)
        self.assertEqual(calibration["tau"], 0.03)
        self.assertEqual(calibration["tau_quantile"], 0.75)
        self.assertEqual(calibration["num_reasoning_steps"], 5)
        self.assertEqual(calibration["source"], "stage2_training_corpus")
        rho = saturation_rho(steps, calibration["tau"])
        torch.testing.assert_close(rho, steps / (steps + 0.03), atol=0, rtol=0)
        self.assertEqual(float(rho[0]), 0)
        self.assertEqual(float(rho[3]), 0.5)
        self.assertTrue(bool(((rho >= 0) & (rho < 1)).all()))
        extreme = saturation_rho(torch.tensor([1e200], dtype=torch.float64), 1e-200)
        self.assertLess(float(extreme[0]), 1)
        self.assertEqual(fit_disagreement_calibration(steps, quantile=0.5)["tau"], 0.02)
        for values in (torch.zeros(5), torch.tensor([]), torch.tensor([float("nan")])):
            with self.assertRaises(ValueError):
                fit_disagreement_calibration(values)
        for tau in (0, -1, float("nan")):
            with self.assertRaises(ValueError):
                saturation_rho(steps, tau)

    def test_one_global_tau_is_shared_and_finalization_never_forwards_teachers(self):
        config = output_config()
        model, names, parameters = tiny_online_council(False)
        first = two_step_record()
        second = copy.deepcopy(first)
        second.sample_id = "different-training-record"
        second.input_ids[3] = second.labels[3] = 12
        pending = [
            compile_record(model, names, rec, config, torch.device("cpu"))[0]
            for rec in (first, second)
        ]
        steps = torch.cat([v["step_disagreement"] for v in pending])
        calibration = fit_disagreement_calibration(steps)
        targets = [finalize_record(v, calibration, config) for v in pending]
        for target in targets:
            self.assertEqual(float(target["tau"]), calibration["tau"])
            self.assertFalse(any(k.startswith("teacher_") for k in target))
            torch.testing.assert_close(
                target["step_rho"],
                target["step_disagreement"] / (target["step_disagreement"] + calibration["tau"]),
                atol=0,
                rtol=0,
            )
        with self.assertRaisesRegex(RuntimeError, "Missing/invalid"):
            finalize_record(pending[0], {}, config)
        broken = dict(targets[0])
        broken.pop("tau")
        with self.assertRaisesRegex(RuntimeError, "Missing fitted tau"):
            compute_record_gradient(
                model,
                names,
                parameters,
                first,
                None,
                config,
                torch.device("cpu"),
                cached_target=broken,
            )
        # Changing a channel coefficient never changes the already fitted target.
        adjusted = copy.deepcopy(config)
        adjusted["aggregation"]["sft_weight"] = 0
        result = compute_record_gradient(
            model,
            names,
            parameters,
            first,
            None,
            adjusted,
            torch.device("cpu"),
            cached_target=targets[0],
        )
        self.assertEqual(result.loss, result.metrics["kd_loss"])

    def test_old_cache_and_missing_fitted_tau_fail_before_optimization(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            write_json(
                path / "manifest.json", {"artifact": "stage2_council_cache", "cache_version": 1}
            )
            with self.assertRaisesRegex(RuntimeError, "rebuild stage2-cache"):
                CouncilCache(path, "old-cache")
            write_json(
                path / "manifest.json",
                {
                    "artifact": "stage2_council_cache",
                    "cache_version": CACHE_VERSION,
                    "identity": {},
                    "fingerprint": "incorrect",
                },
            )
            with self.assertRaisesRegex(RuntimeError, "fingerprint mismatch"):
                CouncilCache(path, "incorrect")
            key = fingerprint({})
            write_json(
                path / "manifest.json",
                {
                    "artifact": "stage2_council_cache",
                    "cache_version": CACHE_VERSION,
                    "identity": {},
                    "fingerprint": key,
                },
            )
            with self.assertRaisesRegex(RuntimeError, "Missing/invalid fitted"):
                CouncilCache(path, key)

    def test_checkpoint_with_missing_or_different_tau_rejected_before_loading_state(self):
        calibration = fit_disagreement_calibration(torch.tensor([0.01, 0.02, 0.03]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            for saved in (None, {**calibration, "tau": 0.1}):
                torch.save(
                    {
                        "method": OUTPUT_SPACE_METHOD,
                        "run_fingerprint": "test-run",
                        "disagreement_calibration": saved,
                    },
                    path,
                )
                with self.assertRaisesRegex(RuntimeError, "missing/different fitted tau"):
                    _load_checkpoint(
                        path,
                        None,
                        None,
                        None,
                        "test-run",
                        method=OUTPUT_SPACE_METHOD,
                        expected_calibration=calibration,
                    )


if __name__ == "__main__":
    unittest.main()
