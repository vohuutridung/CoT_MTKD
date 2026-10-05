from __future__ import annotations

import copy
import unittest
from pathlib import Path

import yaml

from cot_mtkd.stage2.trainer import METHOD, _validate_stage2_config, gradient_accumulation_steps


class CouncilConfigTest(unittest.TestCase):
    def test_default_configs_use_one_epoch_and_valid_global_batch_boundaries(self):
        root = Path(__file__).resolve().parents[1]
        for filename in ("qwen25_7b_council.yaml", "qwen25_7b_council_local.yaml"):
            config = yaml.safe_load((root / "configs" / "stage2" / filename).read_text())
            _validate_stage2_config(config)
            self.assertEqual(config["method"], METHOD)
            self.assertEqual(config["stage2"]["epochs"], 1)
            self.assertEqual(config["stage2"]["global_batch_size"], 8)
            self.assertEqual(config["stage2"]["micro_batch_size"], 1)
            self.assertEqual(config["council"]["tau_min"], 0.8)
            self.assertEqual(config["council"]["tau_max"], 1.5)
            self.assertEqual(config["council"]["js_max"], "p95")
            self.assertEqual(config["council"]["alpha"], 1.0)
            self.assertGreaterEqual(config["council"]["beta"], 0.1 * config["council"]["alpha"])
            self.assertLessEqual(config["council"]["beta"], 0.5 * config["council"]["alpha"])
            self.assertEqual(config["council"]["temperature_schedule"], "linear")
            self.assertNotIn("aggregation", config)
            self.assertNotIn("geometry", config)
            self.assertNotIn("medoid", config["paths"])
            for world, expected in ((1, 8), (2, 4), (4, 2), (8, 1)):
                self.assertEqual(gradient_accumulation_steps(config["stage2"], world), expected)
            for world in (3, 16, 32, 0):
                with self.assertRaises(ValueError):
                    gradient_accumulation_steps(config["stage2"], world)
            changed = copy.deepcopy(config["stage2"])
            changed["gradient_accumulation_steps"] = 4
            with self.assertRaises(ValueError):
                gradient_accumulation_steps(changed, 1)
            changed["gradient_accumulation_steps"] = None
            changed["micro_batch_size"] = 2
            self.assertEqual(gradient_accumulation_steps(changed, 4), 1)
            changed["micro_batch_size"] = 1.5
            with self.assertRaises(ValueError):
                gradient_accumulation_steps(changed, 4)

    def test_legacy_methods_and_sections_are_rejected(self):
        root = Path(__file__).resolve().parents[1]
        config = yaml.safe_load(
            (root / "configs" / "stage2" / "qwen25_7b_council.yaml").read_text()
        )
        for mutate in (
            lambda c: c.update(method="disagreement_adaptive_distribution_aggregation_mtkd"),
            lambda c: c.update(method="task_anchored_gradient_geometry_mtkd"),
            lambda c: c.update(aggregation={"js_temperature": 1.0}),
            lambda c: c["stage2"].update(epochs=0),
            lambda c: c["stage2"].update(kd_loss_weight=0.5),
            lambda c: c["lora"].update(dropout=0.1),
        ):
            changed = copy.deepcopy(config)
            mutate(changed)
            with self.assertRaises(ValueError):
                _validate_stage2_config(changed)


if __name__ == "__main__":
    unittest.main()
