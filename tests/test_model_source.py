from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import yaml

from cot_mtkd.config import load_config
from cot_mtkd.models.multi_adapter import (
    load_base_causal_lm,
    load_tokenizer,
    require_same_model_source,
)
from cot_mtkd.utils.manifest import fingerprint


class ModelSourceTest(unittest.TestCase):
    def test_tokenizer_uses_default_hub_revision_even_with_a_legacy_pin(self):
        for config in (
            {"name_or_path": "fixture/model"},
            {"name_or_path": "fixture/model", "revision": "legacy-pin"},
        ):
            with self.subTest(config=config):
                tokenizer = SimpleNamespace(is_fast=True, pad_token_id=None, eos_token="eos")
                with patch(
                    "transformers.AutoTokenizer.from_pretrained", return_value=tokenizer
                ) as download:
                    self.assertIs(load_tokenizer(config), tokenizer)
                download.assert_called_once_with(
                    "fixture/model", use_fast=True, trust_remote_code=False
                )
                self.assertEqual(tokenizer.pad_token, "eos")
                self.assertEqual(tokenizer.padding_side, "right")

    def test_backbone_and_attention_fallback_never_pass_a_revision(self):
        for legacy in (False, True):
            with self.subTest(legacy=legacy):
                config = {
                    "name_or_path": "fixture/model",
                    "dtype": "float32",
                    "attn_implementation": "flash_attention_2",
                }
                if legacy:
                    config["revision"] = "legacy-pin"
                model = Mock()
                with (
                    patch(
                        "transformers.AutoModelForCausalLM.from_pretrained",
                        side_effect=[ImportError("flash unavailable"), model],
                    ) as download,
                    self.assertLogs("cot_mtkd.models.multi_adapter", level="WARNING"),
                ):
                    self.assertIs(load_base_causal_lm(config, torch.device("cpu")), model)
                self.assertEqual(download.call_count, 2)
                for call in download.call_args_list:
                    self.assertEqual(call.args, ("fixture/model",))
                    self.assertNotIn("revision", call.kwargs)
                self.assertEqual(
                    [call.kwargs["attn_implementation"] for call in download.call_args_list],
                    ["flash_attention_2", "sdpa"],
                )
                model.to.assert_called_once_with(torch.device("cpu"))
                self.assertFalse(model.config.use_cache)

    def test_model_compatibility_checks_name_but_ignores_legacy_revisions(self):
        actual = {"name_or_path": "fixture/model"}
        for expected in (
            {"name_or_path": "fixture/model"},
            {"name_or_path": "fixture/model", "revision": "old-pin"},
        ):
            require_same_model_source(actual, expected, "fixture")
        actual["revision"] = "another-old-pin"
        require_same_model_source(actual, expected, "fixture")
        with self.assertRaisesRegex(RuntimeError, "model source mismatch"):
            require_same_model_source(actual, {"name_or_path": "different/model"}, "fixture")

    def test_config_discards_only_backbone_revision_before_fingerprinting(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(
                "model:\n  name_or_path: fixture/model\n  revision: old-pin\n"
                "dataset:\n  revision: dataset-pin\n"
                "teacher_source:\n  revision: teacher-pin\n"
                "benchmarks:\n  - revision: benchmark-pin\n"
            )
            config = load_config(path)
            changed = load_config(path, ["model.revision=another-old-pin"])
            self.assertNotIn("revision", config["model"])
            self.assertEqual(fingerprint(config), fingerprint(changed))
            self.assertEqual(config["dataset"]["revision"], "dataset-pin")
            self.assertEqual(config["teacher_source"]["revision"], "teacher-pin")
            self.assertEqual(config["benchmarks"][0]["revision"], "benchmark-pin")

    def test_repository_configs_have_no_backbone_revision(self):
        root = Path(__file__).resolve().parents[1]
        for path in (root / "configs").rglob("*.yaml"):
            with self.subTest(config=path.name):
                config = yaml.safe_load(path.read_text())
                if "model" in config:
                    self.assertNotIn("revision", config["model"])


if __name__ == "__main__":
    unittest.main()
