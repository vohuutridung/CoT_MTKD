from __future__ import annotations

import copy
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors.torch import save_file
from test_stage2_online import LORA, CharacterTokenizer, tiny_online_council, two_step_record
from transformers import Qwen2ForCausalLM

from cot_mtkd.data.prepare import tokenizer_fingerprint
from cot_mtkd.models.multi_adapter import (
    extract_adapter_state,
    load_adapter_bundle,
    load_adapter_state,
)
from cot_mtkd.stage2.council_cache import build_council_cache
from cot_mtkd.stage2.teachers import (
    canonical_adapter_state,
    ensure_stage2_teachers,
    require_teacher_dataset,
)
from cot_mtkd.stage2.trainer import METHOD, train_stage2
from cot_mtkd.utils.distributed import DistributedContext
from cot_mtkd.utils.manifest import (
    file_sha256,
    fingerprint,
    read_json,
    write_config_snapshot,
    write_json,
)


class HubTeachersTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.model, cls.names, _ = tiny_online_council(checkpointing=False)
        for name in cls.names:
            cls.model.peft_config[name].base_model_name_or_path = "fixture/model"

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def fixture(self, root, *, legacy_backbone_revision=None):
        snapshot = root / "hub"
        self.model.save_pretrained(
            snapshot,
            selected_adapters=self.names,
            safe_serialization=True,
            save_embedding_layers=False,
        )
        source_config = {
            "model": {"name_or_path": "fixture/model"},
            "lora": LORA,
            "seed": 42,
        }
        if legacy_backbone_revision is not None:
            source_config["model"]["revision"] = legacy_backbone_revision
        write_config_snapshot(snapshot / "config.yaml", source_config)
        original = {
            "artifact": "stage1_checkpoint",
            "adapter_names": self.names,
            "config": source_config,
            "config_file_sha256": "0" * 64,
            "prepared_manifest_fingerprint": "historical-training-data",
            "tokenizer_fingerprint": "fixture-tokenizer",
            "global_step": 94,
            "adapter_bundle": "final/adapter_states.pt",
            "adapter_bundle_sha256": "not-uploaded",
        }
        write_json(snapshot / "manifest.json", original)
        config = copy.deepcopy(source_config)
        config["paths"] = {"stage1": str(root / "imported")}
        config["teacher_source"] = {
            "type": "huggingface",
            "repo_id": "fixture/council",
            "revision": "a" * 40,
            "adapter_names": self.names,
            "weight_sha256": {
                name: file_sha256(snapshot / name / "adapter_model.safetensors")
                for name in self.names
            },
        }
        return snapshot, config, original

    def test_real_peft_roundtrip_and_offline_reuse_preserve_every_teacher_tensor(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot, config, original = self.fixture(Path(directory))
            with patch(
                "cot_mtkd.stage2.teachers.snapshot_download", return_value=str(snapshot)
            ) as download:
                manifest = ensure_stage2_teachers(config)
                self.assertEqual(download.call_count, 1)
                call = download.call_args.kwargs
                self.assertEqual(call["revision"], "a" * 40)
                self.assertEqual(len(call["allow_patterns"]), 8)
                self.assertNotIn("checkpoint.pt", call["allow_patterns"])
            imported = Path(config["paths"]["stage1"])
            self.assertFalse(manifest["original_config_checksum_matches_upload"])
            self.assertEqual(read_json(imported / "source/manifest.json"), original)
            self.assertEqual(manifest["prepared_manifest_fingerprint"], "historical-training-data")
            self.assertEqual(manifest["original_manifest_fingerprint"], fingerprint(original))
            bundle = load_adapter_bundle(imported / manifest["adapter_bundle"])
            for name in self.names:
                reference = extract_adapter_state(self.model, name)
                self.assertEqual(bundle[name].keys(), reference.keys())
                for key in reference:
                    torch.testing.assert_close(bundle[name][key], reference[key], atol=0, rtol=0)
            # Load the imported state into the actual PEFT student and compare logits.
            self.model.eval()
            self.model.set_adapter(self.names[1])
            with torch.no_grad():
                expected = self.model(input_ids=torch.tensor([[1, 2, 3, 4]])).logits
                load_adapter_state(self.model, "student", bundle[self.names[1]])
                self.model.set_adapter("student")
                actual = self.model(input_ids=torch.tensor([[1, 2, 3, 4]])).logits
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            with patch(
                "cot_mtkd.stage2.teachers.snapshot_download", side_effect=AssertionError("network")
            ):
                self.assertEqual(ensure_stage2_teachers(config), manifest)

    def test_legacy_backbone_and_adapter_pins_do_not_block_hub_import(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot, config, _ = self.fixture(
                Path(directory), legacy_backbone_revision="historical-backbone-pin"
            )
            config["model"].pop("revision")
            for name in self.names:
                path = snapshot / name / "adapter_config.json"
                adapter = read_json(path)
                adapter["revision"] = "another-historical-pin"
                write_json(path, adapter)
            with patch(
                "cot_mtkd.stage2.teachers.snapshot_download", return_value=str(snapshot)
            ) as download:
                manifest = ensure_stage2_teachers(config)
                config["model"]["revision"] = "ignored-legacy-setting"
                self.assertEqual(ensure_stage2_teachers(config), manifest)
                download.assert_called_once()
            self.assertEqual(manifest["adapter_names"], self.names)

    def test_corrupted_or_changed_cached_sources_are_rejected(self):
        for change in ("weight", "bundle", "source", "revision", "configured_checksum"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                snapshot, config, _ = self.fixture(Path(directory))
                with patch(
                    "cot_mtkd.stage2.teachers.snapshot_download", return_value=str(snapshot)
                ):
                    ensure_stage2_teachers(config)
                imported = Path(config["paths"]["stage1"])
                paths = {
                    "weight": imported / "source/expert_0/adapter_model.safetensors",
                    "bundle": imported / "adapter_states.pt",
                    "source": imported / "source/manifest.json",
                }
                if change in paths:
                    paths[change].write_bytes(b"corrupted")
                elif change == "revision":
                    config["teacher_source"]["revision"] = "b" * 40
                else:
                    config["teacher_source"]["weight_sha256"]["expert_0"] = "0" * 64
                with patch("cot_mtkd.stage2.teachers.snapshot_download") as download:
                    with self.assertRaises(RuntimeError):
                        ensure_stage2_teachers(config)
                    download.assert_not_called()

    def test_hub_import_through_council_and_actual_phase2_update(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot, config, original = self.fixture(root)
            tokenizer = CharacterTokenizer()
            original["tokenizer_fingerprint"] = tokenizer_fingerprint(tokenizer)
            write_json(snapshot / "manifest.json", original)
            prepared = root / "prepared"
            prepared.mkdir()
            data_path = prepared / "data.jsonl"
            data_path.write_text(json.dumps(asdict(two_step_record())) + "\n")
            prepared_config = {"model": config["model"]}
            write_config_snapshot(prepared / "config.yaml", prepared_config)
            write_json(
                prepared / "manifest.json",
                {
                    "config": prepared_config,
                    "records": 1,
                    "tokenizer_fingerprint": original["tokenizer_fingerprint"],
                    "data_file": data_path.name,
                    "data_file_sha256": file_sha256(data_path),
                    "config_file": "config.yaml",
                    "config_file_sha256": file_sha256(prepared / "config.yaml"),
                },
            )
            config.update(
                {
                    "method": METHOD,
                    "medoid": {"epsilon_rel": 1.0e-4, "epsilon_abs": 1.0e-8, "tau_b": 0.0},
                    "council": {
                        "k_max": None,
                        "tau_min": 0.5,
                        "tau_max": 2.0,
                        "temperature_schedule": "linear",
                        "alpha": 1.0,
                        "beta": 0.25,
                        "epsilon_m": 1.0e-6,
                    },
                    "stage2": {
                        "epochs": 1,
                        "micro_batch_size": 1,
                        "global_batch_size": 1,
                        "learning_rate": 2e-4,
                        "max_length": 128,
                        "max_grad_norm": 1.0,
                        "checkpoint_every_steps": 1,
                        "log_every_steps": 1,
                        "resume_from": None,
                    },
                    "optimizer": {
                        "name": "adamw",
                        "betas": [0.9, 0.999],
                        "eps": 1e-8,
                        "weight_decay": 0,
                    },
                    "scheduler": {"name": "cosine", "warmup_ratio": 0.1, "min_lr_ratio": 0},
                    "runtime": {"lm_head_chunk_tokens": 2, "dataloader_workers": 0},
                    "_project_root": str(root),
                }
            )
            config["paths"].update(
                {
                    "prepared": str(prepared),
                    "teacher_cache_dir": str(root / "cache"),
                    "output": str(root / "output"),
                }
            )

            def backbone(_config, device):
                with torch.random.fork_rng():
                    torch.manual_seed(17)
                    model = Qwen2ForCausalLM(copy.deepcopy(self.model.config))
                model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
                model.enable_input_require_grads()
                return model.to(device)

            context = DistributedContext(0, 0, 1, torch.device("cpu"))
            with (
                patch(
                    "cot_mtkd.stage2.teachers.snapshot_download", return_value=str(snapshot)
                ) as download,
                patch("cot_mtkd.models.multi_adapter.load_base_causal_lm", side_effect=backbone),
                patch("cot_mtkd.stage2.council_cache.load_tokenizer", return_value=tokenizer),
                patch("cot_mtkd.stage2.trainer.load_tokenizer", return_value=tokenizer),
            ):
                council = build_council_cache(config, context)
                imported_hash = file_sha256(Path(config["paths"]["stage1"]) / "adapter_states.pt")
                result = train_stage2(config, context)
                self.assertEqual(download.call_count, 1)
            self.assertFalse(council["stage1_training_dataset_identity_verified"])
            self.assertEqual(
                council["stage1_prepared_manifest_fingerprint"], "historical-training-data"
            )
            self.assertEqual(result["global_step"], 1)
            self.assertEqual(result["initial_expert_adapter"], council["selected_expert"])
            self.assertEqual(
                file_sha256(Path(config["paths"]["stage1"]) / "adapter_states.pt"),
                imported_hash,
            )

    def test_invalid_imports_do_not_publish_partial_artifacts(self):
        for change in ("checksum", "base", "alpha", "adapter_alpha", "unsupported"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                snapshot, config, _ = self.fixture(Path(directory))
                if change == "checksum":
                    config["teacher_source"]["weight_sha256"]["expert_0"] = "0" * 64
                elif change == "base":
                    config["model"]["name_or_path"] = "different-base"
                elif change == "alpha":
                    config["lora"]["alpha"] = 9
                else:
                    path = snapshot / "expert_0/adapter_config.json"
                    adapter = read_json(path)
                    adapter["lora_alpha" if change == "adapter_alpha" else "use_dora"] = (
                        9 if change == "adapter_alpha" else True
                    )
                    write_json(path, adapter)
                with (
                    patch("cot_mtkd.stage2.teachers.snapshot_download", return_value=str(snapshot)),
                    self.assertRaises((ValueError, RuntimeError)),
                ):
                    ensure_stage2_teachers(config)
                self.assertFalse(Path(config["paths"]["stage1"]).exists())

    def test_partial_tensor_pairs_and_nonfinite_tensors_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "adapter.safetensors"
            save_file({"base.q_proj.lora_A.weight": torch.ones(2, 8)}, path)
            with self.assertRaisesRegex(ValueError, "complete LoRA"):
                canonical_adapter_state(path, LORA)
            save_file({"base.q_proj.lora_A.weight": torch.full((2, 8), float("nan"))}, path)
            with self.assertRaisesRegex(ValueError, "Non-finite"):
                canonical_adapter_state(path, LORA)

    def test_historical_dataset_fingerprint_is_preserved_with_explicit_import_limitation(self):
        stage1 = {
            "artifact": "stage1_hub_import",
            "prepared_manifest_fingerprint": "old",
            "config": {"model": {"name_or_path": "fixture/model"}},
            "tokenizer_fingerprint": "tokenizer",
        }
        prepared = {"config": stage1["config"], "tokenizer_fingerprint": "tokenizer"}
        with self.assertLogs("cot_mtkd.stage2.teachers", level="WARNING"):
            require_teacher_dataset(stage1, prepared)
        self.assertEqual(stage1["prepared_manifest_fingerprint"], "old")
        prepared["tokenizer_fingerprint"] = "different"
        with self.assertRaisesRegex(RuntimeError, "tokenizer"):
            require_teacher_dataset(stage1, prepared)
        stage1["artifact"] = "stage1_checkpoint"
        with self.assertRaisesRegex(RuntimeError, "dataset mismatch"):
            require_teacher_dataset(stage1, prepared)

    def test_unpinned_revision_and_existing_other_council_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            _, config, _ = self.fixture(Path(directory))
            config["teacher_source"]["revision"] = "main"
            with self.assertRaisesRegex(ValueError, "commit SHA"):
                ensure_stage2_teachers(config)
            config["teacher_source"]["revision"] = "a" * 40
            destination = Path(config["paths"]["stage1"])
            write_json(destination / "manifest.json", {"artifact": "stage1_checkpoint"})
            with self.assertRaisesRegex(RuntimeError, "another source"):
                ensure_stage2_teachers(config)


if __name__ == "__main__":
    unittest.main()
