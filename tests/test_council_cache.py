from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from test_output_space_online import output_config
from test_stage2_online import tiny_online_council, two_step_record
from cot_mtkd.stage1.kneedle import build_union_support, local_k_from_probe
from cot_mtkd.models.chunked_head import full_vocab_probe
from cot_mtkd.stage2.council_cache import (
    CACHE_VERSION,
    CouncilCache,
    cache_identity,
    compile_record,
    select_best_expert,
    validate_council_config,
)
from cot_mtkd.stage2.output_space import unpack_cached_target
from cot_mtkd.stage2.output_space_losses import support_from_probes
from cot_mtkd.utils.manifest import file_sha256, fingerprint, write_json


class CouncilCacheTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_support_reuses_phase1_excludes_gold_adds_gold_and_deduplicates(self):
        torch.manual_seed(71)
        head = torch.nn.Linear(5, 37)
        targets = torch.tensor([3, 11, 20])
        probes = [full_vocab_probe(torch.randn(3, 5), head, targets, 512, 2) for _ in range(3)]
        top_values = torch.stack([p[0] for p in probes])
        top_ids = torch.stack([p[1] for p in probes])
        self.assertFalse(bool((top_ids == targets[None, :, None]).any()))
        support, mask, stats = support_from_probes(top_values, top_ids, targets)
        k = torch.stack([local_k_from_probe(v)[1] for v in top_values])
        union, valid = build_union_support(top_ids, k)
        for t in range(3):
            actual = support[t, mask[t]].tolist()
            expected = sorted(set(union[t, valid[t]].tolist()) | {int(targets[t])})
            self.assertEqual(actual, expected)
            self.assertEqual(actual.count(int(targets[t])), 1)
        self.assertFalse(bool(stats["gold_present_before_add"].any()))
        # Defensive path with duplicate gold and duplicate alternatives.
        top_ids[:, :, 0] = targets
        top_ids[:, :, 1] = targets
        support, mask, stats = support_from_probes(top_values, top_ids, targets)
        self.assertTrue(bool(stats["gold_present_before_add"].all()))
        for t in range(3):
            actual = support[t, mask[t]].tolist()
            self.assertEqual(actual, sorted(set(actual)))
            self.assertEqual(actual.count(int(targets[t])), 1)
        self.assertFalse(support.requires_grad)

    def test_variable_support_and_detached_kneedle(self):
        values = torch.tensor(
            [[[9.0, 0.0, -1.0, -2.0], [9.0, 8.0, 7.0, 0.0]]] * 3, requires_grad=True
        )
        ids = torch.tensor(
            [
                [[1, 2, 3, 4], [1, 2, 3, 4]],
                [[1, 2, 3, 4], [5, 6, 7, 8]],
                [[1, 2, 3, 4], [9, 10, 11, 12]],
            ]
        )
        support, mask, stats = support_from_probes(values, ids, torch.tensor([0, 0]))
        self.assertEqual(mask.sum(-1).tolist(), [5, 13])
        self.assertFalse(stats["selected_k"].requires_grad)
        self.assertIsNone(values.grad)

    def test_best_expert_minimum_and_deterministic_ties(self):
        self.assertEqual(select_best_expert(torch.tensor([2.0, 1.0, 3.0])), 1)
        self.assertEqual(select_best_expert(torch.tensor([1.0, 1.0, 3.0])), 0)
        with self.assertRaises(ValueError):
            select_best_expert(torch.tensor([float("nan")]))

    def test_cache_roundtrip_preserves_support_js_target_and_checksums(self):
        model, names, _ = tiny_online_council(False)
        value, _, _ = compile_record(
            model, names, two_step_record(), output_config(), torch.device("cpu")
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_file(value, str(root / "sample.safetensors"))
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
            write_json(
                root / "manifest.json",
                {
                    "artifact": "stage2_council_cache",
                    "cache_version": CACHE_VERSION,
                    "fingerprint": key,
                    "identity": identity,
                    "records": 1,
                    "index_file": "index.json",
                    "index_file_sha256": file_sha256(root / "index.json"),
                },
            )
            loaded = CouncilCache(root, key).get("sample")
            for name in value:
                torch.testing.assert_close(loaded[name], value[name], atol=0, rtol=0)
            _, _, logp = unpack_cached_target(loaded)
            self.assertEqual(len(logp), len(names))
            # Every expert's reduced (support + tail) distribution is normalized.
            torch.testing.assert_close(
                logp.exp().sum(-1),
                torch.ones(logp.shape[:2], dtype=logp.dtype),
                atol=2e-7,
                rtol=2e-7,
            )
            with self.assertRaisesRegex(RuntimeError, "fingerprint mismatch"):
                CouncilCache(root, "wrong")
            (root / "sample.safetensors").write_bytes(b"corrupt")
            with self.assertRaisesRegex(RuntimeError, "checksum mismatch"):
                CouncilCache(root, key).get("sample")

    def test_identity_invalidates_static_changes_but_never_epoch_or_sft_weight(self):
        config = output_config()
        config["model"] = {"name_or_path": "fixture", "revision": "a", "dtype": "float32"}
        prepared = {"data_file_sha256": "data", "tokenizer_fingerprint": "tokenizer"}
        teachers = {"adapter_bundle_sha256": "weights"}
        original = fingerprint(cache_identity(config, prepared, teachers))
        for field, value in (
            ("epochs", 20),
            ("global_batch_size", 16),
            ("learning_rate", 0.001),
            ("resume_from", "abc"),
        ):
            changed = copy.deepcopy(config)
            changed["stage2"][field] = value
            self.assertEqual(original, fingerprint(cache_identity(changed, prepared, teachers)))
        for field, value in (
            ("sft_weight", 0.5),
            ("kd_weight", 0.0),
            ("rho_mapping", "constant"),
            ("rho_constant", 0.0),
            ("teachers", "best"),
            ("loss_normalization", "step"),
        ):
            changed = copy.deepcopy(config)
            changed["aggregation"][field] = value
            self.assertEqual(original, fingerprint(cache_identity(changed, prepared, teachers)))
        changed = copy.deepcopy(config)
        changed["stage2"]["student_init"] = "best_expert"
        self.assertEqual(original, fingerprint(cache_identity(changed, prepared, teachers)))
        for section, field, value in (
            ("aggregation", "js_temperature", 2.0),
            ("aggregation", "kd_temperature", 1.0),
            ("stage2", "max_length", 32),
            ("model", "revision", "b"),
        ):
            changed = copy.deepcopy(config)
            changed[section][field] = value
            self.assertNotEqual(original, fingerprint(cache_identity(changed, prepared, teachers)))
        teachers["adapter_bundle_sha256"] = "changed"
        self.assertNotEqual(original, fingerprint(cache_identity(config, prepared, teachers)))
        with patch("cot_mtkd.stage2.council_cache.file_sha256", return_value="changed-code"):
            self.assertNotEqual(original, fingerprint(cache_identity(config, prepared, teachers)))

    def test_old_config_and_independent_selection_rules_are_rejected(self):
        config = output_config()
        validate_council_config(config)
        for field, value in (
            ("search_k", 256),
            ("k_min", 4),
            ("temperature", 2.0),
            ("k_max", 64),
            ("sft_weight", -1),
            ("kd_temperature", float("nan")),
            ("kd_weight", -1),
            ("rho_mapping", "softmax"),
            ("rho_constant", 0.5),
            ("loss_normalization", "sample"),
            ("teachers", ""),
        ):
            changed = copy.deepcopy(config)
            changed["aggregation"][field] = value
            with self.assertRaises(ValueError):
                validate_council_config(changed)
        changed = copy.deepcopy(config)
        changed["aggregation"].update(rho_mapping="constant", rho_constant=1.5)
        with self.assertRaises(ValueError):
            validate_council_config(changed)
        changed = copy.deepcopy(config)
        changed["aggregation"].update(kd_weight=0.0, sft_weight=0.0)
        with self.assertRaises(ValueError):
            validate_council_config(changed)
        changed = copy.deepcopy(config)
        changed["stage2"]["student_init"] = "medoid"
        with self.assertRaises(ValueError):
            validate_council_config(changed)

    def test_teacher_selection_resolves_council_best_and_named_expert(self):
        from cot_mtkd.stage2.council_cache import resolve_teacher_indices

        names = ["expert_0", "expert_1", "expert_2", "expert_3"]
        self.assertEqual(resolve_teacher_indices("all", names, 2), [0, 1, 2, 3])
        self.assertEqual(resolve_teacher_indices("best", names, 2), [2])
        self.assertEqual(resolve_teacher_indices("expert_3", names, 2), [3])
        with self.assertRaises(ValueError):
            resolve_teacher_indices("expert_9", names, 2)

    def test_default_configs_and_global_batch_boundaries(self):
        import yaml
        from cot_mtkd.stage2.trainer import gradient_accumulation_steps, _validate_stage2_config

        root = Path(__file__).resolve().parents[1]
        outputs = set()
        for filename in (
            "qwen25_7b_output_space.yaml",
            "qwen25_7b_output_space_local.yaml",
            "qwen25_7b_output_space_geometric.yaml",
            "qwen25_7b_output_space_arithmetic.yaml",
            "qwen25_7b_output_space_single.yaml",
            "qwen25_7b_output_space_sft.yaml",
        ):
            config = yaml.safe_load((root / "configs" / "stage2" / filename).read_text())
            _validate_stage2_config(config)
            self.assertEqual(config["stage2"]["epochs"], 1)
            self.assertEqual(config["stage2"]["global_batch_size"], 8)
            self.assertEqual(config["stage2"]["micro_batch_size"], 1)
            self.assertEqual(config["stage2"]["student_init"], "base")
            self.assertEqual(config["aggregation"]["js_temperature"], 1.0)
            self.assertEqual(config["aggregation"]["kd_temperature"], 1.0)
            self.assertEqual(config["aggregation"]["loss_normalization"], "token")
            self.assertNotIn("medoid", config["paths"])
            outputs.add(config["paths"]["output"])
            if filename != "qwen25_7b_output_space_local.yaml":
                # Controls share the main council cache and differ only at training time.
                self.assertEqual(
                    config["paths"]["teacher_cache_dir"], "artifacts/teacher_cache/output_space"
                )
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
        self.assertEqual(len(outputs), 6)

    def test_initializer_copies_full_winning_adapter_and_instantiates_only_student(self):
        from types import SimpleNamespace
        from cot_mtkd.models.multi_adapter import extract_adapter_state
        from cot_mtkd.stage2.initialization import create_cached_student

        model, names, _ = tiny_online_council(False)
        winner = extract_adapter_state(model, names[1])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            torch.save(winner, root / "best.pt")
            cache = SimpleNamespace(
                directory=root,
                manifest={
                    "adapter_names": names,
                    "selected_expert": names[1],
                    "best_expert_file": "best.pt",
                    "best_expert_file_sha256": file_sha256(root / "best.pt"),
                },
            )
            config = output_config()
            config["model"], config["lora"], config["seed"] = {}, {}, 42
            # Default (base): the fresh student LoRA is left untouched.
            fresh = extract_adapter_state(model, "student")
            with patch(
                "cot_mtkd.stage2.initialization.create_student_model", return_value=model
            ) as create:
                actual, _, _ = create_cached_student(
                    config, SimpleNamespace(device=torch.device("cpu")), cache
                )
            create.assert_called_once_with({}, {}, torch.device("cpu"), 42)
            for key, tensor in extract_adapter_state(actual, "student").items():
                torch.testing.assert_close(tensor, fresh[key], atol=0, rtol=0)
            config["stage2"]["student_init"] = "best_expert"
            with patch("cot_mtkd.stage2.initialization.create_student_model", return_value=model):
                actual, _, _ = create_cached_student(
                    config, SimpleNamespace(device=torch.device("cpu")), cache
                )
            for key, tensor in extract_adapter_state(actual, "student").items():
                torch.testing.assert_close(tensor, winner[key], atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
