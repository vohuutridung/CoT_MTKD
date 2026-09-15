from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml
from safetensors.torch import save_file

from cot_mtkd.stage1.publish import publish_stage1
from cot_mtkd.utils.manifest import file_sha256, fingerprint, write_json


def completed_stage1(root: Path, expert_count: int = 2) -> Path:
    root.mkdir(parents=True)
    config = {
        "model": {
            "name_or_path": "Qwen/test-base",
            "revision": "base-revision",
        },
        "lora": {
            "rank": 2,
            "alpha": 2,
            "dropout": 0.05,
            "target_modules": ["q_proj"],
        },
        "stage1": {"num_experts": expert_count},
    }
    config_file = root / "config.yaml"
    config_file.write_text(yaml.safe_dump(config), encoding="utf-8")
    bundle = root / "final" / "adapter_states.pt"
    bundle.parent.mkdir(parents=True)
    torch.save({f"expert_{i}": {} for i in range(expert_count)}, bundle)
    names = [f"expert_{i}" for i in range(expert_count)]
    expert_files = {}
    for name in names:
        expert_dir = root / "final" / "adapters" / name
        expert_dir.mkdir(parents=True)
        (expert_dir / "adapter_config.json").write_text(
            json.dumps(
                {
                    "peft_type": "LORA",
                    "task_type": "CAUSAL_LM",
                    "r": 2,
                    "lora_alpha": 2,
                    "target_modules": ["q_proj"],
                    "base_model_name_or_path": "Qwen/test-base",
                }
            ),
            encoding="utf-8",
        )
        save_file(
            {
                "layer.lora_A.weight": torch.ones(2, 2),
                "layer.lora_B.weight": torch.zeros(2, 2),
            },
            expert_dir / "adapter_model.safetensors",
        )
        adapter_config = expert_dir / "adapter_config.json"
        adapter_weights = expert_dir / "adapter_model.safetensors"
        expert_files[name] = {
            "adapter_config": str(adapter_config.relative_to(root)),
            "adapter_config_sha256": file_sha256(adapter_config),
            "adapter_weights": str(adapter_weights.relative_to(root)),
            "adapter_weights_sha256": file_sha256(adapter_weights),
        }
    write_json(
        root / "manifest.json",
        {
            "schema_version": 1,
            "artifact": "stage1_checkpoint",
            "adapter_names": names,
            "expert_files": expert_files,
            "adapter_bundle": "final/adapter_states.pt",
            "adapter_bundle_sha256": file_sha256(bundle),
            "config_file": "config.yaml",
            "config_file_sha256": file_sha256(config_file),
            "config": config,
            "config_fingerprint": fingerprint(config),
        },
    )
    return root


class FakeHub:
    def __init__(self, omit: str | None = None) -> None:
        self.repo_calls: list[dict] = []
        self.commit_calls: list[dict] = []
        self.revisions: list[str] = []
        self.omit = omit

    def create_repo(self, **kwargs):
        self.repo_calls.append(kwargs)

    def create_commit(self, **kwargs):
        self.commit_calls.append(kwargs)
        return SimpleNamespace(oid=f"revision-{len(self.commit_calls)}")

    def list_repo_files(self, **kwargs):
        self.revisions.append(kwargs["revision"])
        files = {
            operation.path_in_repo for operation in self.commit_calls[-1]["operations"]
        }
        if self.omit:
            files.remove(self.omit)
        return sorted(files)


class PublishStage1Test(unittest.TestCase):
    def test_commits_all_experts_together_and_republishes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = completed_stage1(Path(temporary) / "stage1")
            hub = FakeHub()
            for expected_revision in ("revision-1", "revision-2"):
                result = publish_stage1(root, "user/experts", private=False, api=hub)
                self.assertEqual(result["revision"], expected_revision)
                self.assertEqual(hub.revisions[-1], expected_revision)
                self.assertEqual(
                    set(result["files"]),
                    {
                        "README.md",
                        "expert_0/adapter_config.json",
                        "expert_0/adapter_model.safetensors",
                        "expert_1/adapter_config.json",
                        "expert_1/adapter_model.safetensors",
                    },
                )
                self.assertFalse(any("checkpoint" in name for name in result["files"]))
            self.assertEqual(len(hub.commit_calls), 2)
            self.assertTrue(all(call["private"] is False for call in hub.repo_calls))
            self.assertTrue(all(call["repo_type"] == "model" for call in hub.commit_calls))

    def test_invalid_local_artifact_prevents_any_hub_write(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = completed_stage1(Path(temporary) / "stage1")
            (root / "final" / "adapters" / "expert_1" / "adapter_model.safetensors").unlink()
            hub = FakeHub()
            with self.assertRaises(FileNotFoundError):
                publish_stage1(root, "user/experts", api=hub)
            self.assertEqual(hub.repo_calls, [])

    def test_modified_expert_files_fail_before_hub_write(self) -> None:
        for filename in ("adapter_config.json", "adapter_model.safetensors"):
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as temporary:
                root = completed_stage1(Path(temporary) / "stage1")
                path = root / "final" / "adapters" / "expert_1" / filename
                if filename.endswith(".json"):
                    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
                else:
                    save_file(
                        {
                            "layer.lora_A.weight": torch.zeros(2, 2),
                            "layer.lora_B.weight": torch.ones(2, 2),
                        },
                        path,
                    )
                hub = FakeHub()
                with self.assertRaisesRegex(RuntimeError, "Artifact content hash mismatch"):
                    publish_stage1(root, "user/experts", api=hub)
                self.assertEqual(hub.repo_calls, [])

    def test_missing_or_redirected_expert_files_fail_before_hub_write(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = completed_stage1(Path(temporary) / "stage1")
            manifest_path = root / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["expert_files"]["expert_1"]["adapter_weights"] = (
                manifest["expert_files"]["expert_0"]["adapter_weights"]
            )
            write_json(manifest_path, manifest)
            hub = FakeHub()
            with self.assertRaisesRegex(ValueError, "unexpected paths"):
                publish_stage1(root, "user/experts", api=hub)
            self.assertEqual(hub.repo_calls, [])

    def test_missing_remote_file_fails_verification(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = completed_stage1(Path(temporary) / "stage1")
            hub = FakeHub(omit="expert_1/adapter_model.safetensors")
            with self.assertRaisesRegex(RuntimeError, "missing files"):
                publish_stage1(root, "user/experts", api=hub)


if __name__ == "__main__":
    unittest.main()
