"""Publish completed Stage-1 LoRA experts to one Hugging Face model repository."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import yaml
from huggingface_hub import CommitOperationAdd, HfApi
from safetensors import safe_open

from ..utils.manifest import fingerprint, read_json, require_file_sha256

_ADAPTER_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*\Z")


def _manifest_file(
    root: Path, manifest: dict[str, Any], file_key: str, sha256_key: str
) -> Path:
    relative = Path(str(manifest.get(file_key, "")))
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError(f"Unsafe Stage-1 manifest path: {file_key}")
    destination = (root / relative).resolve()
    if not destination.is_relative_to(root.resolve()):
        raise ValueError(f"Stage-1 manifest path escapes artifact directory: {file_key}")
    return require_file_sha256(root, manifest, file_key, sha256_key)


def _validated_artifacts(stage1_dir: Path) -> tuple[dict[str, Any], list[tuple[str, Path]]]:
    root = stage1_dir.resolve()
    manifest = read_json(root / "manifest.json")
    if not isinstance(manifest, dict) or manifest.get("artifact") != "stage1_checkpoint":
        raise ValueError("Expected a completed Stage-1 checkpoint manifest")
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported Stage-1 manifest schema version")
    config = manifest.get("config")
    if not isinstance(config, dict) or manifest.get("config_fingerprint") != fingerprint(config):
        raise ValueError("Stage-1 manifest configuration fingerprint is invalid")
    snapshot = _manifest_file(root, manifest, "config_file", "config_file_sha256")
    with snapshot.open("r", encoding="utf-8") as handle:
        if yaml.safe_load(handle) != config:
            raise ValueError("Stage-1 config snapshot differs from its manifest")
    _manifest_file(root, manifest, "adapter_bundle", "adapter_bundle_sha256")

    names = manifest.get("adapter_names")
    if (
        not isinstance(names, list)
        or not names
        or any(not isinstance(name, str) or not _ADAPTER_NAME.fullmatch(name) for name in names)
        or len(names) != len(set(names))
    ):
        raise ValueError("Stage-1 manifest has invalid adapter names")
    if len(names) != int(config["stage1"]["num_experts"]):
        raise ValueError("Stage-1 expert count differs from its configuration")
    expert_files = manifest.get("expert_files")
    if not isinstance(expert_files, dict) or set(expert_files) != set(names):
        raise ValueError("Stage-1 manifest must list files for every expert")

    adapters_root = (root / "final" / "adapters").resolve()
    if not adapters_root.is_relative_to(root) or not adapters_root.is_dir():
        raise ValueError("Stage-1 PEFT adapter directory is missing or unsafe")
    lora = config["lora"]
    base_model = config["model"]["name_or_path"]
    files: list[tuple[str, Path]] = []
    for name in names:
        specification = expert_files[name]
        if not isinstance(specification, dict):
            raise ValueError(f"Invalid Stage-1 expert file specification: {name}")
        expected_config = f"final/adapters/{name}/adapter_config.json"
        expected_weights = f"final/adapters/{name}/adapter_model.safetensors"
        if (
            specification.get("adapter_config") != expected_config
            or specification.get("adapter_weights") != expected_weights
        ):
            raise ValueError(f"Stage-1 expert files point to unexpected paths: {name}")
        expert_dir = (adapters_root / name).resolve()
        if not expert_dir.is_relative_to(adapters_root) or not expert_dir.is_dir():
            raise ValueError(f"Missing or unsafe PEFT directory for {name}")
        config_path = _manifest_file(
            root, specification, "adapter_config", "adapter_config_sha256"
        ).resolve()
        weights_path = _manifest_file(
            root, specification, "adapter_weights", "adapter_weights_sha256"
        ).resolve()
        for path in (config_path, weights_path):
            if not path.is_relative_to(expert_dir) or not path.is_file():
                raise ValueError(f"Missing or unsafe PEFT file for {name}: {path.name}")
        with config_path.open("r", encoding="utf-8") as handle:
            peft_config = json.load(handle)
        if (
            peft_config.get("peft_type") != "LORA"
            or peft_config.get("task_type") != "CAUSAL_LM"
            or int(peft_config.get("r", -1)) != int(lora["rank"])
            or int(peft_config.get("lora_alpha", -1)) != int(lora["alpha"])
            or set(peft_config.get("target_modules", [])) != set(lora["target_modules"])
        ):
            raise ValueError(f"PEFT configuration does not match Stage 1: {name}")
        saved_base = peft_config.get("base_model_name_or_path")
        if saved_base and saved_base != base_model:
            raise ValueError(f"PEFT base model does not match Stage 1: {name}")
        with safe_open(weights_path, framework="pt", device="cpu") as handle:
            keys = handle.keys()
        if not keys or not any("lora_A" in key for key in keys) or not any(
            "lora_B" in key for key in keys
        ):
            raise ValueError(f"PEFT weights do not contain LoRA factors: {name}")
        files.extend(
            [
                (f"{name}/adapter_config.json", config_path),
                (f"{name}/adapter_model.safetensors", weights_path),
            ]
        )
    return manifest, files


def _model_card(manifest: dict[str, Any]) -> bytes:
    config = manifest["config"]
    model = config["model"]
    lora = config["lora"]
    names = manifest["adapter_names"]
    text = (
        "---\n"
        f"base_model: {model['name_or_path']}\n"
        "tags:\n"
        "- peft\n"
        "- lora\n"
        "- cot-mtkd\n"
        "---\n\n"
        "# CoT-MTKD Stage-1 LoRA experts\n\n"
        "This repository contains the independently named Stage-1 LoRA experts "
        "as PEFT adapters. Each expert has its own directory with "
        "`adapter_config.json` and `adapter_model.safetensors`. The base model "
        "weights and training checkpoints are not included.\n\n"
        f"Base model: `{model['name_or_path']}`\n\n"
        f"Base revision: `{model.get('revision') or 'unspecified'}`\n\n"
        f"Experts: {', '.join(f'`{name}`' for name in names)}\n\n"
        f"LoRA rank: {lora['rank']}; alpha: {lora['alpha']}; "
        f"target modules: {', '.join(lora['target_modules'])}.\n\n"
        f"Stage-1 configuration fingerprint: `{manifest['config_fingerprint']}`.\n"
    )
    return text.encode("utf-8")


def publish_stage1(
    stage1_dir: str | Path,
    repo_id: str,
    *,
    private: bool = True,
    token: str | None = None,
    api: HfApi | None = None,
) -> dict[str, Any]:
    """Validate local experts, commit them together, and verify the new Hub revision.

    ``private`` controls visibility only when the model repository is first created.
    An existing repository retains its current visibility.
    """
    if not repo_id or not repo_id.strip():
        raise ValueError("A Hugging Face model repo ID is required")
    manifest, files = _validated_artifacts(Path(stage1_dir))
    hub = api if api is not None else HfApi(token=token)
    hub.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    operations = [
        CommitOperationAdd(path_in_repo=remote_path, path_or_fileobj=local_path)
        for remote_path, local_path in files
    ]
    operations.append(
        CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=_model_card(manifest))
    )
    commit = hub.create_commit(
        repo_id=repo_id,
        repo_type="model",
        operations=operations,
        commit_message=f"Publish Stage-1 LoRA experts {manifest['config_fingerprint'][:12]}",
    )
    revision = getattr(commit, "oid", None)
    if not revision:
        raise RuntimeError("Hugging Face did not return a commit revision")
    expected = {remote_path for remote_path, _ in files} | {"README.md"}
    remote_files = set(
        hub.list_repo_files(repo_id=repo_id, repo_type="model", revision=revision)
    )
    missing = expected - remote_files
    if missing:
        raise RuntimeError(f"Published Stage-1 revision is missing files: {sorted(missing)}")
    return {
        "repo_id": repo_id,
        "revision": revision,
        "files": sorted(expected),
        "url": f"https://huggingface.co/{repo_id}/tree/{revision}",
    }
