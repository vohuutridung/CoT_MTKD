from __future__ import annotations

import logging
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

import torch
import yaml
from filelock import FileLock
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

from ..models.multi_adapter import require_same_model_source
from ..utils.manifest import (
    file_sha256,
    fingerprint,
    read_json,
    require_file_sha256,
    write_config_snapshot,
    write_json,
)

LOGGER = logging.getLogger(__name__)


def _validate_config(config: dict[str, Any], source_config: dict[str, Any]) -> None:
    require_same_model_source(config["model"], source_config["model"], "Hub teachers/Phase 2")
    for key in ("rank", "alpha", "target_modules"):
        actual, expected = config["lora"][key], source_config["lora"][key]
        if key == "target_modules":
            actual, expected = sorted(actual), sorted(expected)
        if actual != expected:
            raise RuntimeError(f"Hub teacher/student LoRA {key} mismatch")


def _validate_adapter(config: dict[str, Any], source_config: dict[str, Any]) -> None:
    lora = source_config["lora"]
    expected = {
        "peft_type": "LORA", "task_type": "CAUSAL_LM", "bias": "none",
        "base_model_name_or_path": source_config["model"]["name_or_path"],
        "r": lora["rank"], "lora_alpha": lora["alpha"], "lora_dropout": lora["dropout"],
    }
    if any(config.get(key) != value for key, value in expected.items()):
        raise RuntimeError("Hub adapter_config.json does not match the Stage-1 configuration")
    if sorted(config["target_modules"]) != sorted(lora["target_modules"]):
        raise RuntimeError("Hub adapter target modules do not match Stage 1")
    # The council constructor supports ordinary uniform-rank LoRA only.
    unsupported = (
        "use_dora", "use_rslora", "use_qalora", "lora_bias", "fan_in_fan_out",
        "rank_pattern", "alpha_pattern", "modules_to_save", "layers_to_transform",
        "layer_replication", "target_parameters", "trainable_token_indices",
        "alora_invocation_tokens", "use_bdlora", "ensure_weight_tying",
    )
    if any(config.get(key) for key in unsupported):
        raise ValueError("Hub adapter uses LoRA features unsupported by the Phase-2 council")
    revision = config.get("revision")
    if revision is not None and revision != source_config["model"].get("revision"):
        raise RuntimeError("Hub adapter base-model revision mismatch")


def canonical_adapter_state(path: Path, lora: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Convert PEFT's saved A/B keys to the existing shared-backbone bundle format."""
    state = {}
    pairs: dict[str, dict[str, torch.Tensor]] = {}
    for key, value in load_file(str(path), device="cpu").items():
        match = re.fullmatch(r"(.+)\.lora_([AB])\.weight", key)
        if not match or value.ndim != 2 or not value.is_floating_point():
            raise ValueError(f"Unsupported PEFT tensor: {key}")
        module, side = match.groups()
        if module.rsplit(".", 1)[-1] not in lora["target_modules"]:
            raise ValueError(f"Unexpected LoRA target: {module}")
        if value.shape[0 if side == "A" else 1] != int(lora["rank"]):
            raise ValueError(f"LoRA rank mismatch for {key}")
        if not torch.isfinite(value).all():
            raise ValueError(f"Non-finite adapter tensor: {key}")
        pairs.setdefault(module, {})[side] = value
        state[f"{module}.lora_{side}.{{adapter}}.weight"] = value
    if not pairs or any(set(pair) != {"A", "B"} for pair in pairs.values()):
        raise ValueError("Adapter must contain a nonempty set of complete LoRA A/B pairs")
    if {module.rsplit(".", 1)[-1] for module in pairs} != set(lora["target_modules"]):
        raise ValueError("Adapter is missing configured LoRA target modules")
    return state


def ensure_stage2_teachers(config: dict[str, Any]) -> dict[str, Any]:
    """Import pinned Hub teachers once; local Stage-1 artifacts remain supported."""
    source = config.get("teacher_source") or {"type": "local"}
    destination = Path(config["paths"]["stage1"])
    if source.get("type", "local") == "local":
        return read_json(destination / "manifest.json")
    if source.get("type") != "huggingface":
        raise ValueError("teacher_source.type must be local or huggingface")
    revision = str(source["revision"])
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Pin teacher_source.revision to a full 40-character Hub commit SHA")
    names = list(source["adapter_names"])
    if len(names) < 2 or names != [f"expert_{i}" for i in range(len(names))]:
        raise ValueError("Hub adapter_names must be consecutive expert_0, expert_1, ...")
    identity = {key: source[key] for key in ("repo_id", "revision", "adapter_names")}
    expected_hashes = source.get("weight_sha256", {})
    if expected_hashes and set(expected_hashes) != set(names):
        raise ValueError("teacher_source.weight_sha256 must cover every expert")
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Protect imports when multiple torchrun ranks start together. No GPU is loaded.
    with FileLock(str(destination) + ".lock"):
        if (destination / "manifest.json").exists():
            manifest = read_json(destination / "manifest.json")
            if manifest.get("artifact") != "stage1_hub_import" or manifest.get("hub_source") != identity:
                raise RuntimeError("Teacher destination contains another source; use a new paths.stage1")
            _validate_config(config, manifest["config"])
            require_file_sha256(destination, manifest, "adapter_bundle", "adapter_bundle_sha256")
            require_file_sha256(destination, manifest, "config_file", "config_file_sha256")
            for filename, checksum in manifest["hub_file_sha256"].items():
                if file_sha256(destination / filename) != checksum:
                    raise RuntimeError(f"Imported Hub file changed: {filename}")
            for name, checksum in expected_hashes.items():
                if manifest["hub_file_sha256"][f"source/{name}/adapter_model.safetensors"] != checksum:
                    raise RuntimeError(f"Pinned teacher weight checksum mismatch: {name}")
            return manifest
        if destination.exists():
            raise RuntimeError(f"Refusing to overwrite incomplete teacher directory: {destination}")
        filenames = ["manifest.json", "config.yaml"] + [
            f"{name}/{filename}" for name in names
            for filename in ("adapter_config.json", "adapter_model.safetensors")
        ]
        LOGGER.info("Downloading teachers from %s at %s", source["repo_id"], revision)
        snapshot = Path(snapshot_download(
            repo_id=source["repo_id"], revision=revision, allow_patterns=filenames,
        ))
        original = read_json(snapshot / "manifest.json")
        with (snapshot / "config.yaml").open(encoding="utf-8") as handle:
            source_config = yaml.safe_load(handle)
        if source_config != original["config"] or original["adapter_names"] != names:
            raise RuntimeError("Hub configuration/manifest/council mismatch")
        _validate_config(config, source_config)
        bundle = {}
        for name in names:
            _validate_adapter(read_json(snapshot / name / "adapter_config.json"), source_config)
            weight_path = snapshot / name / "adapter_model.safetensors"
            if name in expected_hashes and file_sha256(weight_path) != expected_hashes[name]:
                raise RuntimeError(f"Pinned teacher weight checksum mismatch: {name}")
            bundle[name] = canonical_adapter_state(weight_path, source_config["lora"])
        structures = [{key: tuple(value.shape) for key, value in state.items()} for state in bundle.values()]
        if any(structure != structures[0] for structure in structures[1:]):
            raise RuntimeError("Hub experts have different parameter structures")
        with tempfile.TemporaryDirectory(prefix=f".{destination.name}-", dir=destination.parent) as work:
            staging = Path(work) / "import"
            staging.mkdir()
            hashes = {}
            for filename in filenames:
                target = staging / "source" / filename
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(snapshot / filename, target)
                hashes[str(target.relative_to(staging))] = file_sha256(target)
            bundle_path = staging / "adapter_states.pt"
            torch.save(bundle, bundle_path)
            config_path = write_config_snapshot(staging / "config.yaml", source_config)
            manifest = {
                "schema_version": 1, "artifact": "stage1_hub_import",
                "adapter_names": names,
                "adapter_bundle": bundle_path.name, "adapter_bundle_sha256": file_sha256(bundle_path),
                "config": source_config, "config_fingerprint": fingerprint(source_config),
                "config_file": config_path.name, "config_file_sha256": file_sha256(config_path),
                "tokenizer_fingerprint": original["tokenizer_fingerprint"],
                "prepared_manifest_fingerprint": original["prepared_manifest_fingerprint"],
                "hub_source": identity, "hub_file_sha256": hashes,
                "original_manifest_fingerprint": fingerprint(original),
                "original_config_checksum_matches_upload": (
                    original["config_file_sha256"] == file_sha256(snapshot / "config.yaml")
                ),
                "global_step": original.get("global_step"),
                "dataset_identity": "original_prepared_manifest_not_uploaded",
            }
            write_json(staging / "manifest.json", manifest)
            staging.rename(destination)
        LOGGER.info("Imported %d teacher adapters into %s", len(names), destination)
        return manifest


def require_teacher_dataset(stage1: dict[str, Any], prepared: dict[str, Any]) -> None:
    if stage1["prepared_manifest_fingerprint"] == fingerprint(prepared):
        return
    if stage1.get("artifact") != "stage1_hub_import":
        raise RuntimeError("Stage-1 checkpoint/prepared dataset mismatch during council preprocessing")
    # A published PEFT council can be scored on newly prepared data. Keep the
    # original training fingerprint; never pretend that re-preparation matches it.
    require_same_model_source(prepared["config"]["model"], stage1["config"]["model"], "Prepared/Hub teachers")
    if prepared["tokenizer_fingerprint"] != stage1["tokenizer_fingerprint"]:
        raise RuntimeError("Prepared tokenizer does not match the imported teachers")
    LOGGER.warning(
        "Hub teachers retain their original Phase-1 dataset fingerprint. The original prepared "
        "manifest was not uploaded, so exact training-data identity cannot be verified. "
        "Medoid and Phase 2 will be bound to the verified current prepared corpus."
    )
