from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any, Iterable

import torch
import yaml


def canonical_json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str
    )


def fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: str | Path, chunk_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def files_fingerprint(paths: Iterable[str | Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(Path(item) for item in paths):
        digest.update(str(path.name).encode("utf-8"))
        digest.update(file_sha256(path).encode("ascii"))
    return digest.hexdigest()


def git_commit(root: str | Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def runtime_metadata(root: str | Path) -> dict[str, Any]:
    distributions = (
        "torch",
        "transformers",
        "peft",
        "datasets",
        "safetensors",
        "math-verify",
    )
    versions: dict[str, str | None] = {}
    for distribution in distributions:
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    return {
        "git_commit": git_commit(root),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cudnn": torch.backends.cudnn.version(),
        "world_size": int(os.environ.get("WORLD_SIZE", "1")),
        "packages": versions,
    }


def write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(
            value, handle, ensure_ascii=False, indent=2, sort_keys=True, default=str
        )
        handle.write("\n")
    temporary.replace(destination)


def write_config_snapshot(path: str | Path, value: dict[str, Any]) -> Path:
    """Write-once resolved YAML; refuse to mutate an existing run definition."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        with destination.open("r", encoding="utf-8") as handle:
            existing = yaml.safe_load(handle)
        if existing != value:
            raise RuntimeError(
                f"Refusing to overwrite immutable run config {destination}"
            )
        return destination
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(value, handle, sort_keys=True, allow_unicode=True)
    temporary.replace(destination)
    return destination


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def require_fingerprint(manifest: dict[str, Any], key: str, expected: str) -> None:
    actual = manifest.get(key)
    if actual != expected:
        raise RuntimeError(
            f"Artifact fingerprint mismatch for {key}: expected {expected}, got {actual}"
        )


def require_file_sha256(
    root: str | Path,
    manifest: dict[str, Any],
    file_key: str,
    sha256_key: str,
) -> Path:
    """Resolve and verify a manifest-owned file before it is consumed."""
    if file_key not in manifest or sha256_key not in manifest:
        raise RuntimeError(f"Manifest is missing {file_key!r} or {sha256_key!r}")
    path = Path(root) / str(manifest[file_key])
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = file_sha256(path)
    expected = str(manifest[sha256_key])
    if actual != expected:
        raise RuntimeError(
            f"Artifact content hash mismatch for {path}: expected {expected}, got {actual}"
        )
    return path
