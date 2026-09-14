from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterable

import yaml


def project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


def load_config(path: str | Path, overrides: Iterable[str] = ()) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise TypeError(f"Top-level YAML value must be a mapping: {config_path}")
    config = copy.deepcopy(loaded)
    for override in overrides:
        apply_override(config, override)
    config["_config_path"] = str(config_path)
    config["_project_root"] = str(project_root())
    _resolve_path_fields(config)
    return config


def apply_override(config: dict[str, Any], expression: str) -> None:
    if "=" not in expression:
        raise ValueError(f"Override must be KEY=VALUE, received: {expression!r}")
    dotted_key, raw_value = expression.split("=", 1)
    value = yaml.safe_load(raw_value)
    cursor: dict[str, Any] = config
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        nested = cursor.setdefault(part, {})
        if not isinstance(nested, dict):
            raise TypeError(
                f"Cannot set nested override beneath non-mapping key {part!r}"
            )
        cursor = nested
    cursor[parts[-1]] = value


def _resolve_path_fields(config: dict[str, Any]) -> None:
    root = Path(config["_project_root"])
    paths = config.get("paths")
    if isinstance(paths, dict):
        for key, value in list(paths.items()):
            if isinstance(value, str):
                candidate = Path(value).expanduser()
                paths[key] = str(
                    candidate if candidate.is_absolute() else root / candidate
                )
    if isinstance(config.get("output_dir"), str):
        candidate = Path(config["output_dir"]).expanduser()
        config["output_dir"] = str(
            candidate if candidate.is_absolute() else root / candidate
        )
    dataset = config.get("dataset")
    if isinstance(dataset, dict) and isinstance(dataset.get("local_json"), str):
        candidate = Path(dataset["local_json"]).expanduser()
        dataset["local_json"] = str(
            candidate if candidate.is_absolute() else root / candidate
        )
    benchmarks = config.get("benchmarks")
    if isinstance(benchmarks, list):
        for benchmark in benchmarks:
            if not isinstance(benchmark, dict) or not isinstance(
                benchmark.get("local_json"), str
            ):
                continue
            candidate = Path(benchmark["local_json"]).expanduser()
            benchmark["local_json"] = str(
                candidate if candidate.is_absolute() else root / candidate
            )


def require(config: dict[str, Any], dotted_key: str) -> Any:
    cursor: Any = config
    for part in dotted_key.split("."):
        if not isinstance(cursor, dict) or part not in cursor:
            raise KeyError(f"Required configuration key is missing: {dotted_key}")
        cursor = cursor[part]
    return cursor
