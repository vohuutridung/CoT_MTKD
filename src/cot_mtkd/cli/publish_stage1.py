"""Publish a completed Stage-1 run's LoRA experts to Hugging Face."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from ..config import load_config
from ..stage1.publish import publish_stage1


def _environment_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true or false")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Publish completed Stage-1 PEFT LoRA experts to one HF model repo"
    )
    parser.add_argument(
        "--config",
        default="configs/stage1/qwen25_7b_m5.yaml",
        help="Stage-1 YAML config, used to locate its output directory",
    )
    parser.add_argument(
        "--stage1-dir",
        type=Path,
        help="Completed Stage-1 artifact directory (overrides --config output)",
    )
    parser.add_argument(
        "--repo-id",
        default=os.environ.get("HF_REPO_ID"),
        required=not bool(os.environ.get("HF_REPO_ID")),
        help="Hugging Face model repo ID; defaults to HF_REPO_ID",
    )
    visibility = parser.add_mutually_exclusive_group()
    visibility.add_argument("--private", dest="private", action="store_true")
    visibility.add_argument("--public", dest="private", action="store_false")
    parser.set_defaults(private=None)
    arguments = parser.parse_args()
    private = (
        arguments.private
        if arguments.private is not None
        else _environment_bool("HF_REPO_PRIVATE", True)
    )
    stage1_dir = (
        arguments.stage1_dir
        if arguments.stage1_dir is not None
        else Path(load_config(arguments.config)["paths"]["output"])
    )
    result = publish_stage1(
        stage1_dir,
        arguments.repo_id,
        private=private,
        token=os.environ.get("HF_TOKEN"),
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
