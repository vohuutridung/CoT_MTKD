"""Locate the Stage-2 student as a plain PEFT adapter directory.

Accepted sources, so evaluation can run on a machine that only received the
student weights:

* a Stage-2 output directory (``manifest.json`` + ``final/adapters/student``);
* an unpacked ``scripts/65_pack_student.sh`` archive (same layout);
* any PEFT adapter directory (``adapter_config.json``), e.g. ``final/adapters/student``;
* a Hugging Face Hub repo id, optionally with ``adapter.revision`` and ``adapter.subfolder``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

STUDENT_SUBDIR = Path("final") / "adapters" / "student"


def _peft_dir(path: Path) -> Optional[Path]:
    for candidate in (path, path / STUDENT_SUBDIR, path / "adapters" / "student"):
        if (candidate / "adapter_config.json").is_file():
            return candidate
    return None


def _stage2_manifest(adapter_dir: Path) -> Optional[dict[str, Any]]:
    for parent in (adapter_dir, *adapter_dir.parents[:3]):
        manifest = parent / "manifest.json"
        if manifest.is_file():
            data = json.loads(manifest.read_text(encoding="utf-8"))
            if data.get("artifact") == "stage2_checkpoint":
                return data
    return None


def resolve_adapter(
    adapter: dict[str, Any], project_root: Path
) -> tuple[Optional[Path], Optional[dict[str, Any]]]:
    """Return the PEFT adapter directory and the Stage-2 manifest if present.

    ``adapter.path`` set to ``null`` or ``base`` evaluates the base model.
    """
    source = adapter.get("path")
    if source in (None, "", "base"):
        return None, None
    local = Path(str(source)).expanduser()
    if not local.is_absolute():
        local = project_root / local
    if local.exists():
        resolved = _peft_dir(local)
        if resolved is None:
            raise FileNotFoundError(
                f"No adapter_config.json in {local}, {local / STUDENT_SUBDIR} "
                f"or {local / 'adapters' / 'student'}"
            )
        return resolved, _stage2_manifest(resolved)

    from huggingface_hub import snapshot_download

    subfolder = adapter.get("subfolder")
    snapshot = Path(
        snapshot_download(
            repo_id=str(source),
            revision=adapter.get("revision"),
            allow_patterns=[f"{subfolder}/*"] if subfolder else None,
        )
    )
    root = snapshot / subfolder if subfolder else snapshot
    resolved = _peft_dir(root)
    if resolved is None:
        raise FileNotFoundError(f"No PEFT adapter found in Hub repo {source}/{subfolder or ''}")
    return resolved, _stage2_manifest(resolved)


def adapter_rank(adapter_dir: Optional[Path]) -> int:
    if adapter_dir is None:
        return 16
    config = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    ranks = [int(config.get("r", 16))]
    ranks.extend(int(value) for value in (config.get("rank_pattern") or {}).values())
    return max(ranks)
