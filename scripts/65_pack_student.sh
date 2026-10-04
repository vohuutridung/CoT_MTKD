#!/usr/bin/env bash
# Pack the Stage-2 student for evaluation on another machine.
#   STAGE2_DIR  Stage-2 output directory (default artifacts/stage2/output_space)
#   PACK_OUT    archive path (default artifacts/student_<run>.tar.gz)
#   HF_REPO     optional: also upload to this Hugging Face model repo (private)
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
STAGE2_DIR="${STAGE2_DIR:-artifacts/stage2/output_space}"
ADAPTER="final/adapters/student"
[[ -f "$STAGE2_DIR/manifest.json" && -f "$STAGE2_DIR/$ADAPTER/adapter_config.json" ]] || {
  echo "Stage 2 is not finished: need $STAGE2_DIR/manifest.json and $STAGE2_DIR/$ADAPTER" >&2
  exit 1
}
PACK_OUT="${PACK_OUT:-artifacts/student_$(basename "$STAGE2_DIR").tar.gz}"
mkdir -p "$(dirname "$PACK_OUT")"
tar -czf "$PACK_OUT" -C "$STAGE2_DIR" manifest.json config.yaml "$ADAPTER"
echo "Packed $(du -h "$PACK_OUT" | cut -f1) -> $PACK_OUT"
echo "On the eval machine: mkdir -p student && tar -xzf $(basename "$PACK_OUT") -C student"
echo "                     EVAL_ADAPTER=\$PWD/student ./project_commands.sh evaluate"
if [[ -n "${HF_REPO:-}" ]]; then
  "$PYTHON_BIN" - "$HF_REPO" "$STAGE2_DIR" "$ADAPTER" <<'PY'
import sys
from huggingface_hub import HfApi

repo, stage2, adapter = sys.argv[1:]
api = HfApi()
api.create_repo(repo, private=True, exist_ok=True)
commit = api.upload_folder(
    repo_id=repo,
    folder_path=stage2,
    allow_patterns=["manifest.json", "config.yaml", f"{adapter}/*"],
    commit_message="Stage-2 student adapter",
)
print(f"Uploaded to {repo} at {commit.oid}")
print(f"On the eval machine: EVAL_ADAPTER={repo} ./project_commands.sh evaluate "
      f"--set adapter.revision={commit.oid}")
PY
fi
