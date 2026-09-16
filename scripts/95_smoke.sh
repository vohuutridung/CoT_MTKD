#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

# Tiny Qwen2.5-0.5B-Instruct inputs, then the real README `full` command:
# setup -> test -> prepare -> stage1 -> supervision -> stage2 -> evaluate.

SMOKE_PREPARED="${SMOKE_PREPARED:-artifacts/prepared/s1k_1_1/data.jsonl}"
SMOKE_RECORDS="${SMOKE_RECORDS:-8}"
SMOKE_EVAL_RECORDS="${SMOKE_EVAL_RECORDS:-2}"
SMOKE_DIR="${SMOKE_DIR:-artifacts/smoke}"

export DATA_CONFIG="${DATA_CONFIG:-configs/smoke/data.yaml}"
export STAGE1_CONFIG="${STAGE1_CONFIG:-configs/smoke/stage1.yaml}"
export SIGNALS_CONFIG="${SIGNALS_CONFIG:-configs/smoke/signals.yaml}"
export STAGE2_CONFIG="${STAGE2_CONFIG:-configs/smoke/stage2.yaml}"
export EVAL_CONFIG="${EVAL_CONFIG:-configs/smoke/eval.yaml}"

write_smoke_inputs() {
  if [[ ! -f "$SMOKE_PREPARED" ]]; then
    echo "Prepared dataset not found: $SMOKE_PREPARED" >&2
    echo "Run the 7B prepare once, or point SMOKE_PREPARED at a jsonl with question/thinking/attempt/solution." >&2
    exit 1
  fi
  mkdir -p "$SMOKE_DIR"
  echo "[smoke] slicing $SMOKE_RECORDS shortest records from $SMOKE_PREPARED"
  "$PYTHON_BIN" - "$SMOKE_PREPARED" "$SMOKE_DIR" "$SMOKE_RECORDS" "$SMOKE_EVAL_RECORDS" <<'PY'
import json
import sys
from pathlib import Path

src, dest, n_train, n_eval = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
if n_train <= 0 or n_eval <= 0:
    raise SystemExit("SMOKE_RECORDS and SMOKE_EVAL_RECORDS must be positive")
if n_eval > n_train:
    raise SystemExit("SMOKE_EVAL_RECORDS cannot exceed SMOKE_RECORDS")

candidates = []
with src.open(encoding="utf-8") as handle:
    for index, line in enumerate(handle):
        if not line.strip():
            continue
        record = json.loads(line)
        payload = {
            "id": record.get("sample_id") or record.get("id") or f"smoke-{index:04d}",
            "question": record["question"],
            "deepseek_thinking_trajectory": record.get("thinking")
            or record.get("deepseek_thinking_trajectory"),
            "deepseek_attempt": record.get("attempt") or record.get("deepseek_attempt"),
            "solution": record["solution"],
            "deepseek_grade": record.get("deepseek_grade"),
        }
        if payload["deepseek_thinking_trajectory"] is None or payload["deepseek_attempt"] is None:
            raise SystemExit(f"Record {payload['id']} is missing thinking/attempt fields")
        candidates.append(
            (
                int(record.get("original_length") or record.get("kept_length") or 10**9),
                payload["id"],
                payload,
            )
        )
if len(candidates) < n_train:
    raise SystemExit(f"Source jsonl only has {len(candidates)} records; need {n_train}")
candidates.sort(key=lambda item: (item[0], item[1]))
selected = [item[2] for item in candidates[:n_train]]
source_path = dest / "source.jsonl"
eval_path = dest / "eval.jsonl"
with source_path.open("w", encoding="utf-8") as out_source:
    for payload in selected:
        out_source.write(json.dumps(payload, ensure_ascii=False) + "\n")
eval_rows = [
    {"problem": payload["question"], "answer": payload["solution"], "id": payload["id"]}
    for payload in selected[:n_eval]
]
eval_path.write_text(
    "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in eval_rows),
    encoding="utf-8",
)
print(
    f"[smoke] wrote {source_path} ({len(selected)} records, "
    f"original_length {candidates[0][0]}-{candidates[n_train - 1][0]})"
)
print(f"[smoke] wrote {eval_path} ({len(eval_rows)} eval problems)")
PY
}

rm -rf "$PROJECT_ROOT/artifacts/smoke/prepared" \
  "$PROJECT_ROOT/artifacts/smoke/stage1" \
  "$PROJECT_ROOT/artifacts/smoke/supervision" \
  "$PROJECT_ROOT/artifacts/smoke/teacher_cache" \
  "$PROJECT_ROOT/artifacts/smoke/stage2" \
  "$PROJECT_ROOT/artifacts/smoke/evaluation"
write_smoke_inputs

echo "[smoke] running ./project_commands.sh full with Qwen2.5-0.5B-Instruct configs"
exec "$PROJECT_ROOT/project_commands.sh" full
