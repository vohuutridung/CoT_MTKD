#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

# Tiny Qwen2.5-0.5B-Instruct run of the README pipeline.
# Reuses text from the already-prepared s1K-1.1 jsonl (no HF dataset re-download),
# then re-tokenizes with the 0.5B tokenizer because Stage 1/2 require a matching
# model source.

SMOKE_PREPARED="${SMOKE_PREPARED:-artifacts/prepared/s1k_1_1/data.jsonl}"
SMOKE_RECORDS="${SMOKE_RECORDS:-8}"
SMOKE_EVAL_RECORDS="${SMOKE_EVAL_RECORDS:-2}"
SMOKE_DIR="${SMOKE_DIR:-artifacts/smoke}"

export DATA_CONFIG="${DATA_CONFIG:-configs/smoke/data.yaml}"
export STAGE1_CONFIG="${STAGE1_CONFIG:-configs/smoke/stage1.yaml}"
export SIGNALS_CONFIG="${SIGNALS_CONFIG:-configs/smoke/signals.yaml}"
export STAGE2_CONFIG="${STAGE2_CONFIG:-configs/smoke/stage2.yaml}"
export EVAL_CONFIG="${EVAL_CONFIG:-configs/smoke/eval.yaml}"

usage() {
  cat <<'EOF'
Usage: ./scripts/95_smoke.sh [COMMAND]

Commands (same stages as README / project_commands.sh):
  prepare       Slice prepared s1K text and tokenize with Qwen2.5-0.5B-Instruct.
  stage1        Train five tiny GAC-CoT LoRA experts.
  supervision   Build PAG / importance / teacher features / medoid.
  cache         Compile sparse teacher targets.
  stage2        Train the student adapter.
  evaluate      Generate on a 2-problem local slice.
  all           Run prepare through evaluate (default).
  help          Show this message.

Environment:
  SMOKE_PREPARED, SMOKE_RECORDS, SMOKE_EVAL_RECORDS, NPROC_PER_NODE,
  DATA_CONFIG, STAGE1_CONFIG, SIGNALS_CONFIG, STAGE2_CONFIG, EVAL_CONFIG,
  STAGE1_RESUME, STAGE2_RESUME, HF_HUB_OFFLINE.
EOF
}

write_smoke_inputs() {
  if [[ ! -f "$SMOKE_PREPARED" ]]; then
    echo "Prepared dataset not found: $SMOKE_PREPARED" >&2
    echo "Run ./project_commands.sh prepare first." >&2
    exit 1
  fi
  mkdir -p "$SMOKE_DIR"
  echo "[smoke] slicing $SMOKE_RECORDS records from $SMOKE_PREPARED"
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
            "deepseek_thinking_trajectory": record["thinking"],
            "deepseek_attempt": record["attempt"],
            "solution": record["solution"],
            "deepseek_grade": record.get("deepseek_grade"),
        }
        candidates.append(
            (
                int(record.get("original_length") or record.get("kept_length") or 10**9),
                payload["id"],
                payload,
            )
        )
if len(candidates) < n_train:
    raise SystemExit(f"Prepared jsonl only has {len(candidates)} records; need {n_train}")
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
    f"[smoke] wrote {source_path} ({len(selected)} shortest records, "
    f"original_length {candidates[0][0]}-{candidates[n_train - 1][0]})"
)
print(f"[smoke] wrote {eval_path} ({len(eval_rows)} eval problems)")
PY
}

run_named() {
  local name="$1"
  echo
  echo "========== smoke: $name =========="
  case "$name" in
    prepare)
      rm -rf "$PROJECT_ROOT/artifacts/smoke/prepared"
      write_smoke_inputs
      "$PYTHON_BIN" -m cot_mtkd.cli.prepare_data \
        --config "$DATA_CONFIG" \
        --set "dataset.expected_records=$SMOKE_RECORDS"
      ;;
    stage1) "$PROJECT_ROOT/scripts/20_train_stage1.sh" ;;
    supervision) "$PROJECT_ROOT/scripts/30_build_supervision.sh" ;;
    cache) "$PROJECT_ROOT/scripts/40_build_teacher_cache.sh" ;;
    stage2) "$PROJECT_ROOT/scripts/50_train_stage2.sh" ;;
    evaluate) "$PROJECT_ROOT/scripts/60_evaluate.sh" ;;
    *)
      echo "Unknown smoke command: $name" >&2
      usage >&2
      exit 2
      ;;
  esac
}

command="${1:-all}"
case "$command" in
  help|-h|--help) usage ;;
  all)
    rm -rf "$PROJECT_ROOT/artifacts/smoke/prepared" \
      "$PROJECT_ROOT/artifacts/smoke/stage1" \
      "$PROJECT_ROOT/artifacts/smoke/supervision" \
      "$PROJECT_ROOT/artifacts/smoke/teacher_cache" \
      "$PROJECT_ROOT/artifacts/smoke/stage2" \
      "$PROJECT_ROOT/artifacts/smoke/evaluation"
    run_named prepare
    run_named stage1
    run_named supervision
    run_named cache
    run_named stage2
    run_named evaluate
    echo
    echo "[smoke] done. artifacts under $SMOKE_DIR"
    ;;
  prepare|stage1|supervision|cache|stage2|evaluate)
    run_named "$command"
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
