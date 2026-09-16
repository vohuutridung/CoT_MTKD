#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export WANDB_DISABLED=true
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false

if [[ -x "$PROJECT_ROOT/.venv/bin/python" ]]; then
  PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

run_distributed() {
  local module="$1"
  shift
  if [[ "$NPROC_PER_NODE" -gt 1 ]]; then
    "$PYTHON_BIN" -m torch.distributed.run --standalone \
      --nproc_per_node "$NPROC_PER_NODE" -m "$module" "$@"
  else
    "$PYTHON_BIN" -m "$module" "$@"
  fi
}
