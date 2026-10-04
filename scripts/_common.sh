#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_DISABLED=true
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
# Too many CPU threads slow the CPU-side cache/loss math badly
# (Phase-2 cache: 6.8 s/sample at 24 threads vs 48-74 s/sample at 96).
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-24}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$OMP_NUM_THREADS}"

if [[ -x "$PROJECT_ROOT/.venv/bin/python" ]]; then
  PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

# Evaluation uses its own venv (vLLM pins its own torch); fall back to the training one.
if [[ -x "$PROJECT_ROOT/.venv-eval/bin/python" ]]; then
  EVAL_PYTHON_BIN="$PROJECT_ROOT/.venv-eval/bin/python"
else
  EVAL_PYTHON_BIN="$PYTHON_BIN"
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
