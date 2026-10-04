#!/usr/bin/env bash
# Evaluation environment (vLLM) in .venv-eval, separate from the training .venv
# so vLLM's pinned torch never replaces the training stack.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BOOTSTRAP="${PYTHON_BIN:-python3}"
VENV="$PROJECT_ROOT/.venv-eval"
if [[ ! -x "$VENV/bin/python" ]]; then
  "$PYTHON_BOOTSTRAP" -m venv "$VENV"
fi
"$VENV/bin/python" -m pip install --upgrade pip setuptools wheel
# vLLM first so it pins a torch/transformers pair it supports.
"$VENV/bin/python" -m pip install "vllm>=0.8"
"$VENV/bin/python" -m pip install -e "$PROJECT_ROOT[eval]"
echo "Evaluation environment ready at $VENV"
