#!/usr/bin/env bash
# Evaluation-only environment (vLLM). Use on a machine that only evaluates.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BOOTSTRAP="${PYTHON_BIN:-python3}"
if [[ ! -x "$PROJECT_ROOT/.venv/bin/python" ]]; then
  "$PYTHON_BOOTSTRAP" -m venv "$PROJECT_ROOT/.venv"
fi
"$PROJECT_ROOT/.venv/bin/python" -m pip install --upgrade pip setuptools wheel
# vLLM first so it pins a torch/transformers pair it supports.
"$PROJECT_ROOT/.venv/bin/python" -m pip install "vllm>=0.8"
"$PROJECT_ROOT/.venv/bin/python" -m pip install -e "$PROJECT_ROOT[eval]"
echo "Evaluation environment ready at $PROJECT_ROOT/.venv"
