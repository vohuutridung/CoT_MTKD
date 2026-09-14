#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BOOTSTRAP="${PYTHON_BIN:-python3}"

if [[ ! -x "$PROJECT_ROOT/.venv/bin/python" ]]; then
  "$PYTHON_BOOTSTRAP" -m venv "$PROJECT_ROOT/.venv"
fi

"$PROJECT_ROOT/.venv/bin/python" -m pip install --upgrade pip setuptools wheel
"$PROJECT_ROOT/.venv/bin/python" -m pip install -e "$PROJECT_ROOT[dev]"

echo "Environment ready at $PROJECT_ROOT/.venv"
echo "FlashAttention-2 is optional and must match the node CUDA/PyTorch build."

