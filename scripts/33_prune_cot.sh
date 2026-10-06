#!/usr/bin/env bash
# Prune CoT steps with the frozen HF council, write prepared Phase-2 data, upload it.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

if [[ "${NPROC_PER_NODE:-1}" -gt 1 ]]; then
  echo "prune-cot runs one process. Unset NPROC_PER_NODE or set it to 1." >&2
  exit 1
fi
"$PYTHON_BIN" -m cot_mtkd.cli.prune_cot \
  --config "${STAGE2_CONFIG:-configs/stage2/qwen25_7b_output_space.yaml}"
