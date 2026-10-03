#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
"$PYTHON_BIN" -m cot_mtkd.cli.fetch_stage2_teachers \
  --config "${STAGE2_CONFIG:-configs/stage2/qwen25_7b_task_geometry.yaml}"
