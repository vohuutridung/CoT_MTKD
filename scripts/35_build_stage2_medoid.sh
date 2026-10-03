#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
STAGE2_CONFIG="${STAGE2_CONFIG:-configs/stage2/qwen25_7b_task_geometry.yaml}"
run_distributed cot_mtkd.cli.build_stage2_medoid --config "$STAGE2_CONFIG"
