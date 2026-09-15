#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
"$PYTHON_BIN" -m cot_mtkd.cli.publish_stage1 \
  --config "${STAGE1_CONFIG:-configs/stage1/qwen25_7b_m5.yaml}" "$@"
