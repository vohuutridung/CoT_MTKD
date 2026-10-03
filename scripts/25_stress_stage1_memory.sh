#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
"$PYTHON_BIN" -m cot_mtkd.cli.stress_stage1_memory \
  --config "${STAGE1_CONFIG:-configs/stage1/qwen25_7b_m3.yaml}" \
  --output "${STAGE1_STRESS_OUTPUT:-artifacts/stage1/stress_memory.json}" \
  --warmup "${STAGE1_STRESS_WARMUP:-1}" \
  --repetitions "${STAGE1_STRESS_REPETITIONS:-1}"
