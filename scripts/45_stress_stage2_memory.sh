#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
"$PYTHON_BIN" -m cot_mtkd.cli.stress_stage2_memory \
  --config "${STAGE2_CONFIG:-configs/stage2/qwen25_7b_task_geometry.yaml}" \
  --output "${STAGE2_STRESS_OUTPUT:-artifacts/stage2/stress_memory.json}" \
  --warmup "${STAGE2_STRESS_WARMUP:-1}" \
  --repetitions "${STAGE2_STRESS_REPETITIONS:-1}" \
  --min-headroom-gib "${STAGE2_STRESS_MIN_HEADROOM_GIB:-12}"
