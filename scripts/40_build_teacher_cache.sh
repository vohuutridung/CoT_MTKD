#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
run_distributed cot_mtkd.cli.build_teacher_cache \
  --config "${STAGE2_CONFIG:-configs/stage2/qwen25_7b_top512_tail.yaml}"

