#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
run_distributed cot_mtkd.cli.build_teacher_cache \
  --config "${CACHE_CONFIG:-configs/cache/qwen25_7b_top512_tail.yaml}"

