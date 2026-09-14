#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
run_distributed cot_mtkd.cli.evaluate \
  --config "${EVAL_CONFIG:-configs/eval/p_align.yaml}"

