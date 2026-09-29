#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
arguments=(--config "${STAGE1_CONFIG:-configs/stage1/qwen25_7b_m3.yaml}")
if [[ -n "${STAGE1_RESUME:-}" ]]; then
  arguments+=(--set "stage1.resume_from=$STAGE1_RESUME")
fi
if [[ -n "${STAGE1_FORWARD_MODE:-}" ]]; then
  arguments+=(--set "stage1.forward_mode=$STAGE1_FORWARD_MODE")
fi
run_distributed cot_mtkd.cli.train_stage1 "${arguments[@]}"
