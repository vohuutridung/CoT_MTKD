#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
arguments=(--config "${STAGE2_CONFIG:-configs/stage2/qwen25_7b.yaml}")
if [[ -n "${STAGE2_RESUME:-}" ]]; then
  arguments+=(--set "stage2.resume_from=$STAGE2_RESUME")
fi
if [[ -n "${STAGE2_MERGE_METHOD:-}" ]]; then
  arguments+=(--merge-method "$STAGE2_MERGE_METHOD")
fi
run_distributed cot_mtkd.cli.train_stage2 "${arguments[@]}"
