#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
arguments=(--config "${STAGE1_CONFIG:-configs/stage1/qwen25_7b_m5.yaml}")
if [[ -n "${STAGE1_RESUME:-}" ]]; then
  arguments+=(--set "stage1.resume_from=$STAGE1_RESUME")
fi
run_distributed cot_mtkd.cli.train_stage1 "${arguments[@]}"
if [[ -n "${HF_REPO_ID:-}" ]]; then
  "$PYTHON_BIN" -m cot_mtkd.cli.publish_stage1 \
    --config "${STAGE1_CONFIG:-configs/stage1/qwen25_7b_m5.yaml}" \
    --repo-id "$HF_REPO_ID"
fi
