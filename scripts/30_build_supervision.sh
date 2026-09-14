#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
run_distributed cot_mtkd.cli.build_supervision \
  --config "${SIGNALS_CONFIG:-configs/signals/main.yaml}"

