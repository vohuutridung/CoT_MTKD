#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
"$PYTHON_BIN" -m cot_mtkd.cli.prepare_data \
  --config "${DATA_CONFIG:-configs/data/s1k_1_1.yaml}"

