#!/usr/bin/env bash
# Download sonspeed/Trainhihi and tokenize it for Phase 2.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
"$PYTHON_BIN" -m cot_mtkd.cli.prepare_data \
  --config "${PRUNED_DATA_CONFIG:-configs/data/trainhihi.yaml}"
