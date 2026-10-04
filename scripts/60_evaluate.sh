#!/usr/bin/env bash
# Evaluate the Stage-2 student (vLLM, seeds 42 43 44). Optional environment:
#   EVAL_ADAPTER  adapter source (see configs/eval/p_align.yaml: adapter.path)
#   EVAL_OUTPUT   output directory
#   EVAL_SEEDS    e.g. "[42]"
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
overrides=()
[[ -n "${EVAL_ADAPTER:-}" ]] && overrides+=(--set "adapter.path=$EVAL_ADAPTER")
[[ -n "${EVAL_OUTPUT:-}" ]] && overrides+=(--set "paths.output=$EVAL_OUTPUT")
[[ -n "${EVAL_SEEDS:-}" ]] && overrides+=(--set "seeds=$EVAL_SEEDS")
"$PYTHON_BIN" -m cot_mtkd.cli.evaluate \
  --config "${EVAL_CONFIG:-configs/eval/p_align.yaml}" "${overrides[@]}" "$@"
