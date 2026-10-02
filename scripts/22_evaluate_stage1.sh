#!/usr/bin/env bash
# Evaluate the Stage-1 LoRA experts one after another with vLLM.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

if ! "$PYTHON_BIN" -c "import vllm" >/dev/null 2>&1; then
  echo "vLLM is not installed in $PYTHON_BIN." >&2
  echo "Install it with: INSTALL_VLLM=1 ./project_commands.sh setup" >&2
  echo "or skip this step with SKIP_STAGE1_EVAL=1." >&2
  exit 1
fi

arguments=(--config "${STAGE1_EVAL_CONFIG:-configs/eval/stage1_experts.yaml}")
if [[ -n "${STAGE1_EVAL_EXPERTS:-}" ]]; then
  arguments+=(--experts "$STAGE1_EVAL_EXPERTS")
fi
if [[ "${STAGE1_EVAL_FORCE:-0}" == "1" ]]; then
  arguments+=(--force)
fi
"$PYTHON_BIN" -m cot_mtkd.cli.evaluate_stage1 "${arguments[@]}" "$@"
