#!/usr/bin/env bash
# Grade the Phase-2 student with vLLM. One process; tensor_parallel_size is in the eval config.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

if ! "$PYTHON_BIN" -c "import vllm" >/dev/null 2>&1; then
  echo "vLLM is not installed in $PYTHON_BIN." >&2
  echo "Install it in the project environment before ./project_commands.sh evaluate." >&2
  exit 1
fi
if [[ "${NPROC_PER_NODE:-1}" -gt 1 ]]; then
  echo "evaluate runs one process and uses vllm.tensor_parallel_size from the eval config." >&2
fi
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
"$PYTHON_BIN" -m cot_mtkd.cli.evaluate \
  --config "${EVAL_CONFIG:-configs/eval/p_align.yaml}"

