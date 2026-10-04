#!/usr/bin/env bash
# Wait until Stage 2 has finished, then evaluate its student (P-ALIGN, vLLM).
# Safe to start while Stage 2 is still training; it only reads STAGE2_DIR.
#   STAGE2_DIR    Stage-2 output dir to watch; may live in another checkout
#                 (default artifacts/stage2/output_space)
#   EVAL_OUTPUT   default artifacts/evaluation/<basename of STAGE2_DIR>
#   EVAL_GPUS     GPUs for evaluation, e.g. "0,1,2": seeds are spread over them
#                 and run in parallel (default: first entry of CUDA_VISIBLE_DEVICES, else 0)
#   EVAL_SEEDS    space-separated seeds (default "42 43 44")
#   POLL_SECONDS  check interval while waiting (default 300)
#   PROC_TITLE    process name shown in nvitop (default hieunq10_eval)
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
export PROC_TITLE="${PROC_TITLE:-hieunq10_eval}"
STAGE2_DIR="$(realpath -m "${STAGE2_DIR:-artifacts/stage2/output_space}")"
EVAL_OUTPUT="$(realpath -m "${EVAL_OUTPUT:-artifacts/evaluation/$(basename "$STAGE2_DIR")}")"
IFS=',' read -r -a gpus <<< "${EVAL_GPUS:-${CUDA_VISIBLE_DEVICES:-0}}"
[[ -n "${EVAL_GPUS:-}" ]] || gpus=("${gpus[0]}")
read -r -a seeds <<< "${EVAL_SEEDS:-42 43 44}"

# Stage 2 writes manifest.json only after the final adapter has been saved.
until [[ -f "$STAGE2_DIR/manifest.json" \
         && -f "$STAGE2_DIR/final/adapters/student/adapter_config.json" ]]; do
  echo "$(date -u +%FT%TZ) waiting for Stage 2 to finish in $STAGE2_DIR"
  sleep "${POLL_SECONDS:-300}"
done
echo "$(date -u +%FT%TZ) Stage 2 finished; evaluating on GPUs ${gpus[*]}, seeds ${seeds[*]}"

evaluate() {  # evaluate <gpu> <yaml seed list>
  CUDA_VISIBLE_DEVICES="$1" "$PYTHON_BIN" -m cot_mtkd.cli.evaluate \
    --config "${EVAL_CONFIG:-configs/eval/p_align.yaml}" \
    --set "adapter.path=$STAGE2_DIR" --set "paths.output=$EVAL_OUTPUT" --set "seeds=$2"
}
join() { local IFS=,; echo "[$*]"; }

mkdir -p "$EVAL_OUTPUT"
pids=()
for index in "${!gpus[@]}"; do
  assigned=()
  for position in "${!seeds[@]}"; do
    (( position % ${#gpus[@]} == index )) && assigned+=("${seeds[$position]}")
  done
  (( ${#assigned[@]} )) || continue
  log="$EVAL_OUTPUT/eval_gpu${gpus[$index]}_seeds$(IFS=_; echo "${assigned[*]}").log"
  evaluate "${gpus[$index]}" "$(join "${assigned[@]}")" > "$log" 2>&1 &
  pids+=($!)
  echo "GPU ${gpus[$index]}: seeds ${assigned[*]} -> $log"
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
(( status == 0 )) || { echo "An evaluation process failed; see $EVAL_OUTPUT/eval_gpu*.log" >&2; exit 1; }

# All seeds now have result.json, so this only writes the combined summary.json.
evaluate "${gpus[0]}" "$(join "${seeds[@]}")" 2>&1 | grep -E "Avg Pass@1|Pass@1 .* ±"
echo "Summary: $EVAL_OUTPUT/summary.json"
