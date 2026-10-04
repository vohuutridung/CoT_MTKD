#!/usr/bin/env bash
# One command on a fresh clone: setup -> prepare -> fetch the duyentl04/abc experts
# -> Phase-2 cache -> train every run in STAGE2_RUNS -> evaluate each student.
# Each step skips finished work and training resumes from checkpoint.pt, so
# running the command again after a crash continues where it stopped.
#
#   CUDA_VISIBLE_DEVICES  GPUs to use (training uses NPROC_PER_NODE of them, default 1)
#   STAGE2_RUNS   runs to train, by config suffix (default "main"):
#                   main       configs/stage2/qwen25_7b_output_space.yaml (method)
#                   geometric  ..._geometric.yaml   arithmetic ..._arithmetic.yaml
#                   single     ..._single.yaml      sft        ..._sft.yaml
#                 e.g. STAGE2_RUNS="main geometric arithmetic single sft"
#   EVAL_GPUS     GPUs for evaluation, seeds spread over them (default: first visible GPU)
#   EVAL_SEEDS    default "42 43 44"
#   EVAL_GPU_MEMORY_UTILIZATION  vLLM memory fraction (default 0.90 from the config)
#   RUN_STRESS=1  run the H200 Phase-2 memory preflight before training
#   SKIP_EVAL=1   train only
#   PROC_PREFIX   process-title prefix shown in nvitop (default hieunq10)
#   STAGE2_ATTEMPTS  training attempts per run before giving up (default 3)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
log() { echo "$(date -u +%FT%TZ) [stage2-eval] $*"; }

# A venv counts as ready only if its packages import; an interrupted
# pip install leaves bin/python behind, so its existence proves nothing.
ready() { [[ -x "$1/bin/python" ]] && "$1/bin/python" -c "$2" >/dev/null 2>&1; }
if ! ready .venv "import torch, transformers, peft, cot_mtkd"; then
  log "setting up training .venv"
  ./scripts/00_setup.sh
  ready .venv "import torch, transformers, peft, cot_mtkd" || {
    log "training .venv is broken after setup"; exit 1; }
fi
if [[ "${SKIP_EVAL:-0}" != 1 ]] && ! ready .venv-eval "import vllm, cot_mtkd"; then
  log "setting up evaluation .venv-eval (vLLM)"
  ./scripts/05_setup_eval.sh
  ready .venv-eval "import vllm, cot_mtkd" || {
    log "evaluation .venv-eval is broken after setup"; exit 1; }
fi
source scripts/_common.sh

PREFIX="${PROC_PREFIX:-hieunq10}"
MAIN_CONFIG=configs/stage2/qwen25_7b_output_space.yaml
read -r -a runs <<< "${STAGE2_RUNS:-main}"
config_of() { [[ $1 == main ]] && echo "$MAIN_CONFIG" || echo "configs/stage2/qwen25_7b_output_space_$1.yaml"; }
output_of() {
  "$PYTHON_BIN" -c 'import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))["paths"]["output"])' "$1"
}
for run in "${runs[@]}"; do
  [[ -f "$(config_of "$run")" ]] || { log "unknown run '$run' ($(config_of "$run") missing)"; exit 2; }
done

if [[ -f artifacts/prepared/s1k_1_1_cot_only/manifest.json ]]; then
  log "prepare: already done"
else
  log "prepare"
  PROC_TITLE="${PREFIX}_prepare" ./scripts/10_prepare_data.sh
fi
log "fetch-teachers (verifies an existing import)"
STAGE2_CONFIG="$MAIN_CONFIG" ./scripts/32_fetch_stage2_teachers.sh
log "stage2-cache (a cache hit returns at once)"
STAGE2_CONFIG="$MAIN_CONFIG" PROC_TITLE="${PREFIX}_s2_cache" ./scripts/35_build_stage2_cache.sh
if [[ "${RUN_STRESS:-0}" == 1 ]]; then
  log "stage2-stress"
  STAGE2_CONFIG="$MAIN_CONFIG" PROC_TITLE="${PREFIX}_s2_stress" ./scripts/45_stress_stage2_memory.sh
fi

for run in "${runs[@]}"; do
  config="$(config_of "$run")"
  output="$(output_of "$config")"
  attempt=0
  until [[ -f "$output/manifest.json" ]]; do
    attempt=$((attempt + 1))
    if (( attempt > ${STAGE2_ATTEMPTS:-3} )); then
      log "$run: training failed ${STAGE2_ATTEMPTS:-3} times; see the log above"
      exit 1
    fi
    resume=""
    [[ -f "$output/checkpoint.pt" ]] && resume="$output/checkpoint.pt"
    log "$run: train attempt $attempt ($config, resume=${resume:-none})"
    STAGE2_CONFIG="$config" STAGE2_RESUME="$resume" PROC_TITLE="${PREFIX}_s2_$run" \
      ./scripts/50_train_stage2.sh || { log "$run: training exited with an error"; sleep 30; }
  done
  log "$run: trained -> $output"
done

[[ "${SKIP_EVAL:-0}" == 1 ]] && { log "SKIP_EVAL=1: done"; exit 0; }
for run in "${runs[@]}"; do
  output="$(output_of "$(config_of "$run")")"
  log "$run: evaluate $output"
  STAGE2_DIR="$output" EVAL_OUTPUT="artifacts/evaluation/$(basename "$output")" \
    PROC_TITLE="${PREFIX}_eval_$run" ./scripts/70_evaluate_after_stage2.sh
done

log "results (Pass@1 mean ± std over seeds, 4-benchmark average)"
for run in "${runs[@]}"; do
  summary="artifacts/evaluation/$(basename "$(output_of "$(config_of "$run")")")/summary.json"
  "$PYTHON_BIN" - "$run" "$summary" <<'PY'
import json, sys
run, path = sys.argv[1:]
s = json.load(open(path))
bench = "  ".join(
    f"{name} {100 * v['pass@1']['mean']:.2f}" for name, v in sorted(s["benchmarks"].items())
)
print(f"{run:<11} avg {100 * s['avg']['mean']:.2f} ± {100 * s['avg']['std']:.2f}   {bench}")
PY
done
