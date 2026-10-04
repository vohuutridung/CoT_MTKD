#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

run_step() {
  "$PROJECT_ROOT/scripts/$1"
}

usage() {
  cat <<'EOF'
Usage: ./project_commands.sh COMMAND

Commands:
  setup         Create .venv and install the local project (training).
  setup-eval    Create .venv-eval with vLLM (evaluation; kept apart from training).
  prepare       Download/read and preprocess s1K-1.1.
  stage1        Train three CoT LoRA experts with SFT/DPP/RBF.
  stage1-stress Run two full-interaction H200 VRAM stress cases before Phase 1.
  fetch-teachers Download/import the trained duyentl04/abc experts for Phase 2.
  supervision  Build legacy PAG/features (not used by the new Phase 2).
  cache         Compile legacy Top-512/tail targets (not used by the new Phase 2).
  stage2-cache  Precompute support/tail targets and select the best SFT expert.
  stage2-stress Run real/synthetic full-update H200 Phase-2 VRAM checks.
  stage2        Train the cached KD + SFT student adapter.
  pack-student  Archive the Stage-2 student (optionally upload: HF_REPO=...).
  evaluate      Generate and grade the P-ALIGN suite (vLLM, seeds 42 43 44).
  evaluate-after-stage2  Wait for Stage 2 to finish, then evaluate it.
  stage2-eval   One command on a fresh clone: setup, prepare, fetch experts,
                Phase-2 cache, train every STAGE2_RUNS config, evaluate each.
  test          Run local unit tests without downloading a model.
  all           Same as stage2-eval.
  full          Run setup, tests, and the complete pipeline.
  help          Show this message.

Environment overrides:
  NPROC_PER_NODE, PYTHON_BIN, DATA_CONFIG, STAGE1_CONFIG, STAGE1_FORWARD_MODE,
  STAGE1_STRESS_OUTPUT, STAGE1_STRESS_WARMUP, STAGE1_STRESS_REPETITIONS,
  SIGNALS_CONFIG, STAGE2_CONFIG, EVAL_CONFIG, STAGE1_RESUME,
  STAGE2_RESUME, HF_HUB_OFFLINE, STAGE2_STRESS_OUTPUT, STAGE2_STRESS_WARMUP,
  STAGE2_STRESS_REPETITIONS, STAGE2_STRESS_MIN_HEADROOM_GIB,
  STAGE2_RUNS, EVAL_GPUS, EVAL_SEEDS, RUN_STRESS, SKIP_EVAL, PROC_PREFIX.
EOF
}

command="${1:-help}"
case "$command" in
  setup) run_step 00_setup.sh ;;
  setup-eval) run_step 05_setup_eval.sh ;;
  prepare) run_step 10_prepare_data.sh ;;
  stage1) run_step 20_train_stage1.sh ;;
  stage1-stress) run_step 25_stress_stage1_memory.sh ;;
  fetch-teachers) run_step 32_fetch_stage2_teachers.sh ;;
  supervision) run_step 30_build_supervision.sh ;;
  cache) run_step 40_build_teacher_cache.sh ;;
  stage2-cache) run_step 35_build_stage2_cache.sh ;;
  stage2-stress) run_step 45_stress_stage2_memory.sh ;;
  stage2) run_step 50_train_stage2.sh ;;
  pack-student) run_step 65_pack_student.sh ;;
  evaluate) shift; "$PROJECT_ROOT/scripts/60_evaluate.sh" "$@" ;;
  evaluate-after-stage2) run_step 70_evaluate_after_stage2.sh ;;
  stage2-eval) run_step 80_stage2_and_evaluate.sh ;;
  test) run_step 90_test.sh ;;
  all) run_step 80_stage2_and_evaluate.sh ;;
  full)
    run_step 00_setup.sh
    run_step 90_test.sh
    "$0" all
    ;;
  help|-h|--help) usage ;;
  *)
    echo "Unknown command: $command" >&2
    usage >&2
    exit 2
    ;;
esac
