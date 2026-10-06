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
  setup         Create .venv and install the local project.
  prepare       Download/read and preprocess s1K-1.1.
  stage1        Train three CoT LoRA experts with GAC over SFT/DPP/RBF.
  stage1-stress Run four SFT-only/full-interaction H200 VRAM stress cases before Phase 1.
  fetch-teachers Download/import the trained duyentl04/abc experts for Phase 2.
  prune-cot    Keep the shortest valid CoT prefix, write Phase-2 data, upload it.
  fetch-pruned Download sonspeed/Trainhihi and tokenize it for Phase 2.
  supervision  Build legacy PAG/features (not used by the new Phase 2).
  cache         Compile legacy Top-512/tail targets (not used by the new Phase 2).
  stage2-cache  Precompute support/tail targets and select the best SFT expert.
  stage2-stress Run real/synthetic full-update H200 Phase-2 VRAM checks.
  stage2        Train the cached KD + SFT student adapter.
  evaluate      Generate and grade the P-ALIGN benchmark suite.
  test          Run local unit tests without downloading a model.
  all           Prepare, load teachers, prune CoT, run Phase 2 and evaluate.
  full          Run setup, tests, and the complete pipeline.
  help          Show this message.

Environment overrides:
  NPROC_PER_NODE, PYTHON_BIN, DATA_CONFIG, STAGE1_CONFIG, STAGE1_FORWARD_MODE,
  STAGE1_STRESS_OUTPUT, STAGE1_STRESS_WARMUP, STAGE1_STRESS_REPETITIONS,
  SIGNALS_CONFIG, STAGE2_CONFIG, EVAL_CONFIG, STAGE1_RESUME,
  STAGE2_RESUME, HF_HUB_OFFLINE, STAGE2_STRESS_OUTPUT, STAGE2_STRESS_WARMUP,
  STAGE2_STRESS_REPETITIONS, STAGE2_STRESS_MIN_HEADROOM_GIB.
EOF
}

command="${1:-help}"
case "$command" in
  setup) run_step 00_setup.sh ;;
  prepare) run_step 10_prepare_data.sh ;;
  stage1) run_step 20_train_stage1.sh ;;
  stage1-stress) run_step 25_stress_stage1_memory.sh ;;
  fetch-teachers) run_step 32_fetch_stage2_teachers.sh ;;
  prune-cot) run_step 33_prune_cot.sh ;;
  fetch-pruned) run_step 34_fetch_pruned_data.sh ;;
  supervision) run_step 30_build_supervision.sh ;;
  cache) run_step 40_build_teacher_cache.sh ;;
  stage2-cache) run_step 35_build_stage2_cache.sh ;;
  stage2-stress) run_step 45_stress_stage2_memory.sh ;;
  stage2) run_step 50_train_stage2.sh ;;
  evaluate) run_step 60_evaluate.sh ;;
  test) run_step 90_test.sh ;;
  all)
    run_step 10_prepare_data.sh
    run_step 32_fetch_stage2_teachers.sh
    run_step 33_prune_cot.sh
    run_step 35_build_stage2_cache.sh
    run_step 50_train_stage2.sh
    run_step 60_evaluate.sh
    ;;
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
