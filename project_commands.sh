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
  setup-eval    Create .venv with vLLM for an evaluation-only machine.
  prepare       Download/read and preprocess s1K-1.1.
  stage1        Train the three GAC-CoT LoRA experts.
  stage1-stress Run six H200 SFT/ramp/full VRAM stress cases before Phase 1.
  fetch-teachers Download/import the trained duyentl04/abc experts for Phase 2.
  supervision  Build legacy PAG/features (not used by the new Phase 2).
  cache         Compile legacy Top-512/tail targets (not used by the new Phase 2).
  stage2-medoid Select the functional medoid without legacy PAG/features.
  stage2-stress Run real/synthetic full-update H200 Phase-2 VRAM checks.
  stage2        Train the single medoid-initialized student adapter.
  pack-student  Archive the Stage-2 student (optionally upload: HF_REPO=...).
  evaluate      Generate and grade the P-ALIGN suite (vLLM, seeds 42 43 44).
  test          Run local unit tests without downloading a model.
  all           Prepare, load trained teachers, run Phase 2 and evaluate.
  full          Run setup, tests, and the complete pipeline.
  help          Show this message.

Environment overrides:
  NPROC_PER_NODE, PYTHON_BIN, DATA_CONFIG, STAGE1_CONFIG, STAGE1_FORWARD_MODE,
  STAGE1_STRESS_OUTPUT, STAGE1_STRESS_WARMUP, STAGE1_STRESS_REPETITIONS,
  SIGNALS_CONFIG, STAGE2_CONFIG, EVAL_CONFIG, EVAL_ADAPTER, EVAL_OUTPUT,
  EVAL_SEEDS, STAGE2_DIR, PACK_OUT, HF_REPO, STAGE1_RESUME,
  STAGE2_RESUME, HF_HUB_OFFLINE, STAGE2_STRESS_OUTPUT, STAGE2_STRESS_WARMUP,
  STAGE2_STRESS_REPETITIONS, STAGE2_STRESS_MIN_HEADROOM_GIB.
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
  stage2-medoid) run_step 35_build_stage2_medoid.sh ;;
  stage2-stress) run_step 45_stress_stage2_memory.sh ;;
  stage2) run_step 50_train_stage2.sh ;;
  pack-student) run_step 65_pack_student.sh ;;
  evaluate) shift; "$PROJECT_ROOT/scripts/60_evaluate.sh" "$@" ;;
  test) run_step 90_test.sh ;;
  all)
    run_step 10_prepare_data.sh
    run_step 32_fetch_stage2_teachers.sh
    run_step 35_build_stage2_medoid.sh
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
