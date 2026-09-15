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
  stage1        Train the five GAC-CoT LoRA experts.
  publish-stage1 Publish completed Stage-1 LoRA experts to Hugging Face.
  supervision  Build PAG, importance, teacher features, weights, and medoid.
  cache         Compile Top-512 plus tail-bucket teacher targets.
  stage2        Train the single medoid-initialized student adapter.
  evaluate      Generate and grade the P-ALIGN benchmark suite.
  test          Run local unit tests without downloading a model.
  all           Run prepare through evaluate (assumes setup is complete).
  full          Run setup, tests, and the complete pipeline.
  help          Show this message.

Environment overrides:
  NPROC_PER_NODE, PYTHON_BIN, DATA_CONFIG, STAGE1_CONFIG,
  SIGNALS_CONFIG, STAGE2_CONFIG, EVAL_CONFIG, STAGE1_RESUME,
  STAGE2_RESUME, HF_HUB_OFFLINE.
  HF_REPO_ID, HF_REPO_PRIVATE, HF_TOKEN (or Hugging Face CLI login).
EOF
}

command="${1:-help}"
case "$command" in
  setup) run_step 00_setup.sh ;;
  prepare) run_step 10_prepare_data.sh ;;
  stage1) run_step 20_train_stage1.sh ;;
  publish-stage1) bash "$PROJECT_ROOT/scripts/25_publish_stage1.sh" ;;
  supervision) run_step 30_build_supervision.sh ;;
  cache) run_step 40_build_teacher_cache.sh ;;
  stage2) run_step 50_train_stage2.sh ;;
  evaluate) run_step 60_evaluate.sh ;;
  test) run_step 90_test.sh ;;
  all)
    run_step 10_prepare_data.sh
    run_step 20_train_stage1.sh
    run_step 30_build_supervision.sh
    run_step 40_build_teacher_cache.sh
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
