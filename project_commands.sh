#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

run_step() {
  "$PROJECT_ROOT/scripts/$1"
}

stage1_eval_enabled() {
  [[ "${SKIP_STAGE1_EVAL:-0}" != "1" ]]
}

require_stage1_eval_backend() {
  stage1_eval_enabled || return 0
  local python_bin="$PROJECT_ROOT/.venv/bin/python"
  [[ -x "$python_bin" ]] || python_bin="${PYTHON_BIN:-python3}"
  if ! "$python_bin" -c "import vllm" >/dev/null 2>&1; then
    echo "Stage-1 expert evaluation needs vLLM, which is not installed in $python_bin." >&2
    echo "Run INSTALL_VLLM=1 ./project_commands.sh setup, or set SKIP_STAGE1_EVAL=1." >&2
    exit 1
  fi
}

usage() {
  cat <<'EOF'
Usage: ./project_commands.sh COMMAND

Commands:
  setup         Create .venv and install the local project.
  prepare       Download/read and preprocess s1K-1.1.
  stage1        Train the three GAC-CoT LoRA experts.
  stage1-smoke  Run Stage 1 on Qwen2.5-0.5B with the Stage-1 hyperparameters.
  stage1-eval   Evaluate the three Stage-1 LoRA experts one after another (vLLM).
  publish-stage1 Publish completed Stage-1 LoRA experts to Hugging Face.
  supervision  Build PAG, importance, council U/V/ρ, teacher weights, and medoid.
  cache         Optional Top-512 plus tail-bucket teacher targets (not used by Stage 2).
  stage2        Merge Stage-1 experts and train the weighted-NLL student.
  evaluate      Generate and grade the P-ALIGN benchmark suite.
  smoke         Tiny Qwen2.5-0.5B-Instruct run of setup/tests/full pipeline.
  test          Run local unit tests without downloading a model.
  all           Run prepare, stage1, stage1-eval, supervision, stage2, evaluate
                (assumes setup is complete; SKIP_STAGE1_EVAL=1 drops stage1-eval).
  full          Run setup, tests, and the complete pipeline.
  help          Show this message.

Environment overrides:
  NPROC_PER_NODE, PYTHON_BIN, DATA_CONFIG, STAGE1_CONFIG,
  SIGNALS_CONFIG, STAGE2_CONFIG, CACHE_CONFIG, EVAL_CONFIG, STAGE1_RESUME,
  STAGE2_RESUME, STAGE2_MERGE_METHOD, HF_HUB_OFFLINE.
  STAGE1_EVAL_CONFIG, STAGE1_EVAL_EXPERTS, STAGE1_EVAL_FORCE, SKIP_STAGE1_EVAL,
  INSTALL_VLLM.
  HF_REPO_ID, HF_REPO_PRIVATE, HF_TOKEN (or Hugging Face CLI login).
EOF
}

command="${1:-help}"
case "$command" in
  setup) run_step 00_setup.sh ;;
  prepare) run_step 10_prepare_data.sh ;;
  stage1) run_step 20_train_stage1.sh ;;
  stage1-smoke) run_step 21_smoke_stage1_0p5b.sh ;;
  stage1-eval) run_step 22_evaluate_stage1.sh ;;
  publish-stage1) bash "$PROJECT_ROOT/scripts/25_publish_stage1.sh" ;;
  supervision) run_step 30_build_supervision.sh ;;
  cache) run_step 40_build_teacher_cache.sh ;;
  stage2) run_step 50_train_stage2.sh ;;
  evaluate) run_step 60_evaluate.sh ;;
  smoke)
    bash "$PROJECT_ROOT/scripts/95_smoke.sh"
    ;;
  test) run_step 90_test.sh ;;
  all)
    require_stage1_eval_backend
    run_step 10_prepare_data.sh
    run_step 20_train_stage1.sh
    if stage1_eval_enabled; then
      run_step 22_evaluate_stage1.sh
    fi
    run_step 30_build_supervision.sh
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
