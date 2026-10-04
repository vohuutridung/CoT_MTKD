# CoT-MTKD

Train three LoRA experts, distill them into one student, and evaluate the student.

## Setup

Use Python 3.10+ and CUDA-enabled PyTorch. The commands below target one H200.
FlashAttention 2 is recommended; the loader falls back to SDPA when unavailable.
`setup` does not install FlashAttention 2.

```bash
git clone --branch output-space https://github.com/vohuutridung/CoT_MTKD.git
cd CoT_MTKD
./project_commands.sh setup

export CUDA_VISIBLE_DEVICES=0
export NPROC_PER_NODE=1
```

- `git clone` downloads the `output-space` branch; `cd` enters the repository.
- `setup` creates `.venv` and installs the project and test dependencies. All
  project commands use `.venv` automatically; activation is unnecessary.
- `CUDA_VISIBLE_DEVICES=0` selects GPU 0 for preprocessing, training and evaluation.
- `NPROC_PER_NODE=1` runs one process on that GPU.

## Run

### Phase 1 → Phase 2 → evaluation

After setup, run the complete sequence below. Continue past each stress check
only if it succeeds. Phase 2 defaults are **1 epoch, global batch 8, microbatch 1**.

```bash
export STAGE2_CONFIG=configs/stage2/qwen25_7b_output_space_local.yaml

./project_commands.sh prepare
./project_commands.sh stage1-stress
./project_commands.sh stage1

./project_commands.sh stage2-cache
STAGE2_STRESS_OUTPUT=artifacts/stage2/output_space_local_stress_memory.json \
  ./project_commands.sh stage2-stress
./project_commands.sh stage2

.venv/bin/python -m cot_mtkd.cli.evaluate \
  --config configs/eval/p_align.yaml \
  --set paths.stage2=artifacts/stage2/output_space_local \
  --set paths.output=artifacts/evaluation/p_align_local
```

Meaning of each command:

1. `export STAGE2_CONFIG=...output_space_local.yaml` makes Phase 2 use the experts
   trained locally by Phase 1, with separate cache and student output directories.
2. `prepare` downloads and tokenizes the training corpus, saving it in
   `artifacts/prepared/s1k_1_1_cot_only/` for both phases.
3. `stage1-stress` checks Phase-1 VRAM and update completion on the longest real
   sample and synthetic 32k samples; report: `artifacts/stage1/stress_memory.json`.
4. `stage1` trains the three experts and saves their adapters, checkpoint and
   manifest in `artifacts/stage1/main/`.
5. `stage2-cache` precomputes teacher targets and selects the best expert for
   student initialization; cache: `artifacts/teacher_cache/output_space_local/`.
6. `stage2-stress` checks the cached student training path on real and synthetic
   samples. The output override saves `artifacts/stage2/output_space_local_stress_memory.json`.
7. `stage2` trains the student for one epoch and saves its adapter, checkpoint
   and logs in `artifacts/stage2/output_space_local/`.
8. The `evaluate` command benchmarks that local student on AIME 2025, AIME 2024,
   AMC and MATH-500, saving results in `artifacts/evaluation/p_align_local/`.
   Evaluation is optional after training finishes.

### Phase 2 with the existing Hugging Face experts

After the same setup, use this sequence if you want the existing `duyentl04/abc`
experts. It skips Phase-1 training. Continue past the stress check only if it succeeds.

```bash
export STAGE2_CONFIG=configs/stage2/qwen25_7b_output_space.yaml

./project_commands.sh prepare
./project_commands.sh fetch-teachers
./project_commands.sh stage2-cache
./project_commands.sh stage2-stress
./project_commands.sh stage2
./project_commands.sh evaluate
```

Meaning of each command:

1. `export STAGE2_CONFIG=...output_space.yaml` selects the pinned Hugging Face council.
2. `prepare` builds the same tokenized corpus in `artifacts/prepared/s1k_1_1_cot_only/`.
3. `fetch-teachers` downloads, verifies and imports the three experts into
   `artifacts/stage1/duyentl04_abc/`.
4. `stage2-cache` precomputes teacher targets and selects the initialization expert;
   cache: `artifacts/teacher_cache/output_space/`.
5. `stage2-stress` checks student training VRAM and update completion;
   report: `artifacts/stage2/output_space_stress_memory.json`.
6. `stage2` trains for one epoch, saving the student and logs in
   `artifacts/stage2/output_space/`.
7. `evaluate` optionally benchmarks that student, saving results in
   `artifacts/evaluation/p_align/`.

For either route, the student output directory contains `final/adapters/student/`,
`checkpoint.pt`, `metrics.jsonl`, `performance.jsonl` and `reasoning_steps.jsonl`.
