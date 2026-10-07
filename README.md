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
   sample and synthetic 32k samples in SFT-only/full-interaction modes (four cases);
   report: `artifacts/stage1/stress_memory.json`.
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

### Phase 1 on `phase1-gac`

This branch implements local GAC with a separate own-task contribution and
unnormalized sharing from other experts. The method identifier is
`sft_dpp_rbf_local_gac`. CoT-only preprocessing retains 996 samples and never
reads `deepseek_attempt`. Kneedle, union support and the normalized step-level
DPP loss retain their `output-space` semantics. Phase 2 is unchanged.

The frozen Qwen2.5-7B-Instruct backbone has three LoRA adapters (rank/alpha 16,
dropout 0.05, existing target modules). SFT uses all labeled assistant tokens;
DPP uses reasoning tokens only. Its loss per step remains
`[M log(1 + eps_s) - logdet(L_s + eps_s I)] / M`, averaged over steps per
sample, then samples with reasoning.

For each optimizer window after SFT warmup:

1. Form `g_task_i = g_SFT_i + 0.1 * g_DPP_i`. SFT is normalized by global
   labeled-token count; DPP by global reasoning-sample count. Their contributions
   are combined before one transformer VJP per expert.
2. Compute `Delta W_i = (alpha/r) B_i A_i` and reuse the existing normalized
   squared Frobenius distance `D_ij`, averaged over LoRA modules. Low-rank
   computation avoids materializing full `B A`; `D_ij` is not squared again.
3. Keep the base bandwidth heuristic `median(D_ij for i<j) / log(M+1)`, floor
   `1e-12` and EMA decay 0.9. Set `h_G = 0.5 * h_base` and `h_R = h_base`.
   The kernels are `K_G = exp(-D/h_G)` and `K_R = exp(-D/h_R)`.
4. Set `a_ji = beta/(M-1) * K_G[j,i]` for `j != i`, with `beta=0.5`.
   Form `g_mix_i = (1 - sum_{j!=i} a_ji) * g_task_i + sum_{j!=i} a_ji * g_task_j`.
   Cross-expert coefficients are used directly. The diagonal is excluded and
   the own-task coefficient is at least 0.5 with the default beta.
5. Compute the outward direction `r_i = -grad_i mean_{j<k} K_R[j,k]`, holding
   bandwidth fixed. Keep the existing cap
   `c_i = min(1, norm(g_mix_i)/(norm(r_i)+1e-12))`.
   Form `g_full_i = g_mix_i - 0.5 * c_i * r_i`, clip each expert to norm 1,
   and apply its AdamW optimizer and cosine learning-rate scheduler.

Update mode uses `progress = global_step / max(total_updates - 1, 1)`:
`progress < 0.10` gives `sft_only`, which skips support probing, DPP, sharing
and RBF; `progress >= 0.10` switches directly to `full_interaction`.
The restored optimizer-step cursor selects the same mode on checkpoint resume.
`one_pass` remains the default and `two_pass` replays the same dropout seeds.
Epochs/global batch/microbatch/learning rate remain 3/16/1/5e-5.

GAC remains a pseudo-gradient update. Logs report SFT/DPP losses, pairwise
D/K_G/K_R, both scaled bandwidths and the base bandwidth, own/cross coefficients,
task/mixed norms, raw/capped/weighted repulsion norms and cap factors. The method
identifier and config fingerprint reject checkpoints from older formulations;
changing the forward mode alone remains resume-compatible.

### Change the backbone quickly

Example: Phase 1 with [DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B),
after setup on one GPU. To switch again, change only `MODEL` and `RUN`.

```bash
MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
RUN="deepseek_r1_qwen_1_5b"

.venv/bin/python -m cot_mtkd.cli.prepare_data \
  --config configs/data/s1k_1_1.yaml \
  --set "model.name_or_path=$MODEL" \
  --set "output_dir=artifacts/prepared/$RUN"

.venv/bin/python -m cot_mtkd.cli.train_stage1 \
  --config configs/stage1/qwen25_7b_m3.yaml \
  --set "model.name_or_path=$MODEL" \
  --set "paths.prepared=artifacts/prepared/$RUN" \
  --set "paths.output=artifacts/stage1/$RUN"
```

`prepare_data` retokenizes the same 996 CoT samples; `train_stage1` trains three
experts. These commands keep the repo's current training format; DeepSeek's
[native chat format](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B/blob/main/tokenizer_config.json)
differs. Phase 2/evaluation must use the same model overrides and new experts/cache.

### Phase 2 with the existing Hugging Face experts

After the same setup, use this sequence if you want the existing `duyentl04/abc`
experts. It skips Phase-1 training. Continue past the stress check only if it succeeds.

```bash
export STAGE2_CONFIG=configs/stage2/qwen25_7b_output_space.yaml

./project_commands.sh prepare
./project_commands.sh fetch-teachers
./project_commands.sh prune-cot
./project_commands.sh stage2-cache
./project_commands.sh stage2-stress
./project_commands.sh stage2
./project_commands.sh evaluate
```

Meaning of each command:

1. `export STAGE2_CONFIG=...output_space.yaml` selects the pinned Hugging Face council.
2. `prepare` builds the original tokenized corpus in `artifacts/prepared/s1k_1_1_cot_only/`.
3. `fetch-teachers` downloads, verifies and imports the three experts into
   `artifacts/stage1/duyentl04_abc/`.
4. `prune-cot` uses beam search (width 2) to delete reasoning steps anywhere
   in the trace while keeping at least 95% of the original answer likelihood.
   It writes the shortened corpus to `artifacts/prepared/trainhihi/` and uploads
   it to `sonspeed/Trainhihi`.
5. `stage2-cache` precomputes teacher targets on that shortened corpus and selects
   the initialization expert; cache: `artifacts/teacher_cache/output_space/`.
6. `stage2-stress` checks student training VRAM and update completion;
   report: `artifacts/stage2/output_space_stress_memory.json`.
7. `stage2` trains for one epoch on the pruned data, saving the student and logs in
   `artifacts/stage2/output_space/`.
8. `evaluate` optionally benchmarks that student, saving results in
   `artifacts/evaluation/p_align/`.

To train from the uploaded dataset instead of pruning again:

```bash
./project_commands.sh fetch-pruned
./project_commands.sh stage2-cache
./project_commands.sh stage2
```

For either route, the student output directory contains `final/adapters/student/`,
`checkpoint.pt`, `metrics.jsonl`, `performance.jsonl` and `reasoning_steps.jsonl`.
