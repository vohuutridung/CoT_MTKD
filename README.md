# CoT-MTKD

Train three LoRA experts, distill them into one student, and evaluate the student.

## Quick start: Phase 2 + evaluation on a new server

This route uses the Phase-1 experts already published as `duyentl04/abc`.
On a fresh machine with CUDA and Python 3.10+:

```bash
git clone --branch duyentl https://github.com/vohuutridung/CoT_MTKD.git
cd CoT_MTKD
CUDA_VISIBLE_DEVICES=0 nohup ./project_commands.sh stage2-eval > stage2_eval.log 2>&1 &
```

`stage2-eval` runs every step in order:

1. `setup` creates `.venv` for training; `setup-eval` creates `.venv-eval` with vLLM.
   vLLM pins its own torch version, which is why it gets a separate venv.
2. `prepare` builds the tokenized s1K-1.1 corpus in `artifacts/prepared/s1k_1_1_cot_only/`.
3. `fetch-teachers` downloads and verifies the three experts into
   `artifacts/stage1/duyentl04_abc/`.
4. `stage2-cache` builds the council cache in `artifacts/teacher_cache/output_space/`.
   On one H200 this takes about 1.5–2 h and about 16 GB of disk.
5. `stage2` trains one student per run in `STAGE2_RUNS`, saving each under
   `artifacts/stage2/<run>/`.
6. `evaluate` runs the P-ALIGN protocol with vLLM on seeds 42, 43 and 44 and writes
   `artifacts/evaluation/<run>/summary.json`.
7. Finally it prints the Pass@1 table (mean ± std over seeds).

Every step skips work that is already done, and training resumes from
`checkpoint.pt`. If the job stops, run the same command again.

To train the method together with all its controls, and spread evaluation seeds
over three GPUs:

```bash
CUDA_VISIBLE_DEVICES=0 EVAL_GPUS=0,1,2 \
  STAGE2_RUNS="main geometric arithmetic single sft" \
  nohup ./project_commands.sh stage2-eval > stage2_eval.log 2>&1 &
```

| Variable | Default | Meaning |
| --- | --- | --- |
| `STAGE2_RUNS` | `main` | Runs to train, by config suffix (see [controls](#phase-2-objective-and-controls)) |
| `EVAL_GPUS` | first visible GPU | Evaluation GPUs; the seeds are spread over them |
| `EVAL_SEEDS` | `42 43 44` | Evaluation seeds |
| `EVAL_GPU_MEMORY_UTILIZATION` | `0.90` | vLLM memory fraction; lower it (e.g. `0.45`) on a shared GPU |
| `NPROC_PER_NODE` | `1` | Training processes (GPUs) per run |
| `RUN_STRESS` | `0` | `1` runs the H200 Phase-2 memory preflight first |
| `SKIP_EVAL` | `0` | `1` trains only |
| `PROC_PREFIX` | `hieunq10` | Process titles in nvitop: `<prefix>_s2_<run>`, `<prefix>_eval_<run>` |

FlashAttention 2 is optional; the loader falls back to SDPA when it is not installed.

## Phase 2 objective and controls

The cache (version 2) stores two things:
- the step JS;
- each expert's reduced distribution, i.e. its support tokens plus one tail bucket.

The target is built at training time, so all runs share one cache:

- `rho_mapping: ecdf`: ρ_s is the rank of the step's JS among all corpus steps,
  so it is uniform on (0, 1]. The raw JS / log M of the `duyentl04/abc` council
  has a median of about 0.02. With that value the power mean is the geometric
  mean on almost every step.
- `kd_temperature: 1`: at T = 2 about 64% of the target mass falls in the tail
  bucket, so most of the KD signal only matches one lumped probability.
- `loss_normalization: token`: with equal step weights, steps of ≤ 10 tokens
  got 18% of the loss while holding only 2.8% of the tokens (mostly "Wait,"
  and "Hmm."). ρ is still defined per step.
- `student_init: base`: the student starts from a fresh LoRA and makes one pass.
  `best_expert` instead continues the lowest-SFT expert, which has already seen
  the same data for three epochs.

| Run | Config | Target |
| --- | --- | --- |
| `main` | `qwen25_7b_output_space.yaml` | council, ECDF ρ (method) |
| `geometric` | `qwen25_7b_output_space_geometric.yaml` | council, ρ = 0 |
| `arithmetic` | `qwen25_7b_output_space_arithmetic.yaml` | council, ρ = 1 |
| `single` | `qwen25_7b_output_space_single.yaml` | lowest-SFT expert only (M = 1) |
| `sft` | `qwen25_7b_output_space_sft.yaml` | gold tokens only (plain SFT) |

The runs differ only in training-time keys and the output directory.

To reproduce the old behaviour, set:
- `rho_mapping: linear`
- `kd_temperature: 2.0`
- `loss_normalization: step`
- `student_init: best_expert`
- `learning_rate: 2.0e-5`

Changing `kd_temperature` or `js_temperature` builds a new cache. The council
size is read from the Phase-1 manifest, and any M ≥ 2 works.

## Step by step

```bash
./project_commands.sh setup
./project_commands.sh setup-eval
export CUDA_VISIBLE_DEVICES=0
export STAGE2_CONFIG=configs/stage2/qwen25_7b_output_space.yaml

./project_commands.sh prepare
./project_commands.sh fetch-teachers
./project_commands.sh stage2-cache
./project_commands.sh stage2-stress     # optional H200 preflight
./project_commands.sh stage2
STAGE2_DIR=artifacts/stage2/output_space ./project_commands.sh evaluate-after-stage2
```

To train Phase 1 locally first, use `STAGE2_CONFIG=configs/stage2/qwen25_7b_output_space_local.yaml`
and run `stage1-stress` and `stage1` after `prepare`. The experts are saved in
`artifacts/stage1/main/`. Then run `stage2-cache` and `stage2` as above.

## Evaluation

`evaluate` reproduces the exp_s1k P-ALIGN evaluation (`eval/run_eval.py`):

| Setting | Value |
|---|---|
| Benchmarks | MATH-500 (500), AIME 2024 (30), AIME 2025 opencompass I+II (30), AMC12 from the P-ALIGN repo (83); all vendored in `data/eval/` |
| Prompt | chat template, one user turn: `Please reason step by step, and put your final answer within \boxed{}. <problem>` |
| Engine | vLLM, LoRA adapter, bf16, `max_model_len = 4096 + 2048`, prefix caching |
| Sampling | k = 3 samples, T = 0.6, top-p = 0.9, top-k off, repetition penalty 1.05, 4096 new tokens, per-request seed |
| Seeds | 42, 43, 44; report mean ± std over seeds |
| Grading | `math_verify` with the gold wrapped in `\boxed{}`, then normalized match of the last boxed answer (or last number) |
| Metrics | Pass@1 = mean accuracy over the k samples; Pass@3 = any sample correct; Avg = unweighted mean over the four benchmarks |

Outputs go to `artifacts/evaluation/<run>/`:
- `seed<N>/<benchmark>.jsonl`: completions, token counts, truncation, and per-sample correctness.
- `seed<N>/result.json`: per-seed scores.
- `summary.json`: mean and std over seeds.

A seed whose `result.json` already exists is skipped; `--set overwrite=true` re-runs it.

Other ways to evaluate:

```bash
# one adapter, any GPU
CUDA_VISIBLE_DEVICES=0 EVAL_ADAPTER=artifacts/stage2/output_space \
  EVAL_OUTPUT=artifacts/evaluation/output_space ./project_commands.sh evaluate
# the untuned base model
EVAL_ADAPTER=base EVAL_OUTPUT=artifacts/evaluation/base ./project_commands.sh evaluate
# pack the student for another machine (or upload with HF_REPO=<user>/<repo>)
STAGE2_DIR=artifacts/stage2/output_space ./project_commands.sh pack-student
```

`EVAL_ADAPTER` accepts any of these:
- a Stage-2 output directory;
- an unpacked `pack-student` archive;
- a PEFT adapter directory;
- a Hub repo id;
- `base`.

`--set engine.name=hf` switches to a slow `transformers` fallback. Report vLLM numbers.

## Resume and overrides

```bash
STAGE2_CONFIG=configs/stage2/qwen25_7b_output_space.yaml \
  STAGE2_RESUME=artifacts/stage2/output_space/checkpoint.pt ./project_commands.sh stage2
STAGE1_RESUME=artifacts/stage1/main/checkpoint.pt ./project_commands.sh stage1
EVAL_CONFIG=configs/eval/p_align.yaml ./project_commands.sh evaluate --set limit=5 --set seeds=[42]
HF_HUB_OFFLINE=1 ./project_commands.sh stage2-eval
```

Outputs are written under `artifacts/`.
