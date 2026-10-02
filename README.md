# GAC-CoT-MTKD

Train three LoRA experts, merge them into one student adapter, then distill
with council-weighted step NLL.

## Setup

```bash
./project_commands.sh setup
# also install vLLM, needed by stage1-eval (Linux + CUDA)
INSTALL_VLLM=1 ./project_commands.sh setup
```

## Run

Complete pipeline (`prepare` → `stage1` → `stage1-eval` → `supervision` →
`stage2` → `evaluate`). `all` checks that vLLM is installed before it starts;
set `SKIP_STAGE1_EVAL=1` to drop `stage1-eval`:

```bash
./project_commands.sh all
```

Setup, tests, then that pipeline:

```bash
./project_commands.sh full
```

Tiny 0.5B smoke of `full`:

```bash
./project_commands.sh smoke
```

Each step:

```bash
./project_commands.sh prepare
./project_commands.sh stage1
./project_commands.sh stage1-eval
./project_commands.sh publish-stage1
./project_commands.sh supervision
./project_commands.sh stage2
./project_commands.sh evaluate
```

`cache` is optional and is **not** part of `all` / `full` / `smoke`. Stage 2
does not read teacher-cache artifacts.

```bash
./project_commands.sh cache
```

Tests (no model download):

```bash
./project_commands.sh test
```

```bash
./project_commands.sh help
```

## Multi-GPU

```bash
NPROC_PER_NODE=2 ./project_commands.sh all
```

`global_batch_size` is 2, so `world_size` must divide 2 (`1` or `2`).

## Resume

```bash
STAGE1_RESUME=artifacts/stage1/main/checkpoint.pt ./project_commands.sh stage1
STAGE2_RESUME=artifacts/stage2/main/checkpoint.pt ./project_commands.sh stage2
```

## Stage 2 merge ablation

Default method is `ta`. Override with `STAGE2_MERGE_METHOD` or `--merge-method`:
`ta`, `ties`, `dare_ties`, `tsv`, `iso_c`.

```bash
STAGE2_MERGE_METHOD=ties ./project_commands.sh stage2
STAGE2_MERGE_METHOD=dare_ties ./project_commands.sh stage2
STAGE2_MERGE_METHOD=tsv ./project_commands.sh stage2
STAGE2_MERGE_METHOD=iso_c ./project_commands.sh stage2
```

Write each run to its own directory:

```bash
STAGE2_MERGE_METHOD=ties ./project_commands.sh stage2
# or
python -m cot_mtkd.cli.train_stage2 \
  --config configs/stage2/qwen25_7b.yaml \
  --merge-method tsv \
  --set paths.output=artifacts/stage2/tsv
```

## Config overrides

```bash
DATA_CONFIG=configs/data/s1k_1_1.yaml ./project_commands.sh prepare
STAGE1_CONFIG=configs/stage1/qwen25_7b_m3.yaml ./project_commands.sh stage1
SIGNALS_CONFIG=configs/signals/main.yaml ./project_commands.sh supervision
STAGE2_CONFIG=configs/stage2/qwen25_7b.yaml ./project_commands.sh stage2
CACHE_CONFIG=configs/cache/qwen25_7b_top512_tail.yaml ./project_commands.sh cache
EVAL_CONFIG=configs/eval/p_align.yaml ./project_commands.sh evaluate
STAGE1_EVAL_CONFIG=configs/eval/stage1_experts.yaml ./project_commands.sh stage1-eval
HF_HUB_OFFLINE=1 ./project_commands.sh all
```

Defaults live in `configs/pipeline.yaml`. Artifacts go under `artifacts/`.
Method notes: `docs/stage1.md`, `docs/stage2.md`.

## Phase 1 repulsion ablation

Default force is the closed-form projection distance
(`stage1.rep_metric: projection_closed_form`). The previous geodesic
QR/SVD/autograd force remains available as `geodesic_autograd`.
`stage1.repulsion_weight` is \(\lambda_{rep}\); set `stage1.lambda_rep` only
when you want to override it. `dpp_topk_cap`, `dpp_token_frac < 1`, and
`dpp_every > 1` change the DPP method and stay off by default.

Same seed and step budget, three runs:

```bash
# closed-form projection force (default)
STAGE1_CONFIG=configs/stage1/qwen25_7b_m3.yaml ./project_commands.sh stage1

# legacy geodesic force
python -m cot_mtkd.cli.train_stage1 \
  --config configs/stage1/qwen25_7b_m3.yaml \
  --set stage1.rep_metric=geodesic_autograd \
  --set paths.output=artifacts/stage1/geodesic

# no repulsion
python -m cot_mtkd.cli.train_stage1 \
  --config configs/stage1/qwen25_7b_m3.yaml \
  --set stage1.repulsion_weight=0 \
  --set paths.output=artifacts/stage1/no_repulsion
```

Compare `rep_mean_d2`, `sft_nll_per_expert`, and `rep_seconds` in each
`metrics.jsonl`. This recipe does not claim which run is better.

### B gate

`stage1.rep_B_start_frac` is a fraction of the optimizer budget. At startup
the trainer sets

\[
t_B = \lceil \texttt{rep\_B\_start\_frac} \times T \rceil,
\quad
T = \lceil \texttt{epochs} \times N_{\mathrm{samples}} / \texttt{effective batch} \rceil.
\]

On the 1,000×3 run, \(T=94\) and \(f_B=0.10\), so \(t_B=10\). B turns on
fully at that 0-based index: `rep_B_ramp_steps` is 0, `rep_B_min_norm` is
null (no \(\tau_B\)), and `rep_B_rel_cap` is null (no relative cap). Startup
raises if \(t_B \ge T\). A nonzero ramp or a non-null cap is rejected,
because those mechanisms are not part of this run. The first step that
passes the gate logs `B repulsion ACTIVE at step t` once. Steps within 2 of
\(t_B\) also log `b_active`, \(\|F_B\|\), and the AdamW ratio. Every
`metrics.jsonl` row includes `b_active`, `rep_fb_norms` (\(\|F_B\|\) per
expert), and `rep_update_ratios`
(\(\|\eta\lambda_{rep} F\|/\|\Delta\phi\|_{\mathrm{AdamW}}\) per expert).
`step` counts completed updates, so the first row with `b_active: true` has
`step` = \(t_B+1\). An absolute `stage1.rep_B_start_step` may be set instead
of the fraction, but not both.

### DPP jitter

Cholesky starts at `dpp.jitter` (\(10^{-5}\)). A failed factorization
multiplies the jitter by 10, up to `dpp.max_jitter` (\(10^{-2}\)). Each
failed attempt on a token increments `cholesky_fallbacks` in
`metrics.jsonl`. The training loop runs `dpp_mode: probe`.

Logged with those rows: `sft_nll_per_expert` (length \(M\), numerator and
denominator summed across ranks before the division), `dpp_tail_mass`
(mean of \(1-\sum_{v\in V_k}\bar p(v)\) over DPP tokens), and
`dpp_ystar_outside_rate` (fraction of DPP tokens whose label is outside
\(V_k\), counted before \(y^*\) is removed from the support).

Timing only, without a training claim:

```bash
python scripts/bench_repulsion.py
python scripts/bench_repulsion.py --full
```

## Stage-1 outputs

`./project_commands.sh stage1` writes to `paths.output` of the Stage-1 config
(default `artifacts/stage1/main`):

```text
artifacts/stage1/main/
├── final/
│   ├── adapter_states.pt                   # all experts in one torch bundle (read by supervision / stage2)
│   └── adapters/
│       ├── expert_0/adapter_config.json
│       ├── expert_0/adapter_model.safetensors
│       ├── expert_1/...
│       └── expert_2/...                    # PEFT LoRA per expert (read by vLLM / publish)
├── benchmark/                              # same layout, saved at stage1.benchmark_checkpoint_epoch (epoch 2; distinct from final/)
├── checkpoint.pt                           # LoRA + optimizer + scheduler state for STAGE1_RESUME
├── manifest.json                           # adapter names, file paths, sha256 of every file
├── metrics.jsonl
└── config.yaml
```

Load one expert in PEFT:

```python
from peft import PeftModel
model = PeftModel.from_pretrained(base_model, "artifacts/stage1/main/final/adapters/expert_0")
```

## Stage-1 expert evaluation

`stage1-eval` runs right after `stage1` in `all`. It loads the base model in
vLLM once, then evaluates `expert_0`, `expert_1` and `expert_2` one after
another as LoRA adapters on AIME25, AIME24, AMC12 and MATH-500:

```bash
./project_commands.sh stage1-eval
# only some experts, a non-default run, or forced regeneration
STAGE1_EVAL_EXPERTS=expert_0,expert_2 ./project_commands.sh stage1-eval
./scripts/22_evaluate_stage1.sh --set paths.stage1=artifacts/stage1/no_repulsion \
  --set paths.output=artifacts/evaluation/stage1_no_repulsion
STAGE1_EVAL_FORCE=1 ./project_commands.sh stage1-eval
# evaluate benchmark/ instead of final/
./scripts/22_evaluate_stage1.sh --set stage1_checkpoint=benchmark
```

Config: `configs/eval/stage1_experts.yaml` (override with `STAGE1_EVAL_CONFIG`).
Each expert's adapter sha256 is checked against the Stage-1 manifest before it
is loaded. Every expert uses the same per-problem sampling seed. Results for
one (expert, benchmark) pair are cached and reused while the adapter and
settings stay the same, so an interrupted run resumes where it stopped.

```text
artifacts/evaluation/stage1/
├── expert_0/
│   ├── aime25.jsonl ... math500.jsonl      # generations, finish_reasons, correct flags
│   ├── aime25.metrics.json ...
│   └── manifest.json                       # per-benchmark metrics + macro average
├── expert_1/ ...
├── expert_2/ ...
├── summary.md                              # Pass@1 / Pass@3 table, one row per expert
└── manifest.json
```

`length_capped_fraction` in each metrics file is the share of generations that
stopped at the token limit.

## Hyperparameters

| Group | Hyperparameter | Value |
| --- | --- | --- |
| LoRA | rank `r` | 16 |
| LoRA | alpha | 16 |
| LoRA | dropout | 0.05 |
| LoRA | target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| LoRA | `use_rslora` / `use_dora` / `use_qalora` | false / false / false |
| Training | epochs | 3 |
| Training | effective batch size | 32 |
| Training | `per_device_train_batch_size` | 1 |
| Training | `gradient_accumulation_steps` | 32 |
| Training | max sequence length | 32,768 |
| Training | learning rate | 5e-5 |
| Training | `max_grad_norm` | 1.0 |
| Training | `label_smoothing_factor` | 0.0 |
| Optimizer | optimizer | AdamW |
| Optimizer | betas, eps | (0.9, 0.999), 1e-8 |
| Optimizer | weight decay | 0.0 |
| Scheduler | schedule | cosine with warmup (`warmup_ratio` 0.1, `LambdaLR`) |
| Repulsion | `rep_B_start_frac` | 0.10, so \(t_B=\lceil 0.10\times 94\rceil=10\) |
| Repulsion | ramp / \(\tau_B\) / relative cap | 0 / off / off |
| DPP | jitter / max jitter | \(10^{-5}\), raised by 10× up to \(10^{-2}\) |
| Evaluation | samples per problem `n` | 3 |
| Evaluation | temperature | 0.6 |
| Evaluation | `top_p` | 0.9 |
| Evaluation | `repetition_penalty` | 1.05 |
| Evaluation | `max_tokens` | 4,096 |
| Evaluation | vLLM `max_model_len` | 4,096 |
| Evaluation | `tensor_parallel_size` | 1 |
| Evaluation | `gpu_memory_utilization` | 0.8 |

`max_model_len` bounds prompt plus completion, so a completion stops at
`4096 − prompt_tokens` tokens even though `max_tokens` is 4,096.

### Metrics

For problem \(i\) with correctness flags \(y_1, y_2, y_3\) of its three
generations, over \(N\) problems:

- **Pass@1** = \(\frac{1}{N}\sum_i \frac{y_1+y_2+y_3}{3}\), which equals
  total correct generations / \(3N\).
- **Pass@3** = \(\frac{1}{N}\sum_i \mathbb{1}[y_1 \lor y_2 \lor y_3]\): a
  problem counts as solved if any of its three generations is correct.

The macro average is the unweighted mean over benchmarks.

## Publish Stage-1 LoRA experts

Set `HF_REPO_ID` before Stage 1 to publish automatically, or run:

```bash
export HF_TOKEN=hf_xxx
export HF_REPO_ID=username/repo-name
./project_commands.sh publish-stage1
```

Authenticate with `HF_TOKEN` or `hf auth login`. The publisher uploads each
expert's `adapter_config.json` and `adapter_model.safetensors`, plus a model
card, then verifies the new commit. It can run again after a completed
training resume.

New repositories are private by default. Set `HF_REPO_PRIVATE=false` or pass
`--public` to `cot-mtkd-publish-stage1`. Existing repos keep their visibility.
Use `STAGE1_CONFIG` or `--stage1-dir` for a non-default run.
