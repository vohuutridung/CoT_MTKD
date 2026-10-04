# CoT-MTKD

Train three LoRA experts and distill them into one student adapter.

## Setup

Clone the `output-space` branch and install the project with Python 3.10 or newer:

```bash
git clone --branch output-space https://github.com/vohuutridung/CoT_MTKD.git
cd CoT_MTKD
./project_commands.sh setup
```

`setup` creates `.venv` and installs the project and test dependencies. The
commands below automatically use `.venv`; activating it is unnecessary. Training
requires a CUDA-capable PyTorch installation. FlashAttention-2 is optional and
must match the machine's CUDA/PyTorch build; the loader falls back to SDPA if it
is unavailable. The stress commands below specifically require one visible H200.

## Run

### Train both Phase 1 and Phase 2

After setup, run these commands in order. This route trains a new local council
and distills those same experts using the local-teacher Phase-2 configuration:

```bash
export STAGE2_CONFIG=configs/stage2/qwen25_7b_output_space_local.yaml

./project_commands.sh prepare
CUDA_VISIBLE_DEVICES=0 ./project_commands.sh stage1-stress
./project_commands.sh stage1
./project_commands.sh stage2-cache
CUDA_VISIBLE_DEVICES=0 \
  STAGE2_STRESS_OUTPUT=artifacts/stage2/output_space_local_stress_memory.json \
  ./project_commands.sh stage2-stress
./project_commands.sh stage2

.venv/bin/python -m cot_mtkd.cli.evaluate \
  --config configs/eval/p_align.yaml \
  --set paths.stage2=artifacts/stage2/output_space_local \
  --set paths.output=artifacts/evaluation/p_align_local
```

Purpose of each command above:

1. `export STAGE2_CONFIG=...output_space_local.yaml` selects the experts produced
   by Phase 1 in `artifacts/stage1/main`. It also selects separate local-teacher
   cache and student output directories.
2. `prepare` downloads the pinned s1K-1.1 dataset and Qwen tokenizer, serializes
   only `deepseek_thinking_trajectory` (including its own final answer), and records
   token regions and reasoning-step boundaries in `artifacts/prepared/s1k_1_1_cot_only`
   for both phases. It excludes the four verified incomplete trajectories
   `s1k-0135`, `s1k-0324`, `s1k-0392`, and `s1k-0438`, retaining 996 samples
   with their original IDs. `deepseek_attempt` is never read or appended.
   Responses end directly with EOS after the CoT; prompt tokens remain masked.
   The 32,768-token limit rejects oversized responses to preserve their final
   answer. Rerun `prepare` after this change; old artifacts containing `attempt`
   are rejected, and checkpoints tied to the old corpus cannot be resumed.
3. `stage1-stress` checks Phase-1 VRAM on the longest real example and synthetic
   32k examples, with every Phase-1 interaction enabled. It uses a temporary
   model and writes `artifacts/stage1/stress_memory.json`.
4. `stage1` trains the three LoRA experts and saves their adapter bundle,
   checkpoint and manifest in `artifacts/stage1/main`.
5. `stage2-cache` precomputes support/tail targets and SFT scores on the prepared
   corpus, selecting the best expert to copy into the student. Its manifest is saved in
   `artifacts/teacher_cache/output_space_local/<fingerprint>`.
6. `stage2-stress` checks the actual Phase-2 forward, loss, gradients and temporary
   optimizer update at real and synthetic context lengths. The explicit output
   override saves its report as `artifacts/stage2/output_space_local_stress_memory.json`.
7. `stage2` trains the student for one epoch with cached KD + SFT and writes
   checkpoints, the final student adapter and logs to `artifacts/stage2/output_space_local`.
8. The final `evaluate` command optionally generates and grades the local
   student on AIME 2025, AIME 2024, AMC and MATH-500. It writes benchmark outputs
   to `artifacts/evaluation/p_align_local`; it is separate from training.

### Train Phase 2 using the existing Hugging Face experts

After the same clone/setup steps, run this route when using the already-trained
experts from `duyentl04/abc`. Phase 1 does not need to be run again:

```bash
export STAGE2_CONFIG=configs/stage2/qwen25_7b_output_space.yaml

./project_commands.sh prepare
./project_commands.sh fetch-teachers
./project_commands.sh stage2-cache
CUDA_VISIBLE_DEVICES=0 ./project_commands.sh stage2-stress
./project_commands.sh stage2

# Optional benchmark evaluation after training finishes:
./project_commands.sh evaluate
```

Purpose of each command above:

1. `export STAGE2_CONFIG=...output_space.yaml` selects the pinned Hugging Face
   experts and the default Phase-2 artifact paths.
2. `prepare` builds the current tokenized training corpus. It is still required
   on a fresh clone, even when the experts were trained elsewhere.
3. `fetch-teachers` downloads, verifies and imports the three frozen PEFT adapters
   from the pinned Hub revision into `artifacts/stage1/duyentl04_abc`.
4. `stage2-cache` precomputes support/tail targets and selects the best SFT expert
   from the imported council, saving `artifacts/teacher_cache/output_space/<fingerprint>/manifest.json`.
5. `stage2-stress` checks Phase-2 VRAM and update completion on one H200. Its
   report is `artifacts/stage2/output_space_stress_memory.json`; it does not
   save a trained student.
6. `stage2` runs one training epoch and exports the student to
   `artifacts/stage2/output_space/final/adapters/student`. The same output root
   contains `checkpoint.pt`, `manifest.json`, `metrics.jsonl` and the detailed
   `reasoning_steps.jsonl` and `performance.jsonl` logs.
7. `evaluate` optionally benchmarks the exported student. Training is complete
   when `stage2` finishes; evaluation is not needed to export the adapter.

The H200 stress checks are preflights; on another CUDA GPU, use the individual
training commands and a memory check suited to that hardware. Neither route
needs the legacy `supervision` or `cache` commands for this Phase-2 method.

### Shortcuts and tests

With the default Hub-teacher config, `all` runs `prepare`, `fetch-teachers`,
`stage2-cache`, `stage2`, and `evaluate`. It skips Phase-1 training and both
H200 preflights. `full` runs setup and tests before that same `all` route:

```bash
STAGE2_CONFIG=configs/stage2/qwen25_7b_output_space.yaml ./project_commands.sh all
STAGE2_CONFIG=configs/stage2/qwen25_7b_output_space.yaml ./project_commands.sh full
```

Run the local tests independently; they do not download a model, and CUDA tests
are skipped when CUDA is unavailable:

```bash
./project_commands.sh test
```

## Multi-GPU

Run preparation/import once, and use `NPROC_PER_NODE` for distributed council
preprocessing, training and evaluation. Both global batch and dataset record
count must divide the GPU count; the current 996-sample corpus supports 1/2/4
GPUs with the default batch 16. For example, using four GPUs with the Hub experts:

```bash
export STAGE2_CONFIG=configs/stage2/qwen25_7b_output_space.yaml
./project_commands.sh prepare
./project_commands.sh fetch-teachers
NPROC_PER_NODE=4 ./project_commands.sh stage2-cache
NPROC_PER_NODE=4 ./project_commands.sh stage2
NPROC_PER_NODE=4 ./project_commands.sh evaluate
```

The stress commands require one visible H200 rather than a distributed launch.

## Phase 1 adaptive non-target support

Phase 1 uses **local Kneedle over the top-512 non-target logits**. Each expert
excludes the observed target and probes `K = min(search_k, vocabulary_size - 1)`
descending candidates. For one-based rank `j`, both axes use that same window:

```text
x_j = (j - 1) / (K - 1)
u_j = (z_j - z_K) / (z_1 - z_K + epsilon)
raw_k = 1 + argmax((1 - x) - u)       # argmax uses zero-based indices
k = min(K, max(raw_k, k_min))
```

The default configuration is `kneedle.search_k: 512`, `kneedle.k_min: 8`.
There is no additional upper clipping. A single candidate gives `raw_k = k = 1`;
when `K < 8`, all available candidates are retained. Flat logits select the
first raw rank before applying the lower bound; ties select the first maximum.
Each expert retains its own top-`k` candidates, then DPP uses the detached union
of these supports. With three experts, this union can contain up to 1,536 token
identities. The step-level DPP objective uses this shared union support.

For each reasoning step, DPP uses the mean token Gram matrix `L_s` of the
experts' L2-normalized exponentiated-logit vectors on that union:

```math
L_{\mathrm{DPP},s}
= \frac{1}{M}\left[M\log(1+\epsilon_D)
  - \log\det(\mathbf L_s+\epsilon_D\mathbf I)\right].
```

This normalization gives zero loss for an identity Gram matrix, up to numerical
rounding. It adds a constant to the earlier negative log-volume loss, so its
gradient is unchanged at fixed jitter. Jitter starts at `1e-4` and retries up
to `1e-2`; both terms use each step's actual jitter, not the batch maximum.
Losses are averaged over steps within each example, then over examples.

`artifacts/stage1/main/metrics.jsonl` records `mean_raw_k`, `mean_selected_k`,
`raw_k_histogram`, `selected_k_histogram`, `support_selection_count`, and
`probe_saturation_rate`. Histogram index is the raw/final support size; counts
sum across experts, accumulated microbatches and distributed ranks for that
optimizer update. Batches with no reasoning tokens have zero counts. Saturation counts raw knees
at the last available candidate, including windows smaller than 512.

This support-selection change alters the training method and its configuration
fingerprint; start a new Phase-1 run rather than resuming a checkpoint produced
with the earlier criterion. Check H200 memory again for the larger union support.

## Phase 1 scalar objective

Phase 1 minimizes the sum of the three experts' SFT losses plus output-space
DPP and effective-update RBF repulsion:

```math
L = \sum_{m=1}^{M} L_{\mathrm{SFT}}^{(m)}
  + 0.2 L_{\mathrm{DPP}} + 0.01 L_{\mathrm{RBF}},
\qquad
L_{\mathrm{RBF}} = \frac{1}{\binom{M}{2}}\sum_{m<q}e^{-D_{mq}/h}.
```

For each LoRA module, `Delta W = (alpha/r) B A`. `D_mq` is the mean across
modules of squared Frobenius distances divided by the module's output/input
dimension product. The low-rank calculation keeps both A and B in the autograd
graph without materializing a full `B A` matrix. Minimizing the mean pairwise
kernel produces repulsion through each expert's own factors.

SFT keeps its global response-token mean per expert. DPP keeps its step means
within each reasoning-bearing example, then its global example mean. RBF is
evaluated once per optimizer window, after task-gradient accumulation and
distributed reduction; its gradient is added once without a microbatch/world-size
multiplier. Each expert receives its own SFT/DPP gradient plus the weighted RBF
loss gradient, followed by total-gradient norm clipping at `1.0` and AdamW.

`h` uses the existing median-distance/log(M+1) heuristic, EMA decay `0.9`, and
floor `1e-12`. The EMA estimate is detached and fixed during each backward, so
this is a scalar objective conditional on that update's bandwidth. With all
initial B factors zero, effective updates coincide: RBF loss is 1 and its gradient
is zero on the initial update; the loss remains enabled throughout training.

`stage1.rbf_weight: 0.01` is a conservative starting value for the uncapped loss,
not a tuned result. Logs include `rbf_loss`, `weighted_rbf_loss`, `phase1_loss`,
`task_gradient_norms`, `rbf_gradient_norms`, `weighted_rbf_gradient_norms`, and
`rbf_task_gradient_ratios`. The ratio for expert m is
`||lambda_R grad L_RBF|| / max(||grad(SFT_m + lambda_D DPP)||, 1e-12)` before
final clipping. Use it with competence/diversity metrics when tuning lambda_R.
`sft_nll` is the average expert NLL for logging; `phase1_loss` uses their sum.
The scalar-objective identity is included in the run fingerprint and manifest.

The corpus has 996 fixed CoT-only examples. With three epochs, global batch 16,
microbatch 1, and accumulation crossing epoch boundaries, the default run has
187 optimizer updates. Expert seeds remain 42/43/44, lambda_D remains 0.2, and
the local Kneedle/union support and Phase-2 objectives keep their current definitions.

## Phase 1 forward mode and H200 memory check

Phase 1 uses one forward per expert by default. The existing two-pass replay
remains available with `STAGE1_FORWARD_MODE=two_pass ./project_commands.sh stage1`.
Phase 1 uses `stage1.interaction_mode: full`: SFT, DPP, and RBF loss are active
from the first optimizer update through the last, with interaction strength 1.
SFT and weighted DPP share one transformer VJP, preserving the original global
token and sample normalizers. The cosine learning-rate schedule and its 10%
warm-up are unchanged; they do not disable any objective or interaction.

After preparing the dataset and before full H200 training, run
`./project_commands.sh stage1-stress` with exactly one H200 visible. It executes
two full-interaction cases: the longest prepared real sample and a synthetic
32,768-token extension of its reasoning. Both use the normalized combined task
gradient and one transformer VJP, with DPP and RBF loss active from the first
warm-up iteration. The default
config keeps gradient checkpointing enabled, places probe/support on CUDA, and
uses `lm_head_chunk_tokens: 32768` while retaining configurable head chunking.

Each case runs one warm-up followed by one measured iteration. Adjust these
with `STAGE1_STRESS_WARMUP` and `STAGE1_STRESS_REPETITIONS` (warm-up may be zero;
repetitions must be positive). The report at `artifacts/stage1/stress_memory.json`
records peak allocated/reserved VRAM across warm-up and measurement, plus measured
times and completion counters. Times are for **one sample/microbatch including
an optimizer update**, not a training update accumulating the global batch of
16 samples. These longest-sample timings do not estimate the complete training run.

A worst reserved peak below 110 GiB is marked ready for one-pass; 110–120 GiB
asks for review; 120 GiB or more, or any CUDA OOM, recommends the two-pass fallback
and exits with status 2. Both cases must complete before the preflight is ready.
Stress updates affect only the temporary model in that process; no training
checkpoint or adapters are saved.

## Phase 2: cached adaptive support + tail MTKD

The `output-space` branch defaults to three frozen LoRA experts distilling into
one student LoRA on fixed teacher-forced trajectories. Phase 1 is unchanged.
Agreement favors geometric consensus; disagreement moves the generalized power
mean toward arithmetic coverage. No teacher or student rollout is generated.

Teachers default to the pinned
[duyentl04/abc council](https://huggingface.co/duyentl04/abc/tree/52aff0a878826b09fa62fa5a8229a6d49910ff6a).
`fetch-teachers` verifies SHA-256 weights, backbone revision, compatible LoRA
structure and finite tensors, preserving uploaded source files and historical
training-data identity. The upload does not contain the original prepared-data
manifest, so exact historical dataset identity remains unverified. The importer
records that limitation rather than rewriting the historical fingerprint.
Local experts use `qwen25_7b_output_space_local.yaml`; remote and local configs
use separate cache/output roots with identical objective defaults.

### Support and two temperatures

`stage2-cache` calls the existing Phase-1 `full_vocab_probe`,
`stage1.kneedle.local_k_from_probe`, and `build_union_support` under `no_grad`.
For each expert and reasoning-content token it excludes observed gold `y_t`,
searches `K=min(512,|V|-1)` descending non-target logits, and applies exactly:

```math
x_j=\frac{j-1}{K-1},\quad
u_j=\frac{z_j^\downarrow-z_K^\downarrow}{z_1^\downarrow-z_K^\downarrow+\epsilon},\quad
\hat k=1+\arg\max_j[(1-x_j)-u_j],\quad
k=\min(K,\max(\hat k,8)).
```

The shared support is `V_t = union_m S_m,t ∪ {y_t}`. Valid IDs are ascending and
unique; gold appears exactly once. There is no additional `k_max`. K=1/0 retain
Phase-1 edge semantics. Sorting, Kneedle and selection receive no gradient.

JSD uses `js_temperature=1`, restricts and renormalizes on `V_t`, and has no tail:

```math
\pi^{JS}_{m,t}=\operatorname{softmax}_{V_t}(z_{m,t}/T_{JS}),\quad
\bar\pi_t=\frac13\sum_m\pi^{JS}_{m,t},\quad
D^{JS}_t=\frac13\sum_m KL(\pi^{JS}_{m,t}\Vert\bar\pi_t).
```

This equals restriction+renormalization of the full softmax because its partition
function cancels. Each step shares exactly one scalar:

```math
D^{JS}_s=\frac1{|T_s|}\sum_{t\in T_s}D^{JS}_t,\qquad
\rho_s=\operatorname{clamp}(D^{JS}_s/\log 3,0,1).
```

There is no percentile, threshold, sigmoid or learned calibration.

KD independently uses `kd_temperature=2` and **full-softmax mass**, without
renormalizing support first. Its categorical space is `V_t ∪ {tail}`:

```math
\log Z_{m,t}=\operatorname{logsumexp}(z_{m,t}/T_{KD}),\quad
r_{m,t}(v)=\exp(z_{m,t,v}/T_{KD}-\log Z_{m,t}),\quad
r_{m,t}(\bot)=1-\sum_{v\in V_t}r_{m,t}(v).
```

Only support logits are gathered; no full teacher probability vector is kept.
Tail calculation uses `-expm1(log support mass)`. Near saturation it computes
outside-support logsumexp, preserving extremely small tails and student gradients.
An empty complement gets a `1e-30` floor; floors, roundoff corrections and
complement fallbacks are counted. Reduced distributions are corrected for
summation roundoff to sum to one. Full-vocabulary logits still occur in bounded
head chunks to calculate the partition function and hard-label CE.

The target is the normalized, detached **power mean**, on support plus tail:

```math
q_{s,t}(v)\propto
\begin{cases}
[\frac13\sum_m r_{m,t}(v)^{\rho_s}]^{1/\rho_s},&\rho_s>0,\\
\exp(\frac13\sum_m\log r_{m,t}(v)),&\rho_s=0.
\end{cases}
```

FP64 log-domain pooling retains the positive-power correction near zero; this
is never linear interpolation. `rho=0` is normalized geometric mean, `rho=1`
arithmetic mean, both on reduced categories. The student forms `r_S` identically.

### Objective, mask and initialization

```math
L^{KD}_s=\frac{T_{KD}^2}{|T_s|}\sum_{t\in T_s}KL(\operatorname{sg}(q_{s,t})\Vert r_{S,t}),\qquad
L^{SFT}_s=\frac1{|T_s|}\sum_{t\in T_s}-\log\operatorname{softmax}(z_{S,t})_{y_t},\qquad
L=\operatorname{Mean}_{sample}\operatorname{Mean}_{step}(L^{KD}_s+0.25L^{SFT}_s).
```

KD and SFT share the existing `output_space.plan_record` content mask: retained
complete `TokenRegion.REASONING` steps only, within the 32,768-token context.
Delimiters/control/EOS stay in context without a loss; no separate gold-solution
suffix is added. An answer already inside the reasoning trajectory is reasoning
content. Empty samples contribute zero and remain in the sample denominator;
a fully empty batch skips optimizer and scheduler while advancing data progress.

The same preprocessing pass scores each expert with ordinary T=1 NLL using
exactly token→step→sample means, then picks the lowest corpus SFT score. Ties
choose the first expert in manifest adapter order. The complete winning adapter
is copied into the student; no merging or barycenter distance is used.
The former Stage-2 selection pass and full-vocabulary online output-space KD have
been removed. The legacy PAG/features tool also no longer computes that selection.

### Static council cache

Run `./project_commands.sh stage2-cache` **before** `stage2`. Preprocessing does
one decoder forward per expert per retained sample, a bounded head sweep for
Phase-1 probes, and a second head sweep for JSD, reduced probabilities and SFT
scores. Reduced teacher values are retained only within a step until its rho is
known. Student training loads only the student model, winning adapter and cached
final targets; it never instantiates teacher adapters or forwards the council.
A cache miss in training fails with the preprocessing command to run.

Cache version 1 lives at `paths.teacher_cache_dir/<fingerprint>/`. Each sample
has a checksummed safetensors file containing:

- Ragged ascending `support_ids` (int32), `support_offsets` (int64), token positions
  (int32), and reasoning `step_offsets` (int64).
- Final normalized `support_log_target` and one `tail_log_target` per token (FP32).
  Log storage preserves tiny mass that FP16 probabilities could underflow. The
  loaded target is normalized for FP32 serialization roundoff before KL.
- Raw step JSD/rho (FP64), per-expert raw/selected K (int16), union sizes (int16),
  and gold-present-before-add flags (bool).

`index.json` owns every sample file/checksum; `manifest.json` owns the index hash,
expert SFT scores, deterministic winner, statistics and numerical counters.
`best_expert.pt` owns only the selected LoRA and has its own SHA-256.
Files and the final manifest are published atomically; incomplete preprocessing
has no usable manifest and must rerun. Cache hits verify content and skip model
loading. Training verifies every cache file at startup.

The identity binds teacher bundle/checkpoint SHA-256 and manifest, backbone name,
revision/precision/attention implementation, prepared manifest/data/tokenizer,
max length, search_k/k_min, both temperatures, support/mask/storage semantics,
head chunk size, PyTorch version, and source-file SHA-256 for selection,
preprocessing, loss, planner and model code. Epoch count, optimizer, global batch,
resume and SFT weight are excluded because they do not alter static targets.
Changing any bound field selects a fresh fingerprint directory; an explicitly
loaded incompatible fingerprint or corrupted cache fails loudly.

### Defaults and diagnostics

| Config key | Default |
| --- | --- |
| `aggregation.js_temperature` | `1.0` |
| `aggregation.kd_temperature` | `2.0` |
| `aggregation.sft_weight` | `0.25` (set `0.5` for an ablation) |
| `aggregation.search_k` / `k_min` | `512` / `8`, no extra cap |
| `aggregation.teacher_execution` | `precomputed_support_tail` |
| `stage2.epochs` | `1` |
| `stage2.global_batch_size` / `micro_batch_size` | `8` / `1` |
| `stage2.gradient_accumulation_steps` | `null`: derive `8/(world_size×micro)` |
| `stage2.max_length` | `32768` |
| `runtime.lm_head_chunk_tokens` | `4096` |
| `runtime.preprocessing_hidden_storage` | `cpu` |
| `model.gradient_checkpointing` | `true` |

Nondivisible global batches or conflicting explicit accumulation fail. A final
partial effective batch is normalized by its actual sample count. To change the
SFT anchor, run the training CLI with `--set aggregation.sft_weight=0.5`; the same
cache is reusable, but a different training configuration requires a new output
run and cannot resume the old checkpoint.

The current runner processes records sequentially inside each loader microbatch;
raising `micro_batch_size` does not batch decoder forwards or improve GPU
parallelism. Keep it at 1 until a batched gradient path is implemented and
benchmarked. With 996 samples, one epoch and one GPU, the defaults give 125
optimizer updates, including a final four-sample update. Epoch and batch changes
reuse the same council cache.

The cache manifest records support/union mean, median, p90, p95 and max, per-expert
raw/selected-K histograms, gold-present rate, raw step JSD/rho histograms and
quantiles, target tail mass, cache bytes/counts/wall time and anomalies.
`metrics.jsonl` records cache hit/fingerprint/init scores and winner, KD, SFT,
weighted SFT, total loss, both temperatures, JSD/rho, target/student tail mass,
reduced target entropy and numerical counters. `reasoning_steps.jsonl` gives the
same diagnostics per step, with original sample/step/token offsets and epoch.
Detailed rows are emitted independently of aggregate logging frequency.

`performance.jsonl` retains host wall timing, chunk counts, cache-hit step counts,
zero recomputed steps/teacher-head sweeps, throughput and allocator memory.
Host timing does not synchronize CUDA; CPU memory fields are null. Distributed
ranks write separate step/performance files. Resume trims rows after the saved
checkpoint cursor and partial trailing writes; final manifests own log hashes.
Checkpoints bind the full training config plus council-cache fingerprint.

On one H200, run `stage2-cache`, `stage2-stress`, then `stage2`. The stress command
precomputes the synthetic trajectory separately, releases teachers, then measures
cached student-only updates for longest real and full-context synthetic samples.
Synthetic preprocessing is reported separately and excluded from training timing.
No 7B/H200 speed, memory, training duration or model-quality claim is established
by CPU unit/tiny-Qwen tests. Missing H200 yields an unmeasured requirement report.

The older `qwen25_7b_task_geometry.yaml` remains an explicit, separate legacy
online gradient-space objective. It now uses best-SFT initialization from the
council preprocessing artifact; its teacher-gradient/answer-anchor objective and
batch defaults stay separate. Its online full-vocabulary losses are not used by
the default output-space method. Legacy fixed Top-512/PAG cache tooling is also
separate and cannot substitute for the new council cache.

## Resume training

```bash
STAGE1_RESUME=artifacts/stage1/main/checkpoint.pt ./project_commands.sh stage1
STAGE2_CONFIG=configs/stage2/qwen25_7b_output_space.yaml \
  STAGE2_RESUME=artifacts/stage2/output_space/checkpoint.pt \
  ./project_commands.sh stage2
STAGE2_CONFIG=configs/stage2/qwen25_7b_output_space_local.yaml \
  STAGE2_RESUME=artifacts/stage2/output_space_local/checkpoint.pt \
  ./project_commands.sh stage2
```

## Useful overrides

```bash
DATA_CONFIG=configs/data/s1k_1_1.yaml ./project_commands.sh prepare
STAGE1_CONFIG=configs/stage1/qwen25_7b_m3.yaml ./project_commands.sh stage1
SIGNALS_CONFIG=configs/signals/main.yaml ./project_commands.sh supervision
STAGE2_CONFIG=configs/stage2/qwen25_7b_output_space.yaml ./project_commands.sh stage2
EVAL_CONFIG=configs/eval/p_align.yaml ./project_commands.sh evaluate
HF_HUB_OFFLINE=1 ./project_commands.sh all
```

Outputs are written under `artifacts/`. Default paths are listed in
`configs/pipeline.yaml`.
