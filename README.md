# GAC-CoT-MTKD

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
./project_commands.sh stage2-medoid
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
   medoid and student output directories.
2. `prepare` downloads the pinned s1K-1.1 dataset and Qwen tokenizer, serializes
   the fixed DeepSeek trajectories, and records token regions and reasoning-step
   boundaries in `artifacts/prepared/s1k_1_1` for both phases.
3. `stage1-stress` checks Phase-1 VRAM on the longest real example and synthetic
   32k examples, including SFT, ramp and full training paths. It uses a temporary
   model and writes `artifacts/stage1/stress_memory.json`.
4. `stage1` trains the three LoRA experts and saves their adapter bundle,
   checkpoint and manifest in `artifacts/stage1/main`.
5. `stage2-medoid` scores the frozen local experts on the prepared corpus and
   selects the adapter to copy into the student. Its manifest is saved in
   `artifacts/stage2_medoid/output_space_local`.
6. `stage2-stress` checks the actual Phase-2 forward, loss, gradients and temporary
   optimizer update at real and synthetic context lengths. The explicit output
   override saves its report as `artifacts/stage2/output_space_local_stress_memory.json`.
7. `stage2` trains the student for three epochs with output-space KD and writes
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
./project_commands.sh stage2-medoid
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
4. `stage2-medoid` selects the student initialization from those imported experts
   and saves `artifacts/stage2_medoid/output_space/manifest.json`.
5. `stage2-stress` checks Phase-2 VRAM and update completion on one H200. Its
   report is `artifacts/stage2/output_space_stress_memory.json`; it does not
   save a trained student.
6. `stage2` runs the three training epochs and exports the student to
   `artifacts/stage2/output_space/final/adapters/student`. The same output root
   contains `checkpoint.pt`, `manifest.json`, `metrics.jsonl` and the detailed
   `reasoning_steps.jsonl` log.
7. `evaluate` optionally benchmarks the exported student. Training is complete
   when `stage2` finishes; evaluation is not needed to export the adapter.

The H200 stress checks are preflights; on another CUDA GPU, use the individual
training commands and a memory check suited to that hardware. Neither route
needs the legacy `supervision` or `cache` commands for this Phase-2 method.

### Shortcuts and tests

With the default Hub-teacher config, `all` runs `prepare`, `fetch-teachers`,
`stage2-medoid`, `stage2`, and `evaluate`. It skips Phase-1 training and both
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

Run preparation/import once, and use `NPROC_PER_NODE` for distributed medoid
selection, training and evaluation. For example, using eight GPUs with the Hub
experts:

```bash
export STAGE2_CONFIG=configs/stage2/qwen25_7b_output_space.yaml
./project_commands.sh prepare
./project_commands.sh fetch-teachers
NPROC_PER_NODE=8 ./project_commands.sh stage2-medoid
NPROC_PER_NODE=8 ./project_commands.sh stage2
NPROC_PER_NODE=8 ./project_commands.sh evaluate
```

The stress commands require one visible H200 rather than a distributed launch.

## Phase 1 forward mode and H200 memory check

Phase 1 uses one forward per expert by default. The existing two-pass replay
remains available with `STAGE1_FORWARD_MODE=two_pass ./project_commands.sh stage1`.
The first 10% of optimizer updates use SFT alone. During the 10–30% ramp,
SFT and DPP use separate transformer VJPs; after 30%, they share one VJP.
The combined path preserves the original global token and sample normalizers.

After preparing the dataset and before full H200 training, run
`./project_commands.sh stage1-stress` with exactly one H200 visible. It executes
six cases: SFT, ramp, and full paths for both the longest prepared real sample
and a synthetic 32,768-token extension of its reasoning. SFT skips probe/DPP/RBF;
ramp retains separate SFT/DPP transformer VJPs and gradient buffers; full uses
the normalized combined task gradient and one transformer VJP. The default
config keeps gradient checkpointing enabled, places probe/support on CUDA, and
uses `lm_head_chunk_tokens: 32768` while retaining configurable head chunking.

Each case runs one warm-up followed by one measured iteration. Adjust these
with `STAGE1_STRESS_WARMUP` and `STAGE1_STRESS_REPETITIONS` (warm-up may be zero;
repetitions must be positive). The report at `artifacts/stage1/stress_memory.json`
records peak allocated/reserved VRAM across warm-up and measurement, plus measured
times and completion counters. Times are for **one sample/microbatch including
an optimizer update**, not a training update accumulating the global batch of
32 samples. These longest-sample timings do not estimate the complete training run.

A worst reserved peak below 110 GiB is marked ready for one-pass; 110–120 GiB
asks for review; 120 GiB or more, or any CUDA OOM, recommends the two-pass fallback
and exits with status 2. All six cases must complete before the preflight is ready.
Stress updates affect only the temporary model in that process; no training
checkpoint or adapters are saved.

## Phase 2: disagreement-adaptive output-space MTKD

The `output-space` branch defaults to disagreement-adaptive distribution
aggregation from the Phase-2 proposal. Phase 1 is unchanged. Three frozen
teachers and one student share a single backbone. Teacher forcing uses fixed
DeepSeek reasoning trajectories; training performs no autoregressive generation.
Only the student LoRA is updated, with dropout disabled.

Teachers default to [duyentl04/abc](https://huggingface.co/duyentl04/abc/tree/52aff0a878826b09fa62fa5a8229a6d49910ff6a),
pinned to commit `52aff0a878826b09fa62fa5a8229a6d49910ff6a`. `fetch-teachers`
downloads the three PEFT adapters (about 242 MB in total) and imports them into
`artifacts/stage1/duyentl04_abc`. It verifies the pinned weight checksums,
base-model revision, LoRA configuration, finite tensors and council structure.
It rebuilds the pipeline's adapter bundle on CPU without downloading the backbone
or a Phase-1 training checkpoint. The Phase-2 commands also import automatically
when this directory is absent; subsequent runs verify and reuse the local files,
including with `HF_HUB_OFFLINE=1`. `all` uses these trained experts and skips
Phase-1 training. `stage1` remains available for training a new local council.

The Hub upload omits `adapter_states.pt` and the original prepared-data manifest,
and its rewritten paths invalidate the original config checksum. The importer
preserves the uploaded files under `source/`, retains the original training-data
fingerprint, and writes fresh checksums for its converted bundle/config.
Exact Phase-1 training-data identity cannot be established from this upload.
Medoid selection validates the current prepared corpus and tokenizer, records
this limitation, and binds subsequent Phase-2 training to that corpus and the
imported teachers. It does not relabel the historical data fingerprint.

To use the locally trained council, select
`configs/stage2/qwen25_7b_output_space_local.yaml` as shown in Run. It uses
`teacher_source.type: local` and `paths.stage1: artifacts/stage1/main`, with
separate medoid/output directories. Keep those separate when changing teacher
sources; old medoid artifacts and checkpoints are rejected.

`stage2-medoid` selects the teacher minimizing the mean full-vocabulary
`KL(uniform council mixture || teacher)` at temperature 2 over all assistant
targets in the complete prepared corpus. It does not compute the old PAG or
static teacher weights. The student copies this adapter.

For each reasoning token, all teachers and the student use the same fixed prefix
and temperature `T`. The default is `T = 2.0`. Here `M = 3` is the teacher count,
`z` denotes full-vocabulary logits, and `S` denotes the student. The teacher
distributions and arithmetic council mean are:

```math
\begin{aligned}
p_{m,t} &= \operatorname{softmax}(z_{m,t}/T), \\
p_{S,t} &= \operatorname{softmax}(z_{S,t}/T), \\
\bar p_t &= \frac{1}{M}\sum_{m=1}^{M}p_{m,t}.
\end{aligned}
```

Every teacher has equal weight. Token disagreement is generalized JSD normalized
by its upper bound `log(M)`. A single power parameter is shared by every token
within reasoning step `s`. Let `T_s` be its reasoning-content token positions and
`n_s = |T_s|`. All logarithms are natural, so raw JSD is measured in nats:

```math
\begin{aligned}
D_t^{\mathrm{JS}}
  &= \frac{1}{M}\sum_{m=1}^{M}\operatorname{KL}(p_{m,t}\Vert\bar p_t), \\
d_s
  &= \frac{1}{n_s}\sum_{t\in T_s}\frac{D_t^{\mathrm{JS}}}{\log M}, \\
\rho_s &= d_s \in [0,1].
\end{aligned}
```

For each token, form the power mean and normalize across the full vocabulary
`V` (152,064 output tokens for the configured model). The zero-power case is
defined separately to avoid division by zero:

```math
\tilde q_{s,t}(v)=
\begin{cases}
\left[\frac{1}{M}\sum_{m=1}^{M}p_{m,t}(v)^{\rho_s}\right]^{1/\rho_s},
  & \rho_s>0, \\
\exp\left(\frac{1}{M}\sum_{m=1}^{M}\log p_{m,t}(v)\right),
  & \rho_s=0.
\end{cases}
```

```math
q_{s,t}(v)=\frac{\tilde q_{s,t}(v)}{\sum_{u\in V}\tilde q_{s,t}(u)}.
```

At `rho = 0`, the continuous limit is the normalized geometric mean,
`q(v) ∝ exp(mean_m log p_m(v))`, equivalent to softmax of mean teacher logits.
At `rho = 1`, it is the arithmetic mean. Intermediate values change the power
mean operator directly; they are not a linear blend of these two endpoints.
JSD, `rho` and the target depend only on the frozen teachers and fixed trajectory.
All are detached during the student update.

The loss direction is `KL(target || student)`. It averages KL over each step's
tokens, then gives each retained step and each example equal weight. `K` is the
number of retained steps in that example, `B` is the effective batch, and `sg`
means stop-gradient:

```math
\begin{aligned}
\ell_s
  &= \frac{T^2}{n_s}\sum_{t\in T_s}
     \operatorname{KL}(\operatorname{sg}(q_{s,t})\Vert p_{S,t}), \\
L_{\mathrm{example}}
  &= \frac{1}{K}\sum_{s=1}^{K}\ell_s, \\
L_B
  &= \frac{1}{|B|}\sum_{i\in B}L_{\mathrm{example},i}.
\end{aligned}
```

The implementation currently uses reasoning content tokens for `T_s`.
Delimiters remain in the teacher-forced prefixes but have no KD target; assistant
control, answer and EOS tokens have no Phase-2 loss. Only complete reasoning
steps fitting the 32,768-token trajectory context are retained. Gold solutions
are not used for a task anchor or an SFT objective. Every retained step
participates in KD, including steps with zero disagreement. An example with no
retained step contributes zero loss and stays in the batch denominator. If a
whole effective batch has no retained steps, AdamW and the LR scheduler are both
skipped while data progress advances.

During student training, teachers run sequentially without gradients; their
selected hidden states stay on the model device by default, avoiding CPU
transfers. Full-vocabulary logits
and targets use head chunks of up to 4096 tokens within each reasoning step,
without Top-K/tail approximation. A step with at most 4096 reasoning tokens
uses one chunk; 4097--8192 uses two. Longer steps use more chunks to bound the
full-vocabulary pooling workspace. This does not rerun the decoder.
Each reasoning step's FP64 teacher log probabilities are retained when they fit
the 8 GiB cache budget and reused for power pooling after computing the step's
JSD. Steps exceeding that budget recompute teacher head projections in chunks;
decoder hidden states are always reused. The budget bounds retained teacher
probabilities only, not total GPU memory or temporary pooling workspace. For
three teachers over 152,064 vocabulary entries, a 4096-token FP64 probability
chunk alone occupies about 13.9 GiB. A step may therefore use one head chunk
while exceeding the 8 GiB cache budget and requiring two teacher-head sweeps.
Power pooling is evaluated in the log domain, including a stable evaluation
near `rho = 0`. Gradient checkpointing remains enabled because long-sequence
student activations dominate memory without it. Each example needs one student
trajectory graph and a final student backward; teacher-gradient and gold-anchor
passes are absent. The GPU probability cache lasts only for the current step;
the teachers are evaluated again whenever the next example/epoch is processed.
There is no dataset-wide target cache, and the legacy mixture cache is unused.

The default training settings are:

| Setting | Value | Purpose |
| --- | --- | --- |
| `stage2.epochs` | `3` | Number of passes through the prepared dataset. |
| `stage2.max_length` | `32768` | Maximum trajectory context, including the prompt. |
| `stage2.micro_batch_size` | `1` | Examples processed at a time on each GPU. |
| `stage2.global_batch_size` | `32` | Effective examples per optimizer update; accumulation is derived from GPU count. |
| `runtime.lm_head_chunk_tokens` | `4096` | Maximum tokens projected together inside a reasoning step. |
| `runtime.teacher_hidden_storage` | `device` | Keep teacher hidden states on the model device; `cpu` enables offload. |
| `runtime.teacher_probability_cache_gib` | `8.0` | Per-step teacher probability cache budget; `0` always recomputes. |
| `model.gradient_checkpointing` | `true` | Recompute student activations during backward to reduce memory. |

The runtime controls affect execution and memory, not the full-vocabulary
objective. Run the H200 stress command with the chosen configuration; speed
and peak VRAM require GPU measurement. Changing the configuration requires a
fresh training run rather than resuming a checkpoint from the old configuration.

Training writes aggregate loss, learning rate, preclip gradient norm, example/step
counts, raw JSD mean (nats), and normalized disagreement/rho to `metrics.jsonl`
in the Phase-2 output directory. The terminal also reports KD, LR, JSD and rho.
By default, `logging.reasoning_steps: true` additionally writes every retained
reasoning step of every sample on every epoch to `reasoning_steps.jsonl`.
Each row identifies `sample_id`, original `step_id`, epoch, minibatch, rank and
the completed data/optimizer-step counters before that minibatch. It includes
the step's reasoning-token count, mean raw JSD in `js_mean` (natural-log nats),
normalized `js_normalized = js_mean / log(M)`, `rho = js_normalized`, the step KD
loss, temperature and teacher count. `step_kd_loss` is the token-mean, T-squared
loss before the equal-step/example averaging; `sample_kd_loss` is the example
mean. Epoch, step ID and minibatch indices start at zero. Token offsets are
zero-based in the prepared trajectory, with an exclusive end.

These rows reuse the existing teacher statistics, independent of
`stage2.log_every_steps`; no additional teacher forward is performed. Context-
discarded steps have no JSD row because they are not evaluated. With multiple
GPUs, each rank writes `reasoning_steps.rank00000.jsonl`, etc., so every rank's
samples are covered without concurrent writes to one file. A fresh run clears
its logs; resume trims records after the saved checkpoint cursor before
appending, including interrupted trailing writes. Final manifests record the
per-rank log filenames and checksums. Set `logging.reasoning_steps: false` to
disable the detailed step log while retaining aggregate metrics.

On one H200, run `stage2-medoid`, then `stage2-stress`, then `stage2`. The stress
command executes the actual full method on the longest eligible real example
and a synthetic example reaching the context limit. It reports component times,
peak VRAM, configuration/source fingerprints and completed optimizer updates
at `artifacts/stage2/output_space_stress_memory.json`. It updates only a temporary model.
Missing CUDA/H200 or incomplete update coverage does not establish readiness.
The single-microbatch measurements do not estimate the whole training run.
`all` uses the new medoid and trainer; the H200-specific preflight is explicit.

The report measures this method only when its cases actually run successfully;
no H200 runtime or memory result is supplied by the implementation change.

The gradient-space variant remains available with
`STAGE2_CONFIG=configs/stage2/qwen25_7b_task_geometry.yaml`. Its objective,
gold-answer anchors and artifact paths remain distinct. Checkpoints bind the
selected method, configuration and source fingerprints, so an output-space run
cannot resume a gradient-space or legacy dual-source checkpoint. Evaluation
defaults to `artifacts/stage2/output_space`; evaluating another variant requires
setting the corresponding `paths.stage2` in an evaluation config.

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
