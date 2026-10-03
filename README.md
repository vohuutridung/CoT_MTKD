# GAC-CoT-MTKD

Train three LoRA experts and distill them into one student adapter.

## Setup

```bash
./project_commands.sh setup
```

## Run

Run preparation, load the trained Hub experts, then run Phase 2 and evaluation:

```bash
./project_commands.sh all
```

Run setup, tests, and the complete pipeline:

```bash
./project_commands.sh full
```

Run each stage separately:

```bash
./project_commands.sh prepare
./project_commands.sh fetch-teachers
./project_commands.sh stage2-medoid
./project_commands.sh stage2-stress
./project_commands.sh stage2
./project_commands.sh evaluate
```

Run tests:

```bash
./project_commands.sh test
```

## Multi-GPU

```bash
NPROC_PER_NODE=8 ./project_commands.sh all
```

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

To use a locally trained council, copy the Phase-2 YAML and set
`teacher_source.type: local` and `paths.stage1: artifacts/stage1/main`, then pass
that file through `STAGE2_CONFIG`. Use fresh medoid/output directories when
changing teacher sources; old medoid artifacts and checkpoints are rejected.

`stage2-medoid` selects the teacher minimizing the mean full-vocabulary
`KL(uniform council mixture || teacher)` at temperature 2 over all assistant
targets in the complete prepared corpus. It does not compute the old PAG or
static teacher weights. The student copies this adapter.

For each reasoning token, all teachers and the student use the same fixed prefix
and temperature `T`. The default is `T: 2.0`; the proposal leaves this as a positive
hyperparameter. With `M` teachers, the arithmetic council mean is

$$
p_{m,t}=\operatorname{softmax}(z_{m,t}/T),\qquad
\bar p_t=\frac{1}{M}\sum_{m=1}^{M}p_{m,t}.
$$

Every teacher has equal weight. Token disagreement is generalized JSD normalized
by its upper bound `log(M)`. A single power parameter is shared by every token
within reasoning step `s`:

$$
D_t^{\mathrm{JS}}=\frac{1}{M}\sum_{m=1}^{M}
\operatorname{KL}(p_{m,t}\Vert\bar p_t),\qquad
\rho_s=d_s=\frac{1}{n_s}\sum_{t\in T_s}\frac{D_t^{\mathrm{JS}}}{\log M}.
$$

The target is the normalized power mean over the full vocabulary:

$$
\tilde q_{s,t}(v)=\left[\frac{1}{M}\sum_{m=1}^{M}
p_{m,t}(v)^{\rho_s}\right]^{1/\rho_s},\qquad
q_{s,t}(v)=\frac{\tilde q_{s,t}(v)}{\sum_{v'}\tilde q_{s,t}(v')}.
$$

At `rho = 0`, the continuous limit is the normalized geometric mean,
`q(v) ∝ exp(mean_m log p_m(v))`, equivalent to softmax of mean teacher logits.
At `rho = 1`, it is the arithmetic mean. Intermediate values change the power
mean operator directly; they are not a linear blend of these two endpoints.
JSD, `rho` and the target depend only on the frozen teachers and fixed trajectory.
All are detached during the student update.

The loss averages KL over each step's tokens, then gives each retained step and
each example equal weight:

$$
\ell_s=\frac{T^2}{n_s}\sum_{t\in T_s}
\operatorname{KL}(\operatorname{sg}(q_{s,t})\Vert p_{S,t}),\qquad
L_{\mathrm{example}}=\frac{1}{K}\sum_{s=1}^{K}\ell_s,\qquad
L_B=\frac{1}{|B|}\sum_{(x,R)\in B}L_{\mathrm{example}}.
$$

The implementation currently uses reasoning content tokens for `T_s`.
Delimiters remain in the teacher-forced prefixes but have no KD target; assistant
control, answer and EOS tokens have no Phase-2 loss. Only complete reasoning
steps fitting the 32,768-token trajectory context are retained. Gold solutions
are not used for a task anchor or an SFT objective. Every retained step
participates in KD, including steps with zero disagreement. An example with no
retained step contributes zero loss and stays in the batch denominator. If a
whole effective batch has no retained steps, AdamW and the LR scheduler are both
skipped while data progress advances.

Teachers run sequentially without gradients; their selected hidden states are
held on CPU. Full-vocabulary logits and targets use head chunks of 64 tokens,
without Top-K/tail approximation. Power pooling is evaluated in the log domain,
including a stable evaluation near `rho = 0`. Gradient checkpointing remains
enabled. Each example needs one student trajectory graph and a final student
backward; teacher-gradient and gold-anchor passes are absent. Teacher targets
could be cached because they are independent of the student, but this branch
currently evaluates them online and does not use the legacy mixture cache.

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
STAGE2_RESUME=artifacts/stage2/output_space/checkpoint.pt ./project_commands.sh stage2
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
