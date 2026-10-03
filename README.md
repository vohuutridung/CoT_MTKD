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

## Phase 2: task-anchored gradient geometry MTKD

The default Phase 2 follows the latest task-anchored proposal. Three frozen
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

For each example and each retained reasoning step, Phase 2 computes every
teacher's KD gradient over all student LoRA parameters, its raw council mean
and agreement, and the student's gold-answer CE gradient after that step.
The anchor uses the entire raw `solution` field, preceded by the existing answer
marker; only solution tokens contribute to its CE. Its gradient includes the
complete reasoning prefix. KD uses reasoning content tokens only; delimiters
remain in prefixes. The full gold continuation is reserved within 32,768 tokens;
only complete trailing reasoning steps may be discarded. A sample with no
retained reasoning steps contributes zero loss. If prompt plus full gold alone
does not fit, training fails with the sample ID rather than truncating gold.

Positive anchor cosines determine teacher weights and the step's absolute
utility weight. The target blends the teachers' weighted arithmetic mixture
with their normalized geometric mixture, gated by raw agreement and common
direction utility. All selectors and targets are detached. The final objective
contains KD only: mean over each step's tokens, weighted sum divided by the
example's fixed retained step count, then mean over examples. Gold-answer CE
is an anchor diagnostic and is never added to the update loss.

Zero-signal examples remain in the batch denominator. If an entire effective
batch has zero useful steps, AdamW and the LR scheduler are both skipped; data
progress still advances. Checkpoints store separate successful-update and data
batch counters. The new method cannot resume a legacy dual-source checkpoint.

Teachers run sequentially without gradients; their temporary hidden states
are held on CPU. Full-vocabulary logits and targets are streamed in blocks of
64 tokens, without Top-K/tail approximation. The old mixture cache is not used.
Gradient checkpointing remains enabled. A student trajectory graph and one
gold-anchor graph coexist; per-step LoRA gradient buffers are released before
the next step. Exact geometry is expensive: an example with K steps requires
3K teacher-KD parameter VJPs, K anchor VJPs and a final student KD VJP.

On one H200, run `stage2-medoid`, then `stage2-stress`, then `stage2`. The stress
command executes the actual full method on the longest eligible real example
and a synthetic example reaching the context limit. It reports component times,
peak VRAM, configuration/source fingerprints and completed optimizer updates
at `artifacts/stage2/stress_memory.json`. It updates only a temporary model.
Missing CUDA/H200 or incomplete update coverage does not establish readiness.
The single-microbatch measurements do not estimate the whole training run.
`all` uses the new medoid and trainer; the H200-specific preflight is explicit.

Positive teacher utilities describe local raw gradient-descent directions.
Geometric pooling and AdamW do not guarantee that the final update improves
the gold-answer loss. The implementation follows the proposal without adding
an extra target-selection guard.

## Resume training

```bash
STAGE1_RESUME=artifacts/stage1/main/checkpoint.pt ./project_commands.sh stage1
STAGE2_RESUME=artifacts/stage2/task_geometry/checkpoint.pt ./project_commands.sh stage2
```

## Useful overrides

```bash
DATA_CONFIG=configs/data/s1k_1_1.yaml ./project_commands.sh prepare
STAGE1_CONFIG=configs/stage1/qwen25_7b_m3.yaml ./project_commands.sh stage1
SIGNALS_CONFIG=configs/signals/main.yaml ./project_commands.sh supervision
STAGE2_CONFIG=configs/stage2/qwen25_7b_task_geometry.yaml ./project_commands.sh stage2
EVAL_CONFIG=configs/eval/p_align.yaml ./project_commands.sh evaluate
HF_HUB_OFFLINE=1 ./project_commands.sh all
```

Outputs are written under `artifacts/`. Default paths are listed in
`configs/pipeline.yaml`.
