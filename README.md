# GAC-CoT-MTKD

Train three LoRA experts and distill them into one student adapter.

## Setup

```bash
./project_commands.sh setup
```

## Run

Run the complete pipeline:

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
./project_commands.sh stage1-stress
./project_commands.sh stage1
./project_commands.sh supervision
./project_commands.sh cache
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

## Resume training

```bash
STAGE1_RESUME=artifacts/stage1/main/checkpoint.pt ./project_commands.sh stage1
STAGE2_RESUME=artifacts/stage2/main/checkpoint.pt ./project_commands.sh stage2
```

## Useful overrides

```bash
DATA_CONFIG=configs/data/s1k_1_1.yaml ./project_commands.sh prepare
STAGE1_CONFIG=configs/stage1/qwen25_7b_m3.yaml ./project_commands.sh stage1
SIGNALS_CONFIG=configs/signals/main.yaml ./project_commands.sh supervision
STAGE2_CONFIG=configs/stage2/qwen25_7b_top512_tail.yaml ./project_commands.sh stage2
EVAL_CONFIG=configs/eval/p_align.yaml ./project_commands.sh evaluate
HF_HUB_OFFLINE=1 ./project_commands.sh all
```

Outputs are written under `artifacts/`. Default paths are listed in
`configs/pipeline.yaml`.
