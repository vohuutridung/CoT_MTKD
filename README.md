./project_commands.sh setup
# GAC-CoT-MTKD

Train five LoRA experts and distill them into one student adapter.

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

## Resume training

```bash
STAGE1_RESUME=artifacts/stage1/main/checkpoint.pt ./project_commands.sh stage1
STAGE2_RESUME=artifacts/stage2/main/checkpoint.pt ./project_commands.sh stage2
```

## Useful overrides

```bash
DATA_CONFIG=configs/data/s1k_1_1.yaml ./project_commands.sh prepare
STAGE1_CONFIG=configs/stage1/qwen25_7b_m5.yaml ./project_commands.sh stage1
SIGNALS_CONFIG=configs/signals/main.yaml ./project_commands.sh supervision
STAGE2_CONFIG=configs/stage2/qwen25_7b_top512_tail.yaml ./project_commands.sh stage2
EVAL_CONFIG=configs/eval/p_align.yaml ./project_commands.sh evaluate
HF_HUB_OFFLINE=1 ./project_commands.sh all
```

Outputs are written under `artifacts/`. Default paths are listed in
`configs/pipeline.yaml`.
