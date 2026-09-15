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

Phase 1 trains each expert on length-normalized reasoning-step NLL with an
independent Bernoulli dropout mask per step and expert. The final answer has
its own always-on loss block. Each optimizer update also applies an outward
Grassmann force computed from the LoRA A-row and B-column subspaces. Diversity
loss uses token-wise DPP on the dynamic Top-k of the full-vocabulary council
distribution, after removing the ground-truth token. The proposal leaves the
loss weights and dropout probability open; their defaults are in
`configs/stage1/qwen25_7b_m5.yaml`. Its `kneedle` probe settings remain for
Phase-2 signal construction and do not cap Phase-1 council Top-k.

Run each stage separately:

```bash
./project_commands.sh prepare
./project_commands.sh stage1
./project_commands.sh publish-stage1
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

## Publish Stage-1 LoRA experts

Set `HF_REPO_ID` before running Stage 1 to publish its completed PEFT experts
automatically. Authenticate with `HF_TOKEN` or `hf auth login`. The publisher
uploads only each expert's `adapter_config.json` and
`adapter_model.safetensors`, plus a model card, to one Hugging Face model repo.
It verifies the files at the new commit and can run again after a completed
training resume.

```bash
HF_REPO_ID=username/cot-mtkd-experts ./project_commands.sh stage1
HF_REPO_ID=username/cot-mtkd-experts ./project_commands.sh publish-stage1
```

New repositories are private by default. Set `HF_REPO_PRIVATE=false` or use
`--public` with `cot-mtkd-publish-stage1` to create a public repository.
For an existing repository, Hugging Face keeps its current visibility.
Use `STAGE1_CONFIG` or `--stage1-dir` when publishing a non-default run.
