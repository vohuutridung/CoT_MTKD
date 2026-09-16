# GAC-CoT-MTKD

Train five LoRA experts, merge them into one student adapter, then distill
with council-weighted step NLL.

## Setup

```bash
./project_commands.sh setup
```

## Run

Complete pipeline (`prepare` → `stage1` → `supervision` → `stage2` → `evaluate`):

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
NPROC_PER_NODE=8 ./project_commands.sh all
```

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
STAGE1_CONFIG=configs/stage1/qwen25_7b_m5.yaml ./project_commands.sh stage1
SIGNALS_CONFIG=configs/signals/main.yaml ./project_commands.sh supervision
STAGE2_CONFIG=configs/stage2/qwen25_7b.yaml ./project_commands.sh stage2
CACHE_CONFIG=configs/cache/qwen25_7b_top512_tail.yaml ./project_commands.sh cache
EVAL_CONFIG=configs/eval/p_align.yaml ./project_commands.sh evaluate
HF_HUB_OFFLINE=1 ./project_commands.sh all
```

Defaults live in `configs/pipeline.yaml`. Artifacts go under `artifacts/`.
Method notes: `docs/stage1.md`, `docs/stage2.md`.

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
