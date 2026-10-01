#!/usr/bin/env bash
# Stage-1 flow check on Qwen2.5-0.5B-Instruct.
# LoRA, optimizer, scheduler, learning rate and grad clip come from the 7B
# Stage-1 config. Only the model, attention backend, subset size and step
# count change, so the run finishes on one small GPU.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

SMOKE_MODEL="${SMOKE_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
SMOKE_REVISION="${SMOKE_REVISION:-7ae557604adf67be50417f59c2c2f167def9a775}"
SMOKE_RECORDS="${SMOKE_RECORDS:-4}"
SMOKE_EPOCHS="${SMOKE_EPOCHS:-1}"
SMOKE_GLOBAL_BATCH="${SMOKE_GLOBAL_BATCH:-2}"
SMOKE_ROOT="${SMOKE_ROOT:-artifacts/smoke_stage1_0p5b}"
DATA_CONFIG_PATH="${DATA_CONFIG:-configs/data/s1k_1_1.yaml}"
STAGE1_CONFIG_PATH="${STAGE1_CONFIG:-configs/stage1/qwen25_7b_m3.yaml}"

rm -rf "$SMOKE_ROOT"
mkdir -p "$SMOKE_ROOT" logs

echo "== [1/3] Export $SMOKE_RECORDS shortest s1K-1.1 records"
"$PYTHON_BIN" - "$DATA_CONFIG_PATH" "$SMOKE_ROOT/source.jsonl" "$SMOKE_RECORDS" <<'EOF'
import json
import sys

from datasets import load_dataset

from cot_mtkd.config import load_config

config_path, output_path, count = sys.argv[1], sys.argv[2], int(sys.argv[3])
dataset_config = load_config(config_path)["dataset"]
records = load_dataset(
    dataset_config["name"],
    split=dataset_config.get("split", "train"),
    revision=dataset_config.get("revision"),
)
order = sorted(
    range(len(records)),
    key=lambda index: len(records[index]["deepseek_thinking_trajectory"]),
)[:count]
if len(order) != count:
    raise SystemExit(f"Need {count} records, dataset has {len(records)}")
with open(output_path, "w", encoding="utf-8") as handle:
    for index in sorted(order):
        row = dict(records[index])
        row.setdefault("id", f"s1k-{index:04d}")
        handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
print(f"wrote {len(order)} records")
EOF

echo "== [2/3] Prepare subset with $SMOKE_MODEL"
"$PYTHON_BIN" -m cot_mtkd.cli.prepare_data \
  --config "$DATA_CONFIG_PATH" \
  --set "dataset.local_json=$SMOKE_ROOT/source.jsonl" \
  --set "dataset.expected_records=$SMOKE_RECORDS" \
  --set "model.name_or_path=$SMOKE_MODEL" \
  --set "model.revision=\"$SMOKE_REVISION\"" \
  --set "output_dir=$SMOKE_ROOT/prepared"

echo "== [3/3] Stage 1, keeping the 7B LoRA and optimizer settings"
run_distributed cot_mtkd.cli.train_stage1 \
  --config "$STAGE1_CONFIG_PATH" \
  --set "model.name_or_path=$SMOKE_MODEL" \
  --set "model.revision=\"$SMOKE_REVISION\"" \
  --set "model.attn_implementation=sdpa" \
  --set "stage1.epochs=$SMOKE_EPOCHS" \
  --set "stage1.global_batch_size=$SMOKE_GLOBAL_BATCH" \
  --set "stage1.gradient_accumulation_steps=$SMOKE_GLOBAL_BATCH" \
  --set "stage1.benchmark_checkpoint_epoch=$SMOKE_EPOCHS" \
  --set "stage1.checkpoint_every_steps=1" \
  --set "paths.prepared=$SMOKE_ROOT/prepared" \
  --set "paths.output=$SMOKE_ROOT/stage1"

"$PYTHON_BIN" - "$SMOKE_ROOT/stage1" <<'EOF'
import json
import math
import sys
from pathlib import Path

import yaml

stage1_dir = Path(sys.argv[1])
manifest = json.loads((stage1_dir / "manifest.json").read_text())
config = yaml.safe_load((stage1_dir / "config.yaml").read_text())
steps = [
    json.loads(line)
    for line in (stage1_dir / "metrics.jsonl").read_text().splitlines()
    if line.strip()
]
steps = [row for row in steps if row.get("event") == "stage1_step"]
lora, stage1 = config["lora"], config["stage1"]
print(
    f"ran {len(steps)} optimizer steps | lora r={lora['rank']} alpha={lora['alpha']} "
    f"dropout={lora['dropout']} rslora={lora['use_rslora']} dora={lora['use_dora']} "
    f"qalora={lora['use_qalora']} | lr={stage1['learning_rate']} clip={stage1['max_grad_norm']} "
    f"| optimizer={config['optimizer']['name']} wd={config['optimizer']['weight_decay']} "
    f"| scheduler={config['scheduler']['name']} warmup={config['scheduler']['warmup_ratio']}"
)
for row in steps:
    if not math.isfinite(float(row["sft_nll"])) or not math.isfinite(float(row["dpp_loss"])):
        raise SystemExit(f"step {row['step']} has a non-finite loss")
    print(
        f"step {row['step']} sft_nll={row['sft_nll']:.4f} "
        f"dpp_loss={row['dpp_loss']:.4f} lr={row['learning_rate']:.3e}"
    )
if not steps or manifest["global_step"] != len(steps):
    raise SystemExit("Stage 1 did not log a completed optimizer step")
print("SMOKE TEST PASSED")
EOF
