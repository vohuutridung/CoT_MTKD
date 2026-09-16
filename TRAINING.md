# Cấu hình huấn luyện (7B)

Nguồn: `configs/stage1/qwen25_7b_m5.yaml`, `configs/stage2/qwen25_7b_top512_tail.yaml`, `configs/data/s1k_1_1.yaml`.

Model: `Qwen/Qwen2.5-7B-Instruct` · seed `42` · dtype `bfloat16`.

---

## 1. Cấu hình kiến trúc LoRA

| Tham số | Giá trị |
|---|---|
| LoRA Rank (r) | 16 |
| LoRA Alpha (α) | 16 |
| LoRA Dropout | 5.0% |
| Target Modules | `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]` |
| Bias | none |

LoRA này dùng chung cho 5 expert Stage 1 và student Stage 2.

---

## 2. Thông số huấn luyện (Training setup)

| Tham số | Stage 1 | Stage 2 |
|---|---|---|
| Số Epoch | 3 | 3 |
| Micro batch size | 1 | 1 |
| Effective Batch Size (`global_batch_size`) | **8** | **8** |
| Gradient accumulation (world_size=1) | 8 | 8 |
| Optimizer updates / run | ceil(1000 × 3 / 8) = 375 | 375 |
| Max Sequence Length | 32,768 | 32,768 |
| Learning Rate (LR) | **5.00e-05** | **2.00e-05** |
| Max grad norm | 1.0 | 1.0 |
| Số expert | 5 | 1 (init từ medoid) |
| Step dropout | 0.20 | — |
| Diversity weight (DPP) | 0.2 | — |
| Repulsion weight (Grassmann) | 1.0 | — |
| Hard loss weight | — | 1.0 |
| KD loss weight | — | 1.0 |

Effective batch = `micro_batch_size × world_size × accumulation` = 1 × 1 × 8 = **8**.

---

## 3. Bộ tối ưu hóa & lập lịch (Optimizer & Scheduler)

| Tham số | Giá trị |
|---|---|
| Optimizer | AdamW |
| Optimizer Betas | (0.9, 0.999) |
| Eps | 1e-8 |
| Weight Decay | 0.0 |
| LR Scheduler | Cosine with warmup |
| Warmup | `warmup_ratio = 0.1` |
| Implementation | `torch.optim.lr_scheduler.LambdaLR` |
| min_lr_ratio | 0.0 |
