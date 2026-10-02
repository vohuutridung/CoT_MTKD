# Cấu hình huấn luyện (7B)

Nguồn: `configs/stage1/qwen25_7b_m3.yaml`, `configs/stage2/qwen25_7b.yaml`, `configs/data/s1k_1_1.yaml`.

Model: `Qwen/Qwen2.5-7B-Instruct` · seed `42` · dtype `bfloat16`.

---

## 1. Cấu hình kiến trúc LoRA

| Tham số | Giá trị |
|---|---|
| LoRA Rank (r) | 16 |
| LoRA Alpha (α) | 16 |
| LoRA Dropout | 0.05 |
| use_rslora / use_dora / use_qalora | false / false / false |
| Target Modules | `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]` |
| Bias | none |

LoRA này dùng chung cho 3 expert Stage 1 và student Stage 2.

---

## 2. Thông số huấn luyện (Training setup)

| Tham số | Stage 1 | Stage 2 |
|---|---|---|
| Số Epoch | 3 | 3 |
| Scheduler horizon | 3 epoch | 3 epoch |
| Benchmark checkpoint | epoch 2 (khác `final/`) | epoch 3 (bước cuối) |
| Micro batch size | 1 | 1 |
| Effective Batch Size (`global_batch_size`) | **32** | **32** |
| Gradient accumulation (world_size=1) | 32 | 32 |
| Optimizer updates / run | ceil(1000 × 3 / 32) = 94 | 94 |
| Max Sequence Length | 32,768 | 32,768 |
| Learning Rate (LR) | **5.00e-05** | **5.00e-05** |
| Max grad norm | 1.0 | 1.0 |
| Số expert | 3 | 1 (init từ merge) |
| Step dropout | 0.20 | — |
| Diversity weight (DPP) | 0.2 | — |
| Repulsion weight (Grassmann) | 1.0 | — |
| Merge method | — | `ta` (ablate: ties, dare_ties, tsv, iso_c) |
| $\lambda_U$ / $\lambda_D$ | — | 0.5 / 0.5 |

Effective batch = `micro_batch_size × world_size × accumulation` = 1 × 1 × 32 = **32**. Cấu hình này gắn với một process: `gradient_accumulation_steps` được ghi tường minh là 32, nên `world_size` phải là 1.

Cosine schedule trải đủ 3 epoch (94 bước). Warmup là `round(0.1 × 94) = 9` bước. Cổng \(B\) bật hẳn tại \(t_B=\lceil 0.10\times 94\rceil=10\): không ramp, không \(\tau_B\), không chặn tương đối. Snapshot `benchmark/` lấy ở epoch 2 (bước \(\lceil 2000/32\rceil=63\)), khác checkpoint `final/` ở bước 94.

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
