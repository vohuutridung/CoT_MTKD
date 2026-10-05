# Phase 2 — thông số huấn luyện

Method: `council_topk_dynamic_temperature_self_distillation`.

Phase 2 chỉ giữ một adapter LoRA student. Hội đồng 3 expert lấy từ Hub, đóng băng. Công thức bám `output/pdf/phase2.tex` mục 3 và mục 5. Phase 1 không được train lại.

Config: `configs/stage2/qwen25_7b_council.yaml`.

## Nguồn expert

| Mục | Giá trị |
| --- | --- |
| Hub | `duyentl04/abc` |
| Revision (pin) | `52aff0a878826b09fa62fa5a8229a6d49910ff6a` |
| Adapter | `expert_0`, `expert_1`, `expert_2` |
| Base | `Qwen/Qwen2.5-7B-Instruct` |
| LoRA teacher | rank 16, alpha 16, dropout 0.05 |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Phase 1 đã chạy | 3 epoch, 94 optimizer step, global batch 32, lr `5e-5`, AdamW, cosine |
| Sha256 weight | `expert_0` `d3ff1ce81c7c014a37d77074bed25286f43bc90e42a0a9fff289cecfd4282a2b` |
| | `expert_1` `d579de75c7270ccfb5bede6aa8c80937883e4c3879d824d0a2ab81025af2218c` |
| | `expert_2` `be99ab3f0b8c5882b4ab127822ab41a8eada9700631917563921f3f853ba1078` |

Import đã chạy trên máy này vào `artifacts/stage1/duyentl04_abc/`. Ba file safetensors khớp sha256 trong config. `config.yaml` trên Hub trùng khối `config` nhúng trong `manifest.json`. `config_file_sha256` ghi trong manifest không còn khớp file `config.yaml` vì path tuyệt đối đã được viết lại thành path tương đối lúc upload; importer ghi nhận cờ này và vẫn nhận bundle.

Phase 1 của repo Hub là objective cũ (`sft` + DPP + RBF), Kneedle cũ `probe_k: 512`, `max_k: 128`, `min_k: 3`. Đó không phải Phase 1 trong `phase2.tex` (dropout theo step, lực đẩy khoảng cách chiếu, DPP trên `V_k \ {y*}`). Phase 2 dùng đúng công cụ Council Top-k mới của spec, trên các adapter đã train xong.

## Medoid trên adapter Hub

Khoảng cách chiếu, ridge `ε_rel = 1e-4`, `ε_abs = 1e-8`, fp32, trung bình trên 196 module (28 layer × 7 target). Phía `B` bật vì `min ||B||_F = 0.122 > τ_B = 0`. Trần lý thuyết của `d²` là `2r = 32`.

| | expert_0 | expert_1 | expert_2 |
| --- | ---: | ---: | ---: |
| `D_m = Σ_{q≠m} d²` | 57.708 | 57.618 | **57.543** |

Medoid: **`expert_2`**. Student được copy nguyên adapter này (`student_init.pt`), không SVD, không trung bình task-vector.

## Siêu tham số train

Những số không có trong spec là lựa chọn vận hành của config, không phải hệ quả của công thức.

| Nhóm | Khoá | Giá trị | Vai trò |
| --- | --- | --- | --- |
| Vòng lặp | `stage2.epochs` | **1** | Một epoch, đúng yêu cầu |
| | `stage2.learning_rate` | `2e-5` | AdamW |
| | `stage2.micro_batch_size` | 1 | Một mẫu mỗi forward |
| | `stage2.global_batch_size` | 8 | Trung bình gradient theo số mẫu trong cửa sổ |
| | `stage2.max_length` | 32768 | Chuỗi prepared dài hơn mức này thì dừng, không cắt |
| | `stage2.max_grad_norm` | 1.0 | Cắt chuẩn ℓ2 toàn cục, chỉ gradient dữ liệu |
| | `stage2.checkpoint_every_steps` | 10 | |
| | `stage2.log_every_steps` | 1 | |
| Optimizer | AdamW | β `(0.9, 0.999)`, `eps 1e-8`, `weight_decay 0` | |
| Scheduler | cosine | `warmup_ratio 0.10`, `min_lr_ratio 0` | |
| Model | dtype | bfloat16 | |
| | attention | `flash_attention_2` (rơi về SDPA nếu không có) | |
| | `gradient_checkpointing` | true | |
| Student LoRA | rank / alpha / dropout | 16 / 16 / **0.0** | Dropout 0 lúc distill; weight teacher vẫn là bản dropout 0.05 |
| Medoid | `epsilon_rel`, `epsilon_abs`, `tau_b` | `1e-4`, `1e-8`, `0` | Ridge và cổng `B` của spec |
| Council | `k_max` | **256** | `N' = 256`, `p_min = p̄` tại hạng 256. `null` thì `N' = N` |
| | `mask_epsilon` | `1e-12` | `ε` trong `M(v)` |
| | `tau_min`, `tau_max` | 0.8, 1.5 | `τ_min < 1 < τ_max` |
| | `temperature_schedule` | `linear` | Mặc định spec. `sigmoid` và `step` có trong code nhưng không dùng |
| | `js_max` | `p95` | Tuyến tính chia cho phân vị 95 của JS trên token reasoning trong cache |
| | `alpha`, `beta` | 1.0, 0.25 | `β = 0.25 α`, nằm trong `[0.1α, 0.5α]` |
| | `epsilon_m` | `1e-6` | Clip `m` vào `[ε_m, 1−ε_m]` |
| Runtime | `lm_head_chunk_tokens` | 4096 | Chia lm-head; cotangent hidden gộp một lần về LoRA |
| Data | `paths.prepared` | `artifacts/prepared/s1k_1_1_cot_only` | Cùng corpus s1K đã prepare |
| | `paths.stage1` | `artifacts/stage1/duyentl04_abc` | Bundle import từ Hub |
| | cache | `artifacts/teacher_cache/council/` | |
| | output | `artifacts/stage2/council/` | |

`τ`, `α`, `β` không nằm trong fingerprint cache. Đổi chúng không cần build lại cache. Đổi `k_max` thì phải build lại.

## Loss

Trên mỗi mẫu, với `|R|` step reasoning, khối đáp án và khối định dạng (control token, delimiter, answer marker, EOS):

```
L_SFT  = (Σ_i L_i + L_ans + L_fixed) / (|R| + 2)
D_KL   = (1/|T|)  Σ_{k≥2}  Σ_{v∈V_k} Q(v) log(Q(v)/P(v))
L_mass = (1/|T'|) Σ_{y*∈V_k} [ q log(q/m) + (1−q) log((1−q)/(1−m)) ]
L      = L_SFT + α D_KL + β L_mass
```

- `P = Softmax(z | V_k)`, `Q = stopgrad Softmax(z | V_k / τ_eff(v))`.
- `T` là token reasoning có `k ≥ 2`. `T'` là token reasoning có `y* ∈ V_k`, kể cả `k = 1`.
- `q = p̄(V_k)` ở `τ = 1`. `m` là khối lượng student trên `V_k`, tính bằng logsumexp rồi clip.
- `τ_eff(v) = 1 + M(v)(τ−1)` khi `τ ≥ 1`, và `1 + (1−M(v))(τ−1)` khi `τ < 1`.
- Chỉ LoRA student nhận gradient. Base và ba expert đóng băng, council chạy `no_grad`.
- Trong một optimizer step, loss và gradient là trung bình đều theo mẫu (mỗi mẫu đã tự chuẩn hoá theo `|R|`, `|T|`, `|T'|` của chính nó).

## Đối chiếu với `phase2.tex`

Khớp:

- Council Top-k: `p̄` là trung bình softmax toàn từ vựng; Kneedle `x_j = j/N'`, `y` chuẩn hoá min-max, `k = argmax((1−x)−y)`; cửa sổ phẳng thì `k = N'`; không loại `y*`.
- `π` trên `V_k` là softmax của logit đã cắt, không softmax hai lần.
- `JS = H(π̄) − mean_m H(π)` trong `[0, ln M]`.
- `σ²` là trung bình `1/M`, `M(v) = σ²(v) / (max σ² + ε)`.
- Ba lịch nhiệt độ. Config đang chạy tuyến tính với `τ_min = 0.8`, `τ_max = 1.5`, `JS_max` = phân vị 95 của JS trên token reasoning. `js_max: null` vẫn còn nghĩa `ln M`.
- Hai nhánh `τ_eff`. KL tự thân có stop-gradient; gradient theo logit trên `V_k` là `(P−Q)/|T|`.
- Mass loss là KL nhị phân, chỉ token `y* ∈ V_k`, không áp nhiệt độ lên `q`.
- Medoid: `d_X² = r − tr(C̃_pp⁻¹ C_pq C̃_qq⁻¹ C_qp)`, trung bình `(d_A² + d_B²)` trên module, ridge đúng hệ số, phía `B` chỉ khi vượt `τ_B`, `m* = argmin D_m`, student copy nguyên adapter.
- SFT toàn từ vựng, một đơn vị cho mỗi step và cho mỗi khối luôn-on.

Khác biệt cần biết, không sửa vì vẫn đúng công thức hoặc vì Phase 1 đã chốt:

1. **`k_max` không trùng Phase 1 đã upload.** Spec nói nếu bật `K_max` thì hai phase dùng cùng một giá trị. Phase 2 đang để 256. Phase 1 trên Hub dùng Kneedle cũ với `max_k: 128` (và `min_k: 3`, `probe_k: 512`), thuật toán khác hẳn công cụ mới. Đặt 128 chỉ khớp trần cửa sổ, không tái tạo cùng tập `V_k`. Muốn đúng chữ “cùng `K_max`” với bản đã train thì đổi `council.k_max` thành `128` rồi build lại cache.
2. **Chia `τ_eff` trên logit thô.** Đúng phương trình `Softmax(z|V_k / τ_eff(v))`. Vì `τ_eff` khác nhau từng token, phép này không bất biến khi dịch logit. Trên smoke 0.5B, logit đỉnh khoảng +21 và `τ` dính `τ_min = 0.5` (JS rất nhỏ), KL thô khoảng 1.91 nat trong khi cùng công thức sau khi trừ max trên `V_k` chỉ khoảng 0.097; khoảng 76% token có argmax của `Q` khác argmax của `P`. Đây là hệ quả của spec.
3. **Mẫu số SFT.** Trên dữ liệu prepare, cả khối đáp án và khối định dạng đều có token, nên mẫu số là `|R|+2`. Nếu một khối rỗng, code bỏ đơn vị đó thay vì vẫn cộng 1 vào mẫu số.
4. **Chuẩn `B`.** Code lấy min chuẩn Frobenius của từng ma trận `B` (từng module, từng expert). Spec viết `min_m ||B_m||_F`. Với `τ_B = 0` và adapter đã train (`min` đo được 0.122), cả hai cách đều bật phía `B`. Medoid không đổi.

## Lệnh

```bash
export STAGE2_CONFIG=configs/stage2/qwen25_7b_council.yaml
./project_commands.sh prepare
./project_commands.sh fetch-teachers   # đã import xong trong phiên này
./project_commands.sh stage2-cache
./project_commands.sh stage2
```

`stage2-stress` chỉ chạy tiếp khi GPU là H200; trên GPU khác lệnh thoát mã 2 sau khi kiểm tra nguồn dữ liệu.

## Kiểm tra luồng bằng model nhỏ

Qwen2.5-0.5B-Instruct không nhận được LoRA 7B, nên smoke không tải Hub. Luồng thuật toán chạy riêng: 16 mẫu s1K ngắn, 3 expert train local 1 epoch, rồi cache và Phase 2 đúng 1 epoch.

- Cache: 16 mẫu, 18447 token reasoning, 223 step. Medoid `expert_0`. `k` trung bình 7.75, median 6, max 36, không có token `k = 1`. `y* ∉ V_k` = 10.9%. `q` trung bình 0.906.
- Train: 2 update, peak khoảng 2.6 GiB. Update 1: loss 3.811 (SFT 1.993, KL 1.818, mass `2e-5`), `τ` 0.500, `k` 7.46. Update 2: loss 4.126 (SFT 2.231, KL 1.896, mass `8e-5`).
- Ra đủ `final/adapters/student`, `checkpoint.pt`, `metrics.jsonl`, `reasoning_steps.jsonl`, `performance.jsonl`.
- Test: 178 pass. 3 fail có sẵn ở DPP Phase 1 và geometry cũ, không đụng Phase 2.

Import Hub ở trên là phần smoke model nhỏ không thay được: weight thật đã vào `artifacts/stage1/duyentl04_abc/` và medoid thật là `expert_2`.
