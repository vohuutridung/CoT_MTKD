# Phase 2: power pooling và calibration từ training corpus

Đã sửa trực tiếp trên `phase1-gac`, baseline
`3e5a1f7f3b8c7c7082b272401c5ad9e824121904`. Không commit/push.
Hai cấu hình chính là `qwen25_7b_output_space.yaml` (HF teachers) và
`qwen25_7b_output_space_local.yaml` (teachers train tại máy).

## Các file thay đổi

| File | Thay đổi |
|---|---|
| `src/cot_mtkd/stage2/disagreement.py` (mới) | Power pooling ổn định số; fit quantile corpus; saturation rho; kiểm tra calibration. |
| `src/cot_mtkd/stage2/council_cache.py` | Cache v2; lưu tạm teacher support/tail; fit tau một lần; hoàn thiện q trên CPU; fingerprint và thống kê corpus. |
| `src/cot_mtkd/stage2/output_space.py` | Dùng cached rho mới; T thống nhất; kiểm tra calibration; log Delta, mean token JS, coverage và từng thành phần loss. |
| `src/cot_mtkd/stage2/trainer.py` | Batch đủ 16; checkpoint/resume kiểm tra tau; log runtime; xuất diagnostics.json và metadata. |
| `src/cot_mtkd/stage2/diagnostics.py` (mới) | Tổng hợp quantile/threshold từ reasoning-step logs, không chạy model thêm. |
| `src/cot_mtkd/cli/stress_stage2_memory.py` | Stress dùng temperature mới và tau đã fit; không fit từ mẫu synthetic. |
| `configs/stage2/qwen25_7b_output_space.yaml` | Defaults HF route: T=1, r=4, quantile=.75, SFT=.01, epoch=1, batch=16, drop incomplete window. |
| `configs/stage2/qwen25_7b_output_space_local.yaml` | Cùng defaults cho local-teacher route. |
| `configs/stage2/qwen25_7b_task_geometry.yaml` | Chỉ chuyển block aggregation dùng chung cache sang schema mới; objective/optimizer geometry lịch sử giữ nguyên. |
| `tests/test_stage2_disagreement.py` (mới) | JS reference, r=1/r=4, saturation, corpus calibration, missing tau, old cache và checkpoint rejection. |
| `tests/test_council_cache.py` | Cache roundtrip/checksum/fingerprint mới; defaults; accumulation và batch boundary. |
| `tests/test_output_space_online.py` | Dense reference tính JS/pooling/quantile/rho độc lập và so sánh loss/LoRA gradient. |
| `tests/test_stage2_updates.py` | Integration cache→train→export→resume; metadata/calibration/diagnostics; zero teacher forwards. |
| `tests/test_stage2_step_logging.py` | Schema reasoning-step log mới và resume trimming. |
| `tests/test_stage2_teachers.py` | Chuyển fixture aggregation sang schema calibration mới. |
| `README.md` | Defaults, quy trình rebuild cache, phân biệt cache manifest với student manifest, vị trí logs/diagnostics. |
| `docs/phase2_calibrated_validation.md` (mới) | Báo cáo thay đổi và bằng chứng validation này. |

## Công thức thực tế

Giữ nguyên token JS và support của implementation hiện hành: JS tính trên
teacher posterior được restriction rồi renormalization trên explicit local
Kneedle-union + gold; JS không có tail bucket. Đây là semantics có sẵn của
nhánh, không chuyển sang full-vocabulary JS trong patch này.

Với phân phối local đó là \(p_{m,t}^{local}\),

\[
D_t=\frac13\sum_{m=1}^3 KL(p_{m,t}^{local}\|\bar p_t^{local}),\quad
\Delta_s=\left(\frac1{n_s}\sum_{t\in s}D_t^r\right)^{1/r},\quad
\tau=Q_{q_\tau}(\{\Delta_s\}_{s\in\mathcal D_{train}}),\quad
\rho_s=\frac{\Delta_s}{\Delta_s+\tau}.
\]

`D_t`, `token_js_mean`, `Delta` và `tau` dùng nats; rho không chia log(3).
Helper JS cũ vẫn trả D/log(3); cache nhân lại log(3) để lấy D trong nats
trước pooling. Giá trị rho dùng FP64; chỉ chặn bằng số biểu diễn ngay dưới 1
nếu phép cộng mất tau do rounding tại tỷ lệ cực lớn. Quantile dùng linear
interpolation, không weighting theo độ dài step. Tau bằng 0 hoặc không hợp lệ
gây lỗi; không tự đặt floor hoặc dùng evaluation để hiệu chỉnh.

Cho KD, \(p_{m,t}^{red}\) là full-softmax mass tại explicit support và một
complement tail bucket, ở cùng nhiệt độ T. Trên các category này:

\[
\tilde q_{s,t}(c)=
\begin{cases}
\left(\frac13\sum_m p_{m,t}^{red}(c)^{\rho_s}\right)^{1/\rho_s},&\rho_s>0,\\
\exp\left(\frac13\sum_m\log p_{m,t}^{red}(c)\right),&\rho_s=0,
\end{cases}
\qquad q_{s,t}=\tilde q_{s,t}/\sum_c\tilde q_{s,t}(c).
\]

Giữ implementation power mean ổn định và endpoints geometric/arithmetic.
Student phân phối reduced cùng support/tail và T. Loss một sample:

\[
L=\frac1{S}\sum_s\frac1{n_s}\sum_{t\in s}
\left[T^2 KL(q_{s,t}\|p_{student,t}^{red})+0.01\,CE(z_{student,t},y_t)\right].
\]

T mặc định bằng 1 nên multiplier KD bằng 1. SFT vẫn dùng logits không temperature
và không disagreement weighting. Segmentation và token→step→sample loss averaging
không đổi. Giữ search_k=512, k_min=8, không k_max, union ba expert + gold + tail;
không mở rộng support theo cumulative mass.

## Runtime defaults và cache

- `epochs=1`, `micro_batch_size=1`, `global_batch_size=16`.
- `aggregation.temperature=1.0`, `disagreement_pooling_power=4.0`,
  `tau_quantile=0.75`, `sft_weight=0.01`; đều configurable.
- Accumulation = 16/(microbatch × world_size), phải nguyên. Đã test world size
  1/2/4/8/16; với microbatch 1 lần lượt accumulation 16/8/4/2/1.
- LR=2e-5, AdamW betas=(.9,.999), eps=1e-8, weight_decay=0;
  cosine warmup=.10, min_lr_ratio=0, max_grad_norm=1 giữ nguyên.
- `incomplete_batch_policy: drop` giữ effective batch đúng 16. Với 996 mẫu:
  62 windows × 16 = 992 mẫu shuffle được optimize, bỏ 4 mẫu cuối; toàn bộ 996
  mẫu vẫn được tính calibration và chọn expert khởi tạo. Không padding/repeat.
- Precompute forward một lần/expert/sample. Sau khi toàn corpus hoàn thành,
  rank 0 fit tau; các rank hoàn thiện targets trên CPU rồi xóa teacher arrays
  tạm. Cache cuối chỉ có supervision và metadata; student train không gọi teacher.
- Version-1 cache và checkpoint cũ không tương thích. Rebuild `stage2-cache`,
  bắt đầu student run mới. Fingerprint gồm temperature, pooling method/power,
  tau quantile, SFT weight, Kneedle, data/teachers/source checksums. Epoch/batch/
  optimizer không làm thay đổi q cache. Checkpoint chứa calibration và kiểm tra
  cả run fingerprint lẫn fitted tau khi resume.

## Validation đã chạy

`PYTHONPATH=src:tests python -m pytest -q`: **193 passed, 6 skipped,
111 subtests passed**. Môi trường CPU: Python 3.13, torch 2.14.1,
transformers 4.57.6, peft 0.21.2. Scoped Ruff E9/F/I, format check,
compileall, git diff --check và shell syntax đều qua.

Tests bao gồm independent dense loss/gradient reference, JS KL reference,
r=1/r=4, Delta=tau→rho=.5, zero disagreement, rho<1, zero-quantile rejection,
global shared tau, config/fingerprint/version/checksum incompatibility,
missing/different calibration checkpoint rejection, batch boundaries và resume.

Dry run CPU dùng tiny Qwen vocabulary 64 và ba LoRA teacher fixtures:
36 training samples, 72 reasoning steps, 180 reasoning tokens để fit tau.
Một epoch optimize 32 mẫu, 64 steps qua 2 windows có effective batch 16;
4 mẫu cuối không optimize. Teacher forwards khi precompute=108; lúc train=0;
student forwards=32. Teacher fixtures được tạo trực tiếp, không chạy/sửa Phase 1.
Best-expert full LoRA clone chính xác; support ascending/unique và gold đúng một
lần; target normalization sai số tối đa 5.96e-8. Loss identity sai số 6.94e-18.

Các số sau là **fixture validation**, không phải thống kê chạy 7B/corpus thật.
Chưa train/evaluate 7B với formulation mới; không kết luận cải thiện benchmark
hoặc coverage. Không lấy log cũ để suy đoán distribution Delta mới.

Tau fit từ toàn bộ 72 steps = **0.00013987730180372877**.

| Thống kê | Delta (nats) | rho |
|---|---:|---:|
| Mean | 0.0001300313342 | 0.4621693475 |
| Median | 0.0001311687945 | 0.4839335861 |
| p75 | 0.0001398773018 | 0.5000000000 |
| p90 | 0.0001785706987 | 0.5607197572 |
| p95 | 0.0002679312453 | 0.6569365658 |
| p99 | 0.0003482816035 | 0.7134594898 |
| Max | 0.0003482816035 | 0.7134594898 |

Fraction rho > .25 = 100%; > .50 = 23.6111% (17/72); > .75 = 0%.
Không buộc đúng 25% vượt midpoint khi quantile có ties.

| Thống kê trên 180 token, T=1 | Giá trị |
|---|---:|
| Mean / median support size (gồm gold) | 14.8722222 / 15 |
| Mean target support mass | 0.2474576530 |
| Mean target tail mass | 0.7525423470 |
| Median target tail mass | 0.7502599955 |
| p75 target tail mass | 0.7677784264 |
| p90 target tail mass | 0.7983579040 |
| p95 target tail mass | 0.7996138483 |
| p99 target tail mass | 0.8171111035 |
| Fraction target tail > .10 / .25 / .50 | 100% / 100% / 100% |

Observed student support/tail means qua 64 training steps =
0.2449803550 / 0.7550196406. Đây là mean của step means, khác token-weighted
cache statistics ở bảng trên. Student tail p75/p90/p95/p99 lần lượt
0.7690956593 / 0.7752842903 / 0.7762678832 / 0.7825535160.

Raw report + script tái chạy nằm ở
`artifacts/audits/phase2_calibrated_smoke_20261005/{report.json,run_smoke.py}`.
`report.json` chứa toàn bộ quantiles, histogram, threshold fractions, losses và
student coverage. Chạy script với `PYTHONPATH=src:tests`; script dùng fixtures
từ tests và không tải HF weights. Artifacts này được Git ignore.

## Logs và phạm vi

Student output chứa `diagnostics.json`: calibration, corpus token/step target
statistics và observed student step statistics. Cache manifest dưới
`<teacher_cache_dir>/<fingerprint>/manifest.json` chứa calibration/diagnostics;
`index.json` trỏ tới token-level support IDs, per-expert K và targets trong
sample safetensors. Student manifest trỏ rõ tới diagnostics và cache fingerprint.
`reasoning_steps.jsonl` có token_js_mean, Delta, r, tau, quantile, rho, support
size, target/student support/tail mass, raw KD/SFT, weighted SFT và total loss.

Kiểm tra byte-for-byte với HEAD: toàn bộ 18 tracked files trong `stage1/`,
`configs/stage1/`, shared `models/`, `data/` và `output_space_losses.py` không đổi.
Không sửa GAC/DPP/RBF, Phase-1 Kneedle, LoRA config, segmentation hoặc teacher
count. Các route Phase-2 geometry/top512 lịch sử không phải route revised chính.

Chi tiết cấu trúc cần nêu: corpus calibration buộc tách teacher preprocessing
và target finalization, có thêm staging support/tail của ba teacher trên disk
và CPU pass; không lưu full-vocabulary probabilities. Exact global batch yêu
cầu bỏ incomplete window như mô tả trên. Không thêm gradient norms riêng KD/SFT
vì sẽ thêm backward/VJP; đây là mục secondary trong prompt. Diagnostics student
dùng per-step means và chỉ có khi reasoning_steps logging bật (default true).
