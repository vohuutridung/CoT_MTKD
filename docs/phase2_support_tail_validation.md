# Phase 2 support + tail: báo cáo triển khai và kiểm tra

Ngày 04/10/2026, branch `output-space`. Đã sửa trực tiếp working tree; chưa commit/push. Phase 1 không đổi. Theo yêu cầu cập nhật sau đó, cả hai config output-space dùng **1 epoch**, microbatch **1**, global batch **8**. Smoke test 3 epochs bên dưới vẫn là bằng chứng cache được reuse qua nhiều epochs.

## 1. File thay đổi

### Thêm (6 files)

- `docs/phase2_support_tail_validation.md`
- `scripts/35_build_stage2_cache.sh`
- `src/cot_mtkd/cli/build_stage2_cache.py`
- `src/cot_mtkd/stage2/council_cache.py`
- `src/cot_mtkd/stage2/initialization.py`
- `tests/test_council_cache.py`

### Sửa (27 files)

- `README.md`
- `configs/pipeline.yaml`
- `configs/signals/main.yaml`
- `configs/stage2/qwen25_7b_output_space.yaml`
- `configs/stage2/qwen25_7b_output_space_local.yaml`
- `configs/stage2/qwen25_7b_task_geometry.yaml`
- `project_commands.sh`
- `pyproject.toml`
- `src/cot_mtkd/cli/build_supervision.py`
- `src/cot_mtkd/cli/stress_stage2_memory.py`
- `src/cot_mtkd/signals/__init__.py`
- `src/cot_mtkd/signals/builder.py`
- `src/cot_mtkd/signals/predictive.py`
- `src/cot_mtkd/stage2/__init__.py`
- `src/cot_mtkd/stage2/online.py`
- `src/cot_mtkd/stage2/output_space.py`
- `src/cot_mtkd/stage2/output_space_losses.py`
- `src/cot_mtkd/stage2/step_logging.py`
- `src/cot_mtkd/stage2/teachers.py`
- `src/cot_mtkd/stage2/trainer.py`
- `tests/test_output_space_losses.py`
- `tests/test_output_space_online.py`
- `tests/test_signals.py`
- `tests/test_stage2_step_logging.py`
- `tests/test_stage2_stress.py`
- `tests/test_stage2_teachers.py`
- `tests/test_stage2_updates.py`

### Xóa (5 files)

- `scripts/35_build_stage2_medoid.sh`
- `src/cot_mtkd/cli/build_stage2_medoid.py`
- `src/cot_mtkd/signals/medoid.py`
- `src/cot_mtkd/stage2/medoid.py`
- `tests/test_stage2_medoid.py`

## 2. Formulation thực tế

Với reasoning-content token được giữ lại, `full_vocab_probe` loại gold trước top-K. Phase 2 gọi trực tiếp `stage1.kneedle.local_k_from_probe` và `build_union_support`, không có bộ selection khác:

\[
K=\min(512,|V|-1),\quad k=\min(K,\max(1+\arg\max_j[(1-x_j)-u_j],8)),
\quad V_t=\bigcup_m S_{m,t}\cup\{y_t\}.
\]

Support được canonicalize ascending/unique, gold đúng một lần, selection hoàn toàn no-grad. JSD tính trên `softmax(z_m[V_t]/1)`; đây là restriction+renormalization của full softmax do log-partition triệt tiêu. Không có tail trong JSD.

\[
D_s^{JS}=\operatorname{Mean}_{t\in s}JS(\pi_{1,t},\pi_{2,t},\pi_{3,t}),
\qquad\rho_s=\operatorname{clamp}(D_s^{JS}/\log3,0,1).
\]

KD dùng full-softmax mass ở T=2 trên support, kèm một bucket đại diện toàn bộ phần còn lại:

\[
r_m(v)=\exp(z_{m,v}/2-\log Z_m),\quad
r_m(\bot)=1-\sum_{v\in V_t}r_m(v),\quad
q_t=\operatorname{Normalize}\operatorname{PowerMean}_{\rho_s}(r_1,r_2,r_3).
\]

Power mean đúng công thức generalized mean: rho=0 geometric, rho=1 arithmetic; không linear blend, không sửa calibration. Target/JSD/rho được detach, student xây reduced distribution bằng cùng helper.

\[
L=\operatorname{Mean}_{sample}\operatorname{Mean}_{step}\operatorname{Mean}_{token}
\left[4KL(\operatorname{sg}(q)\Vert r_S)+0.25\,CE(z_S,y)\right].
\]

KD/SFT/init dùng chung reasoning-content mask của `output_space.plan_record`: chỉ complete retained `TokenRegion.REASONING` steps. Delimiter/control/EOS không có loss; vẫn là context. Không tạo gold-solution suffix. Mọi step có cùng trọng số trong sample; mọi sample có cùng trọng số trong batch. Sample rỗng đóng góp 0 và nằm trong mẫu số; batch toàn sample rỗng skip AdamW/scheduler.

Init là argmin corpus SFT loss, temperature 1, token→step→sample weighting. Tie chọn expert đứng trước trong `adapter_names`. Copy toàn bộ winning LoRA, không merge/SVD/barycenter/KD score.

## 3. Defaults

| Field | Giá trị |
|---|---:|
| `aggregation.js_temperature` | 1.0 |
| `aggregation.kd_temperature` | 2.0 |
| `aggregation.sft_weight` | 0.25, hỗ trợ 0.5 |
| `aggregation.search_k` | 512 |
| `aggregation.k_min` | 8 |
| `aggregation.teacher_execution` | precomputed_support_tail |
| `stage2.global_batch_size` | 8 |
| `stage2.micro_batch_size` | 1 |
| `stage2.gradient_accumulation_steps` | null, derive 8/(world_size × micro) |
| `stage2.epochs` | 1, theo yêu cầu cập nhật |
| `stage2.max_length` | 32768 |
| `runtime.lm_head_chunk_tokens` | 4096 |
| `runtime.preprocessing_hidden_storage` | cpu |

Remote cache root: `artifacts/teacher_cache/output_space`. Local: `artifacts/teacher_cache/output_space_local`. Mỗi cache nằm trong subdirectory mang fingerprint.

Global batch không chia hết world×micro hoặc explicit accumulation khác giá trị derive sẽ fail rõ ràng. Đã kiểm tra world sizes 1/2/4/8, cùng các case không hợp lệ 0/3/16/32 và microbatch fractional. Dataset sampler giữ hợp đồng không padding: record count cũng phải chia hết world_size; với 996 samples, defaults chạy được world_size 1/2/4; world size 8 bị từ chối vì dataset không chia hết, còn 16 không chia hết global batch.

Runner hiện duyệt tuần tự từng record trong loader microbatch, chưa có batched decoder forward/backward. Chỉ tăng microbatch config không tạo GPU parallelism. Defaults 1 epoch/global batch 8/microbatch 1 trên một GPU cho 125 updates; update cuối có 4 samples. Đổi epoch/batch không đổi fingerprint cache.

## 4. Logic cũ đã bỏ

- Xóa toàn bộ Stage-2 functional-medoid module, CLI, script và tests chuyên cho cơ chế đó.
- Xóa KL-to-uniform-barycenter selection trong legacy `signals/predictive.py`/`builder.py`, cùng `signals/medoid.py` và `temperature_medoid`.
- Xóa `aggregation.temperature`, `online_full_vocab`, `paths.medoid`, `teacher_probability_cache_gib` và old online adaptive-KD gradient khỏi output-space configs/path.
- `stage2-medoid` được thay bằng `stage2-cache`; `all` gọi cache trước training.

Legacy task-geometry vẫn là method riêng, explicit config riêng, giữ objective teacher-gradient/answer-anchor của nó. Khởi tạo của variant đó cũng dùng best-SFT cache artifact. Online full-vocabulary geometry losses và legacy PAG/Top-512 tooling không nằm trong output-space default. Không có functional medoid còn chạy trong repo.

## 5. Cache format/fingerprint và logging

Version 1, một safetensors/sample, index JSON SHA-256; manifest chỉ được publish khi đủ samples. Payload:

| Tensor | Dtype/semantics |
|---|---|
| support_ids | int32, ragged ascending/unique |
| support_offsets | int64, N+1 offsets |
| support_log_target | FP32 log(q) trên support, không padding persist |
| tail_log_target | FP32 log(q_tail), một/token |
| token_positions | int32 observed target positions |
| step_offsets | int64 mapping token→step |
| step_js / step_rho | FP64, một/step |
| raw_k / selected_k | int16, 3×N |
| union_sizes | int16, N |
| gold_present_before_add | bool, N |

`best_expert.pt` chỉ chứa winning adapter, SHA-256 riêng. Output-space training tạo đúng adapter `student`; không load ba teacher adapters. Cache lưu final target, training không gọi JSD/power mean/teacher forward.

Fingerprint binds teacher manifest/bundle/checkpoint checksums, backbone name/revision/dtype/attention implementation, prepared manifest/data/tokenizer, max_length, search_k/k_min, JS/KD temperatures, support/mask/storage semantics, head chunk size, PyTorch version và source-file SHA-256 cho Kneedle/planner/preprocess/loss/model/token contract. Thay identity chọn thư mục cache mới; training cache miss yêu cầu chạy preprocessing. Explicit fingerprint mismatch/corrupted file fail. Epoch/batch/optimizer/resume/SFT weight không nằm trong identity của target; full training config vẫn nằm trong checkpoint run fingerprint.

Cache hits kiểm tra content trước khi reuse; không tải teacher model. Startup training xác minh mọi sample checksum. File sample/index/adapter/manifest được ghi atomically. Một preprocessing pass mỗi corpus; không có teacher pass riêng cho init.

Diagnostics manifest chứa support/union quantiles, raw/selected K histograms theo expert, gold-present rate, raw JS/rho histograms/quantiles, teacher-target tail và numerical counters. Training aggregate/step logs chứa KD, SFT, weighted SFT, total loss, hai temperatures, target/student tail, reduced target entropy, cache/init provenance. Performance logs ghi zero teacher-head sweeps/recomputed steps, cache-hit steps, timing/memory. Resume giữ recovery/trim rules và checksum logs.

## 6. Ước tính cache trên dataset hiện tại

Đã verify data SHA-256 `1a2253faf2917cbca9d8a68dad7cfa1cde3d3d0512e7ede6edeaaf77cba6b35f` và đếm đúng mask mới: **996 samples, 9,037,847 tokens, 233,134 steps**, không discarded/empty sample.

Supervision payload xấp xỉ `8 × N × mean_support + 31 × N + 24 × steps + 16 × samples` bytes, cộng safetensors headers/index/manifest và winning adapter (~77 MiB).

| Mean support giả định | Supervision GiB |
|---:|---:|
| 9 | 0.872 |
| 25 | 1.950 |
| 50 | 3.633 |
| 100 | 7.000 |
| 200 | 13.734 |
| 512 | 34.743 |
| 1537 | 103.763 |

Đây là storage scenarios, không phải support histogram/cache size của 7B đã đo. Bound 1537 là union 3×512+gold, không phải kỳ vọng thực nghiệm.

## 7. Tests đã chạy

- Full suite: **172 passed, 6 CUDA skips, 74 subtests passed**.
- Ruff toàn `src` và `tests`: pass.
- Shell syntax và CLI help: pass. `git diff --check`: pass.
- Reference dense độc lập kiểm tra loss, gradient LoRA, step JSD, SFT score init; teacher/council frozen và một final combined student VJP.
- Support/gold/dedup/variable lengths; hai temperatures; JS identical/bounded/step mean; tail gần 0/lớn/full support và finite gradients; power endpoints/tiny rho/extreme probabilities; KL-zero/T²; KD-only λ=0; λ=.25/hierarchy; best expert/ties/full clone; cache roundtrip/hash/code/config invalidation/epoch independence; importer, actual update/export/resume; BF16 hidden cotangents; empty samples và accumulation boundaries.

## 8. Dry-run/smoke thực tế

Fixture CPU tiny Qwen, vocabulary 64, 3 teacher LoRAs, 2 samples × 2 steps, **3 epochs**. Smoke dùng global batch 2 để có một update/epoch; defaults thật hiện là 1 epoch/global batch 8.

- Preprocessing teacher forwards: **6**; training teacher forwards: **0**; student forwards: **6**, 3 updates.
- Support sample đầu: `[15, 15, 15, 15, 9]`, gold đúng một lần, ascending/unique; variable lengths hoạt động.
- JS T=1, KD T=2; raw step JS: `[8.989434392630583e-05, 2.7445187598575694e-05]` nats, rho: `[8.182535809360755e-05, 2.4981686334356007e-05]`.
- Reduced target sum max error: **0.0**.
- Expert SFT scores: `[4.198086222012838, 4.193742851416269, 4.197570860385895]`; chọn **expert_1**; full adapter copy exact.
- Update cuối: KD=0.000026842024, SFT=4.193596839905, weighted SFT=1.048399209976, total=1.048426052000; identity error **0.0**.
- Cache supervision thực tế của fixture: 3462 bytes; 12 step log rows cho epochs 0/1/2.
- Cache hit không load teacher, fingerprint mismatch fail; đổi epoch/SFT weight vẫn reuse target. Không có tail clamp/roundoff anomaly trong fixture.

Raw evidence: `artifacts/audits/phase2_support_tail_smoke/report.json`, cùng cache/student logs trong directory đó; estimate evidence: `artifacts/audits/phase2_support_tail_cache_estimate.json`. Đây là fixtures local, không phải trained 7B student.

## 9. Numerical/performance giới hạn còn lại

- Chưa chạy council preprocessing toàn corpus 7B, H200 stress, multi-GPU execution hoặc đánh giá chất lượng. Chưa có real support/JSD/rho histogram mới hay real expert ranking.
- Log-target FP32 giữ được log probability rất nhỏ; các probability cực nhỏ vẫn có thể underflow khi exp trong FP32, nhưng không gây NaN trong tests.
- Tail near-saturation dùng outside logsumexp; empty complement floor 1e-30 được count. Đây là numerical safeguard, không thêm target smoothing tổng quát.
- Exact partition function và hard CE vẫn cần full-vocabulary logits theo chunk. Chunk 4096 riêng FP32 logits khoảng 2.32 GiB với vocab 152064; peak còn gồm gradient/CE/workspace/activations. Chưa xác nhận peak VRAM hay tốc độ trên H200.
- Preprocessing tái dùng teacher decoder hidden nhưng cần hai head sweeps; teacher reduced values giữ CPU chỉ trong step. Nhiều step nhỏ có overhead head calls. Startup SHA checking và per-sample cache I/O có chi phí chưa benchmark.
- Cache ragged tiết kiệm disk; runtime pad một sample trong CPU RAM. Bound cache khoảng 103.76 GiB nếu mọi support đều cực đại; cần đo histogram thật trước chọn storage/compression khác.
- Preprocessing multi-rank cần shared cache filesystem; chạy một preprocessing job cho mỗi fingerprint. CPU/tiny tests không xác nhận distributed GPU hiệu năng.

Lệnh pipeline mới: `fetch-teachers` → `stage2-cache` → `stage2-stress` (H200) → `stage2`.
