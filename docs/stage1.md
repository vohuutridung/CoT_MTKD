# Stage 1 — Train Five GAC-CoT LoRA Experts

Run:

```bash
./project_commands.sh stage1
```

Entry point:

```text
scripts/20_train_stage1.sh
    -> cot_mtkd.cli.train_stage1
    -> cot_mtkd.stage1.trainer.train_stage1
```

Default config:

```text
configs/stage1/qwen25_7b_m5.yaml
```

Output:

```text
artifacts/stage1/main/
├── manifest.json
├── checkpoint.pt
├── final/
│   └── adapters/
└── metrics.jsonl
```

---

## 1. Mục tiêu

Stage 1 huấn luyện **$M = 5$ LoRA experts** độc lập trên cùng base model:

```text
Qwen/Qwen2.5-7B-Instruct
```

với mục tiêu:

1. **Good** — mỗi expert fit tốt chuỗi CoT (reasoning + answer) qua NLL có chuẩn hóa theo bước.
2. **Diverse** — phân bố token của các expert khác nhau trên support động bằng DPP.
3. **Separated in parameter space** — các không gian con LoRA bị đẩy ra xa nhau bằng Grassmann repulsion.

Base model đóng băng; chỉ train LoRA adapters.

Mỗi expert có optimizer AdamW riêng.

---

## 2. Pipeline tổng quát

Một optimizer step:

```text
┌─────────────────────────────────────────────────────────────┐
│ 1. Probe (no-grad forward tất cả experts)                   │
│    → council Top-k (Kneedle) → DPP loss + dL_DPP/dlogits   │
│                                                             │
│ 2. Replay từng expert (train forward + VJP)                 │
│    → grad SFT + grad DPP trên LoRA params                   │
│                                                             │
│ 3. Accumulate → all-reduce → chuẩn hóa                      │
│                                                             │
│ 4. g_data = grad SFT + lambda_div × grad DPP               │
│                                                             │
│ 5. Clip ||g_data|| ≤ max_grad_norm                          │
│                                                             │
│ 6. AdamW.step(g_data)                                       │
│                                                             │
│ 7. theta ← theta + eta × lambda_rep × F_rep                │
│    (Grassmann force, ngoài optimizer)                       │
│                                                             │
│ 8. Cosine LR scheduler.step()                               │
└─────────────────────────────────────────────────────────────┘
```

Đây là cập nhật Phase-1 theo Eq. (23) trong proposal:

* **data gradient** đi qua AdamW;
* **Grassmann force** được cộng trực tiếp vào tham số sau bước optimizer.

---

## 3. Ký hiệu

| Ký hiệu                  | Ý nghĩa                                    |
| ------------------------ | ------------------------------------------ |
| $M$                      | Số experts (`num_experts = 5`)             |
| $m \in {1,\ldots,M}$     | Chỉ số expert                              |
| $i$                      | Sample trong batch                         |
| $t$                      | Vị trí token (response / reasoning)        |
| $s$                      | Reasoning step id trong sample             |
| $y_t$                    | Ground-truth token tại $t$                 |
| $\ell_m(t,\cdot)$        | Logits vocabulary của expert $m$ tại $t$   |
| $p_m(\cdot \mid x_{<t})$ | Softmax của expert $m$                     |
| $\theta_m$               | Tham số LoRA của expert $m$                |
| $\eta$                   | Learning rate hiện tại (sau warmup/cosine) |
| $\lambda_{\mathrm{div}}$ | `diversity_weight = 0.2`                   |
| $\lambda_{\mathrm{rep}}$ | `repulsion_weight = 1.0`                   |
| $p_{\mathrm{drop}}$      | `step_drop_probability = 0.20`             |

---

## 4. Loss tổng hợp và quy tắc cập nhật

### 4.1 Objective "data"

Loss đưa vào gradient buffer của expert $m$:

$$
\mathcal{L}_{\mathrm{data}}^{(m)}
=
\mathcal{L}_{\mathrm{SFT}}^{(m)}
+
\lambda_{\mathrm{div}}
\mathcal{L}_{\mathrm{DPP}}
$$

Gradient dùng cho AdamW:

$$
g_m
=
\nabla_{\theta_m}
\mathcal{L}_{\mathrm{SFT}}^{(m)}
+
\lambda_{\mathrm{div}}
\nabla_{\theta_m}
\mathcal{L}_{\mathrm{DPP}}
$$

Sau all-reduce / accumulation, $g_m$ bị clip theo global $\ell_2$ norm:

$$
g_m
\leftarrow
\operatorname{clip}
\left(
g_m,\,
\texttt{max\_grad\_norm}=1.0
\right)
$$

Lưu ý: $\mathcal{L}_{\mathrm{DPP}}$ phụ thuộc **cùng lúc** vào logits của mọi expert (council support cố định từ probe), nhưng VJP được tách theo từng expert khi replay.

### 4.2 Cập nhật tham số (Eq. 23)

$$
\theta_m
\leftarrow
\operatorname{AdamW}
\left(
\theta_m,\,
g_m
\right)
+
\eta
\lambda_{\mathrm{rep}}
F_{\mathrm{rep}}^{(m)}
$$

với $F_{\mathrm{rep}}^{(m)}$ là Grassmann outward force (mục 7).

Force **không** đi qua AdamW state.

---

# 5. Task loss — step-normalized NLL + step dropout

## 5.1 Phân vùng token

Mỗi response token thuộc một trong ba nhóm:

* **REASONING** — gắn `step_id = s >= 0`.
* **ANSWER** — khối đáp án cuối, luôn giữ loss.
* **FORMAT / control** — delimiter, answer marker, EOS, v.v.; khối luôn-on riêng, không pha loãng answer NLL.

---

## 5.2 Bernoulli step dropout

Với mỗi cặp $(i,s)$ — sample và reasoning step — và mỗi expert $m$:

$$
M_{m,i,s}
\sim
\operatorname{Bernoulli}
\left(
1-p_{\mathrm{drop}}
\right)
$$

với:

$$
p_{\mathrm{drop}} = 0.20
$$

Mask được sinh từ seed xác định:

```text
sha256(
    base_seed,
    "stage1_step_dropout",
    global_step,
    rng_stream,
    expert
)
```

Nếu:

$$
M_{m,i,s}=0
$$

thì toàn bộ token của step đó có trọng số bằng $0$ và không đóng góp vào loss.

Answer và format blocks **không** bị dropout.

---

## 5.3 Trọng số chuẩn hóa theo độ dài đoạn

Với một segment $S$ có $L = |S|$ token còn lại:

$$
w_t = \frac{1}{L}
$$

cho mọi $t \in S$.

Các segment luôn-on như answer và format cũng được chuẩn hóa nội bộ tương tự: mỗi segment đóng góp tổng trọng số bằng $1$.

Đếm chuẩn hóa $N_{\mathrm{seg}}$ là số segment **đóng góp**:

$$
N_{\mathrm{seg}}
=
\text{số reasoning steps được giữ lại}
+
\text{số fixed blocks}
$$

Step bị drop có trọng số $0$ và **không** được tính trong $N_{\mathrm{seg}}$.

---

## 5.4 Công thức NLL

Cross-entropy từng token:

$$
\ell_{\mathrm{CE}}(t;m)
=
-\log
p_m
\left(
y_t \mid x_{<t}
\right)
$$

Tương đương:

$$
\ell_{\mathrm{CE}}(t;m)
=
-\ell_m(t,y_t)
+
\log
\sum_v
\exp
\left(
\ell_m(t,v)
\right)
$$

Loss SFT trước khi chia cho $N_{\mathrm{seg}}$:

$$
\widetilde{\mathcal{L}}_{\mathrm{SFT}}^{(m)}
=
\sum_{t \in \mathrm{response}}
w_t^{(m)}
\ell_{\mathrm{CE}}(t;m)
$$

Sau accumulation / all-reduce, chia cho số segment đóng góp của chính expert $m$:

$$
\mathcal{L}_{\mathrm{SFT}}^{(m)}
=
\frac{
\widetilde{\mathcal{L}}_{\mathrm{SFT}}^{(m)}
}{
N_{\mathrm{seg}}^{(m)}
}
$$

trong đó $N_{\mathrm{seg}}^{(m)}$ không gồm reasoning step đã drop. Metric `sft_nll` là trung bình của đại lượng này trên các expert.

### Ý nghĩa

Mỗi reasoning step được giữ lại và mỗi khối answer/format có "khối lượng" ngang nhau bất kể số token.

Step dropout làm mỗi expert thấy một tập reasoning steps khác nhau, từ đó tạo ra đa dạng hóa quỹ đạo học, mà không pha loãng loss bằng cách để step đã drop trong mẫu số.

---

# 6. Diversity loss — token-wise DPP trên council Top-k

## 6.1 Council distribution

Tại mỗi reasoning token $t$, lấy softmax đầy đủ của mọi expert và tính trung bình:

$$
\bar p_t(v)
=
\frac{1}{M}
\sum_{m=1}^{M}
p_m
\left(
v \mid x_{<t}
\right)
$$

---

## 6.2 Chọn support bằng full-vocabulary Kneedle

Sắp xếp $\bar p_t$ giảm dần:

$$
\bar p_{(1)}
\ge
\bar p_{(2)}
\ge
\cdots
\ge
\bar p_{(V)}
$$

trong đó $V$ là vocabulary size.

Chuẩn hóa min-max theo rank:

$$
\hat y_r
=
\frac{
\bar p_{(r)}
-
\bar p_{(V)}
}{
\bar p_{(1)}
-
\bar p_{(V)}
+
\varepsilon
}
$$

và:

$$
x_r
=
\frac{r}{V}
$$

Elbow Top-$k$:

$$
k_t
=
\operatorname*{argmax}_{r \in \{1,\ldots,V\}}
\left(
(1-x_r)-\hat y_r
\right)
+
1
$$

Support sau khi bỏ ground-truth:

$$
\mathcal{S}_t
=
\left\{
v_{(1)},
\ldots,
v_{(k_t)}
\right\}
\setminus
\{y_t\}
$$

Candidate IDs được **detach**, do đó không backprop qua:

* lựa chọn $k_t$;
* membership của support.

> `kneedle.probe_k`, `min_k`, `max_k` **không giới hạn council Top-k của Stage 1**. Các config này dành cho Phase-2 signal construction.

---

## 6.3 Feature DPP trên support

Gọi $z^{(m)}_t$ là logits đầy đủ của expert $m$ tại token $t$. DPP **không** softmax lại trên support:

$$
\hat p^{(m)}_t = \operatorname{softmax}(z^{(m)}_t),
\qquad
\hat p^{(m)}_{t,\mathcal{S}_t}
=
\left[\hat p^{(m)}_t\right]_{\mathcal{S}_t}
$$

tức $\log \hat p_c = z_c - \operatorname{logsumexp}(z_{\mathrm{full}})$. Các vị trí padding của mask là $0$ sau khi lấy xác suất.

L2-normalization:

$$
z_{m,t}
=
\frac{
\hat p^{(m)}_{t,\mathcal{S}_t}
}{
\|\hat p^{(m)}_{t,\mathcal{S}_t}\|_2
}
$$

Ma trận feature:

$$
Z_t
\in
\mathbb{R}^{M \times |\mathcal{S}_t|}
$$

với row thứ $m$ là:

$$
(Z_t)_{m,:}
=
z_{m,t}^{\top}
$$

Gram matrix:

$$
G_t
=
Z_t Z_t^\top
$$

Do đó:

$$
G_t
\in
\mathbb{R}^{M \times M}
$$

---

## 6.4 Token loss — negative log-determinant

Với jitter $\varepsilon$:

$$
\mathcal{L}_t^{\mathrm{DPP}}
=
-\log
\det
\left(
G_t+\varepsilon I
\right)
$$

Implementation sử dụng Cholesky:

```text
L = Cholesky(G + epsilon I)
logdet = 2 * sum(log(diag(L)))
loss = -logdet
```

Jitter ban đầu:

```text
dpp.jitter = 1e-5
```

Nếu Cholesky fail:

1. tăng jitter lên $10\times$;
2. retry cho tới `max_jitter = 1e-2`;
3. nếu vẫn fail, fallback sang `slogdet`.

### Ý nghĩa

$\log\det(G_t)$ lớn khi các vector $z_{m,t}$ gần trực giao và span lớn.

Do đó:

* $\log\det(G_t)$ lớn → experts đa dạng → DPP loss nhỏ;
* $\log\det(G_t)$ nhỏ → experts giống nhau → DPP loss lớn.

---

## 6.5 Reduction theo step → sample

### Step level

Trong cùng reasoning step $(i,s)$:

$$
\mathcal{L}_{i,s}^{\mathrm{DPP}}
=
\frac{1}{
|T_{i,s}|
}
\sum_{t \in T_{i,s}}
\mathcal{L}_t^{\mathrm{DPP}}
$$

trong đó $T_{i,s}$ là tập token thuộc step $s$.

### Sample level

Trong sample $i$:

$$
\mathcal{L}_i^{\mathrm{DPP}}
=
\frac{1}{S_i}
\sum_{s=1}^{S_i}
\mathcal{L}_{i,s}^{\mathrm{DPP}}
$$

### Batch / accumulation

Probe dùng:

```text
reduction="sum"
```

Sau all-reduce, chia cho số samples DPP:

$$
\mathcal{L}_{\mathrm{DPP}}
=
\frac{1}{N_{\mathrm{samples}}}
\sum_i
\mathcal{L}_i^{\mathrm{DPP}}
$$

---

## 6.6 Gradient path

Probe tính

$$
\frac{
\partial
\mathcal{L}_{\mathrm{DPP}}
}{
\partial
\log \hat p_m[\mathcal{S}]
}
$$

với $\log \hat p_c = z_c - \operatorname{logsumexp}(z_{\mathrm{full}})$.

Candidate IDs và support membership được detach.

Replay expert $m$ sử dụng:

```text
support_logprob_vjp_hidden_gradient
```

để đưa cotangent đó về hidden representation (VJP đi qua softmax đầy đủ), sau đó autograd xuống LoRA.

Điều này tương đương:

$$
\nabla_{\theta_m}
\mathcal{L}_{\mathrm{DPP}}
$$

với support được giữ cố định.

---

# 7. Grassmann Repulsion

## 7.1 Cơ sở không gian con LoRA

Với mỗi module LoRA:

$$
A
\in
\mathbb{R}^{r \times d_{\mathrm{in}}}
$$

và:

$$
B
\in
\mathbb{R}^{d_{\mathrm{out}} \times r}
$$

Basis row-space của $A$:

$$
Q_A
=
\operatorname{QR}
\left(
A^\top
\right)_{\mathrm{reduced}}
$$

Basis column-space của $B$:

$$
Q_B
=
\operatorname{QR}
\left(
B
\right)_{\mathrm{reduced}}
$$

Nếu singular value nhỏ nhất nhỏ hơn `rank_epsilon`, thêm scaffold $\varepsilon I$ để QR khả vi.

---

## 7.2 Khoảng cách Grassmann bình phương

Với hai basis $Q$ và $Q'$:

$$
\sigma_r
=
\sigma_r
\left(
Q^\top Q'
\right)
$$

là singular values của $Q^\top Q'$.

Principal angles:

$$
\theta_r
=
\arccos
\left(
\operatorname{clamp}
\left(
\sigma_r,\,
0,\,
1-\delta
\right)
\right)
$$

trong đó:

$$
\delta
=
\texttt{angle\_epsilon}
$$

Khoảng cách một module:

$$
d_{\mathrm{mod}}^2(m,j)
=
\sum_r
\theta_r(A)^2
+
\sum_r
\theta_r(B)^2
$$

Khoảng cách giữa hai experts được lấy trung bình trên các modules:

$$
d^2(m,j)
=
\frac{1}{
|\mathrm{modules}|
}
\sum_{\mathrm{mod}}
d_{\mathrm{mod}}^2(m,j)
$$

Khoảng cách này bất biến với phép đổi basis khả nghịch của LoRA tương ứng với cùng $\Delta W = BA$.

---

## 7.3 RBF kernel trên Grassmann

Xét tất cả các cặp:

$$
m < j
$$

Tính median khoảng cách:

$$
d_{\mathrm{median}}
=
\operatorname{median}_{m<j}
\sqrt{
d^2(m,j)
}
$$

Bandwidth:

$$
h
=
\max
\left(
\frac{
d_{\mathrm{median}}^2
}{
\log M
},
\texttt{bandwidth\_floor}
\right)
$$

RBF kernel:

$$
K_{mj}
=
\exp
\left(
-\frac{
d^2(m,j)
}{
h
}
\right)
$$

**Lưu ý:** $d^2(m,j)$ được detach khi tính kernel.

---

## 7.4 Outward force

Potential của expert $m$:

$$
\Phi_m
=
\frac{1}{M}
\sum_{j \neq m}
K_{mj}
d^2(m,j)
$$

Grassmann outward force:

$$
F_{\mathrm{rep}}^{(m)}
=
\nabla_{\theta_m}
\Phi_m
$$

Force được cộng theo hướng làm tăng khoảng cách giữa các experts.

Trong code, potential chia cho $M$ và bao gồm đầy đủ $M-1$ hàng xóm.

---

# 8. Tóm tắt công thức một dòng

### SFT

$$
\mathcal{L}_{\mathrm{SFT}}^{(m)}
=
\frac{1}{N_{\mathrm{seg}}^{(m)}}
\sum_t
w_t^{(m)}
\left(
-\log p_m(y_t)
\right)
$$

với $N_{\mathrm{seg}}^{(m)}$ chỉ đếm reasoning step được giữ lại cộng khối answer/format luôn-on.

### DPP

$$
\mathcal{L}_{\mathrm{DPP}}
=
\mathbb{E}_i
\mathbb{E}_{s\mid i}
\mathbb{E}_{t\mid s}
\left[
-\log
\det
\left(
G_t+\varepsilon I
\right)
\right]
$$

### Data gradient

$$
g_m
=
\nabla_{\theta_m}
\mathcal{L}_{\mathrm{SFT}}^{(m)}
+
\lambda_{\mathrm{div}}
\nabla_{\theta_m}
\mathcal{L}_{\mathrm{DPP}}
$$

### Parameter update

$$
\theta_m
\leftarrow
\operatorname{AdamW}
\left(
\theta_m,\,
g_m
\right)
+
\eta
\lambda_{\mathrm{rep}}
\nabla_{\theta_m}
\Phi_m
$$

với:

$$
G_t
=
Z_tZ_t^\top
$$

và $Z_t$ là L2-normalized $\operatorname{softmax}(z_{\mathrm{full}})[\mathcal{S}_t]$ trên council Top-$k_t$ sau khi loại $y_t$.

Potential:

$$
\Phi_m
=
\frac{1}{M}
\sum_{j\neq m}
K_{mj}
d^2(m,j)
$$

---

# 9. Hyperparameters mặc định

Từ:

```text
configs/stage1/qwen25_7b_m5.yaml
```

| Nhóm                    | Giá trị                                   |
| ----------------------- | ----------------------------------------- |
| Model                   | `Qwen/Qwen2.5-7B-Instruct` (revision pin) |
| Precision               | `bf16`                                    |
| Attention               | FlashAttention-2                          |
| LoRA rank               | $16$                                      |
| LoRA alpha              | $16$                                      |
| LoRA dropout            | $0.05$                                    |
| LoRA targets            | `q,k,v,o,gate,up,down`                    |
| Experts                 | $M=5$                                     |
| Epochs                  | $3$                                       |
| Learning rate           | $5\times10^{-5}$                          |
| Micro batch             | $1$                                       |
| Global batch            | $32$                                      |
| Step dropout            | $p_{\mathrm{drop}}=0.20$                  |
| Diversity weight        | $\lambda_{\mathrm{div}}=0.2$              |
| Repulsion weight        | $\lambda_{\mathrm{rep}}=1.0$              |
| Max grad norm           | $1.0$                                     |
| DPP jitter              | $10^{-5}$                                 |
| Max DPP jitter          | $10^{-2}$                                 |
| Grassmann rank epsilon  | $10^{-3}$                                 |
| Grassmann angle epsilon | $10^{-6}$                                 |
| Bandwidth floor         | $10^{-8}$                                 |
| Optimizer               | AdamW                                     |
| AdamW $\beta$           | $(0.9,0.999)$                             |
| Weight decay            | $0$                                       |
| AdamW epsilon           | $10^{-8}$                                 |
| Scheduler               | cosine                                    |
| Warmup ratio            | $0.10$                                    |
| Minimum LR ratio        | $0$                                       |

---

## 9.1 Canonical optimizer updates

Canonical dataset:

```text
1000 samples × 3 epochs × global batch 32
```

Tổng số samples được xử lý:

$$
1000 \times 3
=
3000
$$

Số optimizer updates:

$$
\left\lceil
\frac{3000}{32}
\right\rceil
=
94
$$

**Accumulation không flush ở biên epoch.**

---

# 10. Artifacts & Resume / Publish

## 10.1 Artifacts

Output directory:

```text
artifacts/stage1/main/
```

### `manifest.json`

Chứa:

* config fingerprint;
* prepared data fingerprint;
* adapter paths;
* SHA256 của adapters.

### `checkpoint.pt`

Chứa:

* adapter states;
* optimizer states;
* scheduler states;
* RNG states.

Checkpoint được thiết kế để resume an toàn.

### `final/adapters/<expert>/`

Mỗi expert chứa PEFT adapter:

```text
adapter_config.json
adapter_model.safetensors
```

### `metrics.jsonl`

Các metric chính:

```text
sft_nll
dpp_loss
Grassmann distances
bandwidth
gradient norms
...
```

---

## 10.2 Resume

Resume từ checkpoint:

```bash
STAGE1_RESUME=artifacts/stage1/main/checkpoint.pt \
./project_commands.sh stage1
```

---

## 10.3 Publish

Nếu set:

```text
HF_REPO_ID
```

script Stage 1 sẽ tự publish sau khi train xong.

Hoặc publish thủ công:

```bash
./project_commands.sh publish-stage1
```

---

# 11. Quan hệ với các stage sau

| Stage         | Phụ thuộc Stage 1                                              |
| ------------- | -------------------------------------------------------------- |
| `supervision` | Load 5 adapters → $U_i$ / $V_i$ / PAG / teacher features |
| `stage2`      | Merge experts (TA/TIES/…) → student; weighted NLL        |

Stage 1 chỉ sinh một **hội đồng experts đa dạng**.

Distillation vào một student adapter diễn ra ở Stage 2.

---

# 12. File mã nguồn chính

| File                                  | Vai trò                                         |
| ------------------------------------- | ----------------------------------------------- |
| `src/cot_mtkd/stage1/trainer.py`      | Vòng train, probe DPP, replay grads, update     |
| `src/cot_mtkd/stage1/dropout.py`      | Step Bernoulli mask + length weights            |
| `src/cot_mtkd/stage1/kneedle.py`      | Council full-vocabulary Kneedle → support       |
| `src/cot_mtkd/stage1/dpp.py`          | Normalized features, $-\log\det$, reduction     |
| `src/cot_mtkd/stage1/grassmann.py`    | $d^2$, kernel, $F_{\mathrm{rep}}$               |
| `src/cot_mtkd/stage1/gac_gradient.py` | $g_{\mathrm{data}}$; Grassmann force. `stable_gac_gradients` là legacy, không vào vòng train |
| `src/cot_mtkd/models/chunked_head.py` | Chunked CE / full-vocab log $p_{\mathcal C}$ / VJP |

---

# 13. Full algorithm

```text
for each optimizer step:

    # ---------------------------------------------------------
    # 1. Probe
    # ---------------------------------------------------------

    for expert m = 1 ... M:

        logits_m = forward_no_grad(expert_m)

    council = mean(softmax(logits_m))

    for each reasoning token t:

        support[t] = Kneedle(council[t])

        support[t] = support[t] - {ground_truth[t]}

        compute DPP on L2(softmax(z_full)[support[t]])

        compute dL_DPP / d log p_m[support[t]]

    # ---------------------------------------------------------
    # 2. Replay
    # ---------------------------------------------------------

    for expert m = 1 ... M:

        forward expert m with gradients

        compute step-dropped SFT loss

        compute VJP of DPP gradient

        g_m =
            grad(SFT_m)
            + lambda_div * grad(DPP_m)

    # ---------------------------------------------------------
    # 3. Synchronize
    # ---------------------------------------------------------

    accumulate gradients
    all-reduce
    normalize

    # ---------------------------------------------------------
    # 4. Clip
    # ---------------------------------------------------------

    g_m = clip(g_m, max_grad_norm=1.0)

    # ---------------------------------------------------------
    # 5. AdamW
    # ---------------------------------------------------------

    AdamW.step(g_m)

    # ---------------------------------------------------------
    # 6. Grassmann repulsion
    # ---------------------------------------------------------

    compute Grassmann distances

    compute RBF kernels

    compute outward force F_rep

    theta_m += eta * lambda_rep * F_rep_m

    # ---------------------------------------------------------
    # 7. Scheduler
    # ---------------------------------------------------------

    scheduler.step()
```

---

# 14. Core design principle

Stage 1 tối ưu ba mục tiêu:

### 1. Goodness

Thông qua SFT:

$$
\mathcal{L}_{\mathrm{SFT}}
$$

### 2. Token-level diversity

Thông qua DPP:

$$
\mathcal{L}_{\mathrm{DPP}}
=
-\log\det(G+\varepsilon I)
$$

### 3. Parameter-space separation

Thông qua Grassmann repulsion:

$$
F_{\mathrm{rep}}
=
\nabla_{\theta}\Phi
$$

Do đó:

$$
\text{Stage 1}
=
\text{Goodness}
+
\text{Token-level Diversity}
+
\text{Parameter-space Separation}
$$

Data gradient:

$$
g_{\mathrm{data}}
=
g_{\mathrm{SFT}}
+
\lambda_{\mathrm{div}}
g_{\mathrm{DPP}}
$$

được đưa qua AdamW.

Grassmann force:

$$
F_{\mathrm{rep}}
$$

được áp dụng trực tiếp lên LoRA parameters sau AdamW.

Kết quả cuối cùng là một council gồm **5 LoRA experts** vừa fit tốt CoT, vừa đa dạng ở token space, vừa được tách trong parameter subspace để đo $U_i,V_i$ và merge thành student ở Stage 2.
