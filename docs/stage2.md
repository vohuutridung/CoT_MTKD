# Stage 2 — Council-Weighted Reasoning Distillation via Model Merging

Run:

```bash
./project_commands.sh stage2
```

Ablation theo phương pháp merge:

```bash
STAGE2_MERGE_METHOD=ties ./project_commands.sh stage2
STAGE2_MERGE_METHOD=dare_ties ./project_commands.sh stage2
STAGE2_MERGE_METHOD=tsv ./project_commands.sh stage2
STAGE2_MERGE_METHOD=iso_c ./project_commands.sh stage2
```

hoặc:

```bash
./project_commands.sh stage2
# tương đương
python -m cot_mtkd.cli.train_stage2 \
  --config configs/stage2/qwen25_7b.yaml \
  --merge-method tsv \
  --set paths.output=artifacts/stage2/tsv
```

Entry point:

```text
scripts/50_train_stage2.sh
    -> cot_mtkd.cli.train_stage2
    -> cot_mtkd.stage2.trainer.train_stage2
```

Default config:

```text
configs/stage2/qwen25_7b.yaml
```

Smoke config:

```text
configs/smoke/stage2.yaml
```

Output:

```text
artifacts/stage2/main/
├── manifest.json
├── checkpoint.pt
├── config.yaml
├── benchmark/
│   ├── adapter_states.pt
│   └── checkpoint.pt
├── final/
│   └── adapters/student/
└── metrics.jsonl
```

Artifact bắt buộc trước khi train:

| Artifact | Script | Dùng cho |
| --- | --- | --- |
| prepared records | `./project_commands.sh prepare` | token / region / label |
| Stage-1 adapters | `./project_commands.sh stage1` | merge → student init |
| supervision | `./project_commands.sh supervision` | $U_i$, $V_i$ (disagreement) |

Teacher cache / KL **không** tham gia Phase 2.

---

## 1. Mục tiêu

Khớp Phase 2 trong `docs/proposal` (các phương trình từ phân phối đồng thuận `eq:pbar` đến `eq:phase2loss` / `eq:weffstudent`).

Sau Phase 1, hội đồng $M$ LoRA experts bị **đóng băng**. Stage 2:

1. Đo mean uncertainty $U_i$ và disagreement $V_i$ trên từng reasoning step.
2. Gán trọng số step $w_i$ từ $U_i$ và $\rho_i = V_i/\max(U_i,\varepsilon_U)$.
3. Merge $M$ expert thành một student $\tilde\theta$ bằng trung bình $\Delta W_m=sB_mA_m$ rồi $\operatorname{TruncSVD}_r$ (mặc định; ablation TIES/DARE-TIES/TSV/Iso-C).
4. Distill bằng **NLL có trọng số step** — không dùng KL.

Base model đóng băng; chỉ train LoRA student.

---

## 2. Pipeline một optimizer step

```text
┌─────────────────────────────────────────────────────────────┐
│ 0. Init (một lần): θ̃ = F_merge(φ_1, …, φ_M)                │
│                                                             │
│ 1. Forward decoder student → hidden states                  │
│                                                             │
│ 2. Gán w_i từ U, ρ (z-score trong sample, tanh)             │
│    Token weight = w_s / T_s trên reasoning step s           │
│    Answer / format: khối luôn-on, trọng số 1                │
│                                                             │
│ 3. Chunk LM head → CE có trọng số (length-normalized NLL)   │
│                                                             │
│ 4. VJP hidden → LoRA params                                 │
│                                                             │
│ 5. Accumulate → all-reduce → chia Σ w                       │
│                                                             │
│ 6. Clip ||g|| ≤ max_grad_norm                               │
│                                                             │
│ 7. AdamW.step(g)  +  cosine scheduler.step()                │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Ký hiệu (proposal)

| Ký hiệu | Ý nghĩa |
| --- | --- |
| $\Phi=\{\phi_m\}_{m=1}^{M}$ | Hội đồng LoRA Stage 1 |
| $\tilde\theta$ | Student sau merge |
| $s_i$ | Reasoning step $i$ trong sample, $T_i$ token |
| $\hat p_{i,t}^{(m)}$ | Phân phối expert $m$ tại token $t$ của step $i$ |
| $\bar p_{i,t}$ | Mixture đều $\frac{1}{M}\sum_m \hat p_{i,t}^{(m)}$ (`eq:pbar`) |
| $U_i$ | Mean uncertainty (`eq:U`) |
| $V_i$ | Disagreement / JS (`eq:V`) |
| $\rho_i=V_i/\max(U_i,\varepsilon_U)$ | Relative disagreement (`eq:rho`), $\varepsilon_U=10^{-6}$ |
| $w_i$ | Trọng số step (`eq:w`) |
| $\lambda_U$ | `stage2.lambda_uncertainty = 0.5` |
| $\lambda_D$ | `stage2.lambda_disagreement = 0.5` |
| $\mathcal{F}_{\mathrm{merge}}$ | `stage2.merge_method` |

---

## 4. Tín hiệu hội đồng (`eq:E`–`eq:rho`)

Tính **offline** lúc `supervision`, trên token **REASONING**, temperature `signals.temperature_features = 1.0`.

Entropy expert $m$ trên step $s_i$:

$$
E_i^{(m)}
=
\frac{1}{T_i}\sum_{t=1}^{T_i}
\mathcal{H}\!\left(\hat p_{i,t}^{(m)}\right)
$$

Mean uncertainty:

$$
U_i = \frac{1}{M}\sum_{m=1}^{M} E_i^{(m)}
$$

Disagreement (mutual information expert–prediction; trùng $\mathrm{mean}_m\mathrm{KL}(\hat p^{(m)}\|\bar p)$):

$$
V_i
=
\frac{1}{T_i}\sum_{t=1}^{T_i}
\left[
\mathcal{H}(\bar p_{i,t})
-
\frac{1}{M}\sum_{m=1}^{M}\mathcal{H}\!\left(\hat p_{i,t}^{(m)}\right)
\right]
$$

Phân rã `eq:decomp`: $\bar{\mathcal{H}}_i = U_i + V_i$. Relative disagreement:

$$
\rho_i = \frac{V_i}{\max(U_i,\varepsilon_U)}, \qquad \varepsilon_U=10^{-6}
$$

Supervision ghi `mean_uncertainty`, `disagreement`, `relative_disagreement`. $\lambda_U,\lambda_D$ **không** nướng vào artifact — ablation hệ số không cần build lại signals.

---

## 5. Step weighting (`eq:zscore`, `eq:w`)

Z-score trên $S$ step của **chính sample đó**:

$$
\hat U_i = \frac{U_i-\mu_U}{\sigma_U},
\qquad
\hat\rho_i = \frac{\rho_i-\mu_\rho}{\sigma_\rho}
$$

Nếu $S\le 1$ hoặc $\sigma=0$ thì z-score $=0$.

$$
w_i = 1 + \lambda_U\tanh(\hat U_i) + \lambda_D\tanh(\hat\rho_i)
$$

$w_i$ bị chặn trong $(1-\lambda_U-\lambda_D,\, 1+\lambda_U+\lambda_D)$ rồi kẹp $w_i\leftarrow\max(w_i,\varepsilon_w)$ với $\varepsilon_w=10^{-6}$. Default $\lambda_U=\lambda_D=0.5$ nên $w_i\in(0,2)$.

---

## 6. Merge student (`eq:merge`)

Toán tử mặc định (`ta`) là trung bình task-vector rồi nén hạng $r$:

$$
\Delta W_m = s B_m A_m,
\qquad
\Delta W_{\mathrm{merge}}
=
\operatorname{TruncSVD}_{r}
\left(
\frac1M\sum_{m=1}^{M}\Delta W_m
\right)
$$

Tách lại LoRA: $\overline{\Delta W}/s = U\Sigma V^{\top}$ thì $B_{\mathrm{eff}}=U_r\Sigma_r$, $A_{\mathrm{eff}}=V_r^{\top}$, nên $s B_{\mathrm{eff}} A_{\mathrm{eff}}\approx\Delta W_{\mathrm{merge}}$. Không cộng $A,B$ riêng và không lấy tổng thô $\sum_m\Delta W_m$.

| `merge_method` | Việc làm |
| --- | --- |
| `ta` | $\operatorname{TruncSVD}_r(\mathrm{mean}_m\Delta W_m)$ |
| `ties` | Trim top-`ties_density`, elect sign, disjoint mean, rồi nén hạng $r$ |
| `dare_ties` | Drop-and-rescale (`dare_drop_prob`) rồi TIES, rồi nén hạng $r$ |
| `tsv` | SVD từng $\Delta W$, ghép hướng kỳ dị, trực giao hóa (reduction mặc định $1/M$), rồi nén hạng $r$ |
| `iso_c` | TA rồi làm phẳng phổ kỳ dị, rồi nén hạng $r$ |

Default: `ta`. Seed DARE = `config.seed`. `merge_scaling` nhân $\Delta W$ sau merge (mặc định $1$).

---

## 7. Weighted NLL (`eq:nll`, `eq:nllstudent`, `eq:phase2loss`)

NLL chuẩn hóa độ dài của student trên step $s_i$:

$$
\mathcal{L}_i^{(\tilde\theta)}
=
-\frac{1}{T_i}\sum_{t\in s_i}
\log p_{\tilde\theta}(y_t\mid y_{<t}, x)
$$

Phần reasoning (`eq:nllstudent`):

$$
\widetilde{\mathcal{L}}_{\mathrm{NLL}}
=
\frac{\sum_{i=1}^{S} w_i\,\mathcal{L}_i^{(\tilde\theta)}}{\sum_{i=1}^{S} w_i}
$$

Proposal gán đáp án như khối luôn-on trọng số $1$, và token định dạng **không** thuộc reasoning step. Hiện thực hóa (cùng tinh thần Phase 1):

$$
\mathcal{L}_{\mathrm{Phase2}}
=
\frac{
\sum_{i=1}^{S} w_i\,\mathcal{L}_i^{(\tilde\theta)}
+
\mathcal{L}_{\mathrm{ans}}
+
\mathcal{L}_{\mathrm{fixed}}
}{
\sum_{i=1}^{S} w_i
+
\mathbf{1}_{\mathrm{ans}}
+
\mathbf{1}_{\mathrm{fixed}}
}
$$

`ANSWER` là $\mathcal{L}_{\mathrm{ans}}$; `ASSISTANT_CONTROL`, `DELIMITER`, `ANSWER_MARKER`, `EOS` vào $\mathcal{L}_{\mathrm{fixed}}$. Cả hai khối chuẩn hóa theo độ dài riêng, mỗi khối đóng góp đúng một đơn vị vào mẫu số.

Token-level: reasoning token thuộc step $s$ nhận $w_s/T_s$; token always-on nhận $1/T_{\mathrm{block}}$. LM head chunk 64 token. Sau accumulation + all-reduce, gradient chia $\sum w$.

Logged metric: `nll` (= `total_loss`).

---

## 8. Init / batch / optimizer

| Hạng mục | Default 7B |
| --- | --- |
| Init | merge $M=5$ expert (`merge_method=ta`) |
| Epochs | 5 (scheduler horizon) |
| Benchmark checkpoint | ~epoch 3 (step 1500) |
| Micro batch | 1 |
| Global batch | 2 |
| Accumulation (1 GPU) | 2 |
| Optimizer updates | $\lceil 1000\times 5/2\rceil=2500$ |
| LR | $5\times 10^{-5}$ |
| Optimizer | AdamW, $\beta=(0.9,0.999)$, wd $=0$ |
| Scheduler | cosine, `warmup_ratio=0.10` |
| LoRA rank / alpha / dropout | $4$ / $8$ / $0$ |
| Clip | 1.0 |
| $\lambda_U$, $\lambda_D$ | 0.5, 0.5 |

Accumulation không reset khi sang epoch.

---

## 9. Suy luận (`eq:weffstudent`)

$$
W_{\mathrm{eff}}^{(\ell)} = W_{\mathrm{base}}^{(\ell)} + s\,\tilde B^{(\ell)}\tilde A^{(\ell)}
$$

Một adapter LoRA rank $r$; không giữ multi-adapter lúc generate.

---

## 10. File mã nguồn chính

| File | Vai trò |
| --- | --- |
| `src/cot_mtkd/stage2/trainer.py` | Merge init, vòng train, weighted NLL |
| `src/cot_mtkd/stage2/merge.py` | TA / TIES / DARE-TIES / TSV / Iso-C |
| `src/cot_mtkd/stage2/weights.py` | $w_i$ và token weight length-normalized |
| `src/cot_mtkd/signals/predictive.py` | $U_i$, $V_i$ từ entropy / mixture |
| `src/cot_mtkd/cli/train_stage2.py` | `--merge-method` + `--set` |

---

## 11. Thuật toán

```text
θ̃ ← F_merge(stage1 experts; method from config/CLI)

for each micro-batch:

    hidden = decoder(student, input_ids)

    for each sample:
        Û, ρ̂ = zscore_within_sample(U), zscore_within_sample(V / max(U, ε_U))
        w = max(1 + λ_U tanh(Û) + λ_D tanh(ρ̂), ε_w)
        reasoning token in step s:  weight = w_s / T_s
        answer tokens:              weight = 1 / T_ans
        format tokens:              weight = 1 / T_fixed

    H = sum_t weight_t * CE(logits_t, y_t)
    denom += sum_s w_s + 1_ans + 1_fixed

    accumulate VJP(H → LoRA)

    if accumulation boundary:
        all-reduce
        nll = H / denom
        g = clip(G / denom, max_grad_norm=1)
        AdamW.step(g)
        scheduler.step()
```
