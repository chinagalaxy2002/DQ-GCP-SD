对，我把它收敛成**一个完整主方案**，不再铺很多实验。

# Sim-DETR Layer-Consistent Binding 完整方案

## 目标

现在不要再验证 Baseline 或 D1-only NativeBind。它们已经在 GitHub 里跑完了：Baseline 和 D1-only NativeBind 都是完整训练、validation-selected 的 200 epoch runs；D1-only NativeBind 用 final D4 Hungarian matching，只监督 D1 cross-attention 对 matched GT window 的 attention mass。

新的方案只做一个主版本：

```text
LCB-Full: Layer-Consistent Binding on native Sim-DETR decoder attention
```

核心目标：

```text
让 final matched query 在 D1–D4 中持续绑定同一个 GT occurrence，
而不是只让 D1 暂时学到 ownership。
```

---

## 1. 为什么这个方案正好适合当前 Sim-DETR 代码

Sim-DETR 的 decoder 每层都会输出 intermediate hidden states；每层内部是先 self-attention，再 cross-attention，然后 FFN。D2–D4 的 self-attention 还会根据上一层的 classification score、IoU score 和 span relation 构造 query competition matrix。

这意味着 D1 里形成的 occurrence ownership，很可能在 D2–D4 的 query self-attention 里被重混合。当前结果也已经证明了这一点：D1-only NativeBind 显著改善 D1 ownership，但 D4 ownership 反而变差，test MR mAP 也下降。

所以这次不应该继续加 prompt/router/residual，而应该直接约束：

```text
D1 ownership → D2 ownership → D3 ownership → D4 ownership
```

---

## 2. 不改模型结构，只改训练 loss

不要改 `sim_detr/transformer.py`，也不要改 inference graph。

当前 `NativeCrossAttentionCapture` 已经能 hook 每个 decoder layer 的 `cross_attn`，并且返回所有 decoder layer 的 attention；它的输出会被标准化成 `[batch, query, source]`。 它安装时遍历 `model.transformer.decoder.layers`，对每层 `cross_attn.forward` 做 wrapper，所以本来就能拿到 D1–D4。

当前 D1-only NativeBind 只取了：

```python
native = attentions[0]
```

新的 LCB-Full 改成使用：

```python
attentions[0], attentions[1], attentions[2], attentions[3]
```

也就是 D1–D4 全部 native cross-attention。

---

## 3. Matching 策略

继续使用 **final D4 prediction 的 Hungarian matching**。

不要每层重新 matching。

原因：我们关心的是“同一个 query identity 是否在 decoder trajectory 中绑定同一个 occurrence”。如果每层重新 Hungarian matching，query identity 会被破坏，方案会变成“每层各自找一个 query 匹配 GT”，这不是 ownership consistency。

现有 `SetCriterion.forward()` 已经是先去掉 `aux_outputs`，再用 final outputs 做 matcher。 这个逻辑保持不变。

对每个 final matched pair：

$$
(j,k) \in \mathcal{M}
$$

其中：

* \(j\)：final D4 matched query；
* \(k\)：matched GT occurrence；
* 同一个 \(j\) 在 D1、D2、D3、D4 都应该保持对 \(k\) 的 ownership。

---

## 4. Attention normalization

对每一层 decoder cross-attention：

$$
A^{(\ell)} \in \mathbb{R}^{B \times Q \times T}
$$

其中 \(\ell \in \{1,2,3,4\}\)。

先用 `video_mask` 去掉 padding clips，然后重新 normalize：

$$
\tilde A^{(\ell)}_{bjt}
=
\frac{
A^{(\ell)}_{bjt} \cdot \mathbf{1}[t \text{ valid}]
}{
\sum_{t'} A^{(\ell)}_{bjt'} \cdot \mathbf{1}[t' \text{ valid}] + \epsilon
}
$$

代码逻辑沿用现在 NativeBind：

```python
att = att[..., :video_mask.shape[-1]]
att = att * video_mask[:, None, :].to(att.dtype)
att = att / att.sum(dim=-1, keepdim=True).clamp_min(eps)
```

---

## 5. GT occurrence mask

继续复用现有 `_overlap_for_targets()`，不要重新写 span discretization。

这个函数已经按照 production loss 语义把 GT span 转成 clip-level overlap mask，并兼容 `l1` 和 `ce` span loss。

对每个 GT occurrence \(k\)，得到：

$$
G_k(t) \in \{0,1\}
$$

表示 clip \(t\) 是否落在 GT occurrence \(k\) 内。

---

# 6. Loss 设计

LCB-Full 由三项组成。

---

## 6.1 All-layer matched binding loss

对 final matched query \(j\) 和 matched GT occurrence \(k\)，每一层计算 query 对该 occurrence 的 attention mass：

$$
m^{(\ell)}_{jk}
=
\sum_t
\tilde A^{(\ell)}_{jt} G_k(t)
$$

然后对 D1–D4 全部监督：

$$
L_{\text{layer-bind}}
=
-\frac{1}{4|\mathcal{M}|}
\sum_{\ell=1}^{4}
\sum_{(j,k)\in\mathcal{M}}
\log(m^{(\ell)}_{jk}+\epsilon)
$$

这一步的含义是：

```text
每一层都必须把 matched query 的注意力质量放到同一个 matched GT occurrence 上。
```

它不是要求每层 attention map 完全一样，只要求每层不要丢掉 matched occurrence。

---

## 6.2 Occurrence-level consistency loss

不要做 token-level KL：

$$
KL(A^{(1)}_{j,:} \| A^{(\ell)}_{j,:})
$$

因为这会强迫每层 attention map 长得一样，太硬。

正确做法是 occurrence-level consistency。

假设一个视频有 \(K\) 个 GT occurrences。对 query \(j\)，把 attention mass 聚合成：

$$
o^{(\ell)}_{jr}
=
\sum_t
\tilde A^{(\ell)}_{jt}G_r(t),
\quad r=1,\dots,K
$$

再加一个 background bin：

$$
o^{(\ell)}_{j,\text{bg}}
=
\sum_t
\tilde A^{(\ell)}_{jt}
\left(1-\mathbf{1}[t\in \cup_r G_r]\right)
$$

得到 occurrence-level distribution：

$$
p^{(\ell)}_j
=
\text{Normalize}
\left[
o^{(\ell)}_{j1},
\dots,
o^{(\ell)}_{jK},
o^{(\ell)}_{j,\text{bg}}
\right]
$$

然后让 D2–D4 的 occurrence distribution 接近 D1：

$$
L_{\text{owner-cons}}
=
\frac{1}{3|\mathcal{M}|}
\sum_{\ell=2}^{4}
\sum_{(j,k)\in\mathcal{M}}
JS
\left(
\text{stopgrad}(p^{(1)}_j),
p^{(\ell)}_j
\right)
$$

这里 D1 用 `stopgrad`。

含义是：

```text
D1 是 ownership anchor。
D2–D4 应该保留 D1 的 occurrence identity，
但不需要复制 D1 的具体 clip-level attention pattern。
```

---

## 6.3 Anti-washout loss

再加一个专门防止 D1 ownership 被后层洗掉的 hinge loss：

$$
L_{\text{drop}}
=
\frac{1}{3|\mathcal{M}|}
\sum_{\ell=2}^{4}
\sum_{(j,k)\in\mathcal{M}}
\left[
m^{(1)}_{jk}
-
m^{(\ell)}_{jk}
-
\delta
\right]^2_+
$$

其中：

```text
delta = 0.05
```

它允许后层的 matched occurrence mass 有轻微变化，但不允许明显低于 D1。

这项 loss 很关键，因为它直接对应你的假设：

```text
D1 学到的 ownership 不应该在 D2–D4 被洗掉。
```

---

# 7. 总目标

最终训练目标：

$$
L
=
L_{\text{Sim-DETR}}
+
\lambda_{\text{layer}}
L_{\text{layer-bind}}
+
\lambda_{\text{cons}}
L_{\text{owner-cons}}
+
\lambda_{\text{drop}}
L_{\text{drop}}
$$

第一版固定用：

```text
lambda_layer = 0.5
lambda_cons  = 0.1
lambda_drop  = 0.1
drop_margin  = 0.05
```

不要再调很多版本。这个配置是主方案。

原因：

* D1-only NativeBind 之前用的是 `lambda=0.5`，所以 all-layer binding 也用 `0.5`；
* `L_layer-bind` 已经对 D1–D4 求平均，不会把 loss 放大 4 倍；
* consistency 和 drop 只是辅助，不应该压过原始 Sim-DETR loss。

---

# 8. 代码改动位置

只改两个地方。

## 8.1 新增 `install_layer_consistent_binding_control`

放在：

```text
causal_occurrence_lab/controls.py
```

新增函数：

```python
def install_layer_consistent_binding_control(
    criterion,
    attention_capture,
    *,
    layer_bind_coef=0.5,
    owner_cons_coef=0.1,
    drop_coef=0.1,
    drop_margin=0.05,
    layers=(0, 1, 2, 3),
):
    ...
```

它和现在的 `install_native_binding_control()` 类似，但区别是：

```text
旧版：只用 attentions[0]
新版：用 attentions[0:4]
```

输出三个 loss：

```python
losses["loss_lcb_layer_bind"] = layer_bind
losses["loss_lcb_owner_cons"] = owner_cons
losses["loss_lcb_drop"] = drop
```

并注册权重：

```python
criterion.weight_dict["loss_lcb_layer_bind"] = 0.5
criterion.weight_dict["loss_lcb_owner_cons"] = 0.1
criterion.weight_dict["loss_lcb_drop"] = 0.1
```

---

## 8.2 在 `train_causal.py` 增加一个 variant

在 `VARIANTS` 里只加一个主 variant：

```python
"lcb_full": {
    "use_dq": False,
    "binding": 0.5,
    "route": 0.0,
    "inject": False,
    "target": None,
    "native": "layer_consistent",
}
```

然后在 `build_variant()` 里：

```python
if cfg["native"] == "layer_consistent":
    capture = NativeCrossAttentionCapture(model).install()
    install_layer_consistent_binding_control(
        criterion,
        capture,
        layer_bind_coef=0.5,
        owner_cons_coef=0.1,
        drop_coef=0.1,
        drop_margin=0.05,
    )
```

这仍然是 **training-only regularization**。Inference 时不需要 attention capture，也不增加任何模型参数。当前 NativeBind 已经验证过，attention capture wrapper 不改变输出、不改变 state dict key，也没有 `query_cgp.*` 参数。

---

# 9. 训练设置

直接沿用现有 Baseline / NativeBind protocol：

```text
seed = 2017
decoder layers = 4
encoder layers = 2
batch size = 32
lr = 1e-4
lr drop = epoch 100
epochs = 200
VTC coefficient = 0.3
CTC coefficient = 0.5
checkpoint selection = validation MR-full-mAP
```

这些设置和现有 NativeBind report 一致。

只跑一个新模型：

```text
LCB-Full
```

对比表格直接用已有：

```text
Baseline
D1-only NativeBind
LCB-Full
```

---

# 10. 评估重点

不要只看 headline mAP。这个方案的核心是验证 ownership propagation，所以主表应该包含：

```text
D1 AEC
D2 AEC
D3 AEC
D4 AEC
D1 ECR
D2 ECR
D3 ECR
D4 ECR
D1→D4 ownership persistence
D4 MR mAP Avg
multi-occurrence MR mAP Avg
```

当前 D1-only NativeBind 的失败点是：

```text
D1 AEC 大幅提升
D1 ECR 大幅下降
但 D4 AEC 下降
D4 ECR 上升
test MR mAP 下降
```

LCB-Full 成功的理想模式是：

```text
D1 AEC 保持高
D4 AEC 提升
D4 ECR 下降
D1→D4 persistence 提升
multi-occurrence mAP 提升
headline mAP 不下降，最好提升
```

---

# 11. 预期结论

这个方案跑完后，你可以明确回答一个比 DQ-CGP 更干净的问题：

```text
D1-only binding 是否失败，是因为 binding 本身没用，
还是因为 early ownership 没有传递到 final decoder layer？
```

如果 LCB-Full 提升 D4 AEC/ECR 和 multi-occurrence mAP，那么结论就是：

```text
Binding 是有用的，但必须 layer-consistent。
D1-only binding 只改变 early evidence ownership，
不能保证 final retrieval query 保持同一个 occurrence identity。
```

如果 LCB-Full 仍然不提升 mAP，但 D4 ownership 变好，那么结论是：

```text
Query-occurrence ownership 可以被稳定化，
但 final ranking / span regression 还没有充分利用这个 ownership。
```

这两个结果都比继续证明“prompt 有没有用”更有研究价值。

---

# 12. 最终方法一句话

可以把方法写成：

```text
We propose Layer-Consistent Binding, a training-only objective for Sim-DETR that supervises native decoder cross-attention across D1–D4 at the occurrence level. Instead of enforcing identical token-level attention maps, it preserves the ownership identity of each final Hungarian-matched query across decoder layers, preventing early query-occurrence binding from being washed out by subsequent self-attention and cross-attention updates.
```

中文核心表述：

```text
Layer-Consistent Binding 不要求每层 attention map 一样，
而是要求同一个 final matched query 在 D1–D4 中始终拥有同一个 GT occurrence。
它解决的不是 prompt 缺失问题，而是 DETR query 在多时刻检索中的 occurrence ownership 漂移问题。
```
