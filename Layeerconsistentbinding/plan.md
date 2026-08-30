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

---

## 3. Matching 策略

继续使用 **final D4 prediction 的 Hungarian matching**。

不要每层重新 matching。

原因：我们关心的是“同一个 query identity 是否在 decoder trajectory 中绑定同一个 occurrence”。如果每层重新 Hungarian matching，query identity 会被破坏，方案会变成“每层各自找一个 query 匹配 GT”，这不是 ownership consistency。

---

## 4. 正式版本：「Acquire → Preserve」解耦目标

将整个机制明确解耦为两个阶段的目标：
$$\boxed{ \text{Acquire ownership at D1} \longrightarrow \text{Preserve ownership through D2–D4} }$$

最终主方案目标函数：
$$\boxed{ L = L_{\text{Sim-DETR}} + 0.5 \cdot L_{\text{D1-bind}} + 0.1 \cdot L_{\text{late-bind}} + 0.1 \cdot L_{\text{owner-cons}} + 0.1 \cdot L_{\text{drop}} }$$

### 4.1 D1 Ownership Acquisition ($L_{\text{D1-bind}}$, $\lambda_{D1}=0.5$)
完全保留已验证过的 NativeBind：
$$L_{\text{D1-bind}} = -\frac{1}{|\mathcal{M}|} \sum_{(j,k)\in\mathcal{M}} \log(m^{(1)}_{jk} + \epsilon)$$
- 作用：让 final matched query 在首层建立对 GT occurrence 的 ownership。

### 4.2 D2–D4 Direct Ownership Maintenance ($L_{\text{late-bind}}$, $\lambda_{\text{late}}=0.1$)
$$L_{\text{late-bind}} = -\frac{1}{3|\mathcal{M}|} \sum_{\ell=2}^{4} \sum_{(j,k)\in\mathcal{M}} \log(m^{(\ell)}_{jk} + \epsilon)$$
- 作用：不是重新学习 owner，而是防止 D2–D4 完全丢掉 GT occurrence。

### 4.3 D1 → D2–D4 Ownership Consistency ($L_{\text{owner-cons}}$, $\lambda_{\text{cons}}=0.1$)
$$L_{\text{owner-cons}} = \frac{1}{3|\mathcal{M}|} \sum_{\ell=2}^{4} \sum_{(j,k)\in\mathcal{M}} JS\left(\text{stopgrad}(p^{(1)}_j), p^{(\ell)}_j\right)$$
- 作用：不允许 occurrence identity 在后续层漂移。

### 4.4 Anti-Washout Protection ($L_{\text{drop}}$, $\lambda_{\text{drop}}=0.1$, $\delta=0.05$)
$$L_{\text{drop}} = \frac{1}{3|\mathcal{M}|} \sum_{\ell=2}^{4} \sum_{(j,k)\in\mathcal{M}} \left[ m^{(1)}_{jk} - m^{(\ell)}_{jk} - 0.05 \right]^2_+$$
- 作用：不允许 matched occurrence mass 在后续层发生严重衰减。

---

## 5. 逻辑总结

```text
D1 bind:             让 query 获取 occurrence ownership
D2-D4 late bind:     后续层仍然知道自己的 GT owner
owner consistency:   不允许 occurrence identity 漂移
anti-washout:        不允许 matched occurrence mass 严重衰减
```
