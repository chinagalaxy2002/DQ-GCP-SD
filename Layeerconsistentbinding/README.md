# Sim-DETR Layer-Consistent Binding (LCB-Full)

This folder contains the complete, self-contained implementation of **Layer-Consistent Binding (LCB-Full)** for Sim-DETR as specified in `plan.md`.

---

## 1. Overview & Motivation

Sim-DETR decoder layers process queries sequentially through self-attention, cross-attention, and FFN sub-layers with query competition. In multi-occurrence video moment retrieval, ownership established in early layers (e.g. D1) can be washed out or remixed in subsequent self-attention layers (D2–D4). 

**LCB-Full** addresses this by applying a training-only regularizer across native cross-attention maps ($D_1 \to D_4$) without adding any model parameters or altering inference:
- **All-Layer Matched Binding ($L_{\text{layer-bind}}$)**: Supervise attention mass of final Hungarian-matched queries on their corresponding GT occurrences across all decoder layers D1–D4.
- **Occurrence-Level Consistency ($L_{\text{owner-cons}}$)**: Anchor early ownership at D1 and enforce that occurrence distributions across D2–D4 stay consistent with D1 using Jensen-Shannon divergence ($\text{stopgrad}(p^{(1)})$).
- **Anti-Washout Protection ($L_{\text{drop}}$)**: Hinge loss preventing attention mass on the matched occurrence in D2–D4 from dropping below D1's mass by more than a margin $\delta = 0.05$.

---

## 2. Mathematical Formulation

### 2.1 Attention Normalization
For decoder layer $\ell \in \{1, 2, 3, 4\}$ with native cross-attention $A^{(\ell)} \in \mathbb{R}^{B \times Q \times T}$:
$$\tilde A^{(\ell)}_{bjt} = \frac{A^{(\ell)}_{bjt} \cdot \mathbf{1}[t \text{ valid}]}{\sum_{t'} A^{(\ell)}_{bjt'} \cdot \mathbf{1}[t' \text{ valid}] + \epsilon}$$

### 2.2 Final D4 Hungarian Matching
Hungarian matching is executed once on the final D4 predictions:
$$\mathcal{M} = \text{Matcher}(\text{outputs}_{\text{final}}, \text{targets})$$
The exact matched query identity $j$ is maintained and supervised across D1, D2, D3, D4 for matched GT occurrence $k$.

### 2.3 Loss Objectives
1. **All-Layer Matched Binding Loss**:
   $$m^{(\ell)}_{jk} = \sum_t \tilde A^{(\ell)}_{jt} G_k(t)$$
   $$L_{\text{layer-bind}} = -\frac{1}{4|\mathcal{M}|} \sum_{\ell=1}^{4} \sum_{(j,k)\in\mathcal{M}} \log(m^{(\ell)}_{jk} + \epsilon)$$

2. **Occurrence-Level Consistency Loss**:
   $$o^{(\ell)}_{jr} = \sum_t \tilde A^{(\ell)}_{jt} G_r(t), \quad r=1,\dots,K$$
   $$o^{(\ell)}_{j,\text{bg}} = \sum_t \tilde A^{(\ell)}_{jt} \left(1 - \mathbf{1}[t \in \cup_r G_r]\right)$$
   $$p^{(\ell)}_j = \text{Normalize}\left[ o^{(\ell)}_{j1}, \dots, o^{(\ell)}_{jK}, o^{(\ell)}_{j,\text{bg}} \right]$$
   $$L_{\text{owner-cons}} = \frac{1}{3|\mathcal{M}|} \sum_{\ell=2}^{4} \sum_{(j,k)\in\mathcal{M}} JS\left(\text{stopgrad}(p^{(1)}_j), p^{(\ell)}_j\right)$$

3. **Anti-Washout Loss**:
   $$L_{\text{drop}} = \frac{1}{3|\mathcal{M}|} \sum_{\ell=2}^{4} \sum_{(j,k)\in\mathcal{M}} \left[ m^{(1)}_{jk} - m^{(\ell)}_{jk} - \delta \right]^2_+, \quad \delta = 0.05$$

4. **Total Training Objective**:
   $$L = L_{\text{Sim-DETR}} + 0.5 \cdot L_{\text{layer-bind}} + 0.1 \cdot L_{\text{owner-cons}} + 0.1 \cdot L_{\text{drop}}$$

---

## 3. Directory Layout

```
Layeerconsistentbinding/
├── plan.md                                # Master specification
├── README.md                              # Documentation & usage guide
├── __init__.py                            # Package initialization
├── controls.py                            # Loss implementations & criterion hooks
├── metrics.py                             # Layerwise AEC, ECR & D1->D4 persistence metrics
├── train_lcb.py                           # Training entrypoint for LCB-Full
├── evaluate_lcb.py                        # Checkpoint evaluation & persistence analyzer
├── scripts/
│   ├── run_train_lcb.sh                   # Reproduction training script
│   └── run_eval_lcb.sh                    # Evaluation script
└── tests/
    ├── __init__.py
    ├── test_lcb_controls.py               # Unit tests for JS div, loss math & gradients
    ├── test_lcb_metrics.py                # Unit tests for persistence and occurrence metrics
    └── test_lcb_integration.py           # Forward/backward step test with Sim-DETR
```

---

## 4. Usage

### 4.1 Running Training
To train LCB-Full on QVHighlights using the standard protocol (seed=2017, 200 epochs, lr=1e-4, lr_drop=100, VTC=0.3, CTC=0.5):

```bash
bash Layeerconsistentbinding/scripts/run_train_lcb.sh
```

Or invoke directly via Python:
```bash
python Layeerconsistentbinding/train_lcb.py \
  --variant lcb_full \
  --output-dir Layeerconsistentbinding/outputs/lcb_full_seed2017 \
  --dset_name hl \
  --ctx_mode video_tef \
  --train_path data/highlight_train_release.jsonl \
  --eval_path data/highlight_val_release.jsonl \
  --v_feat_dirs data/slowfast_features data/clip_b32_vid_k4 \
  --v_feat_dim 5376 \
  --t_feat_dir data/clip_b32_txt_k4 \
  --t_feat_dim 2048 \
  --dec_layers 4 \
  --enc_layers 2 \
  --bsz 32 \
  --lr 0.0001 \
  --lr_drop 100 \
  --n_epoch 200 \
  --seed 2017 \
  --VTC_loss_coef 0.3 \
  --CTC_loss_coef 0.5 \
  --lcb-layer-bind-coef 0.5 \
  --lcb-owner-cons-coef 0.1 \
  --lcb-drop-coef 0.1 \
  --lcb-drop-margin 0.05
```

### 4.2 Running Evaluation
To evaluate a checkpoint across all decoder layers and compute ownership persistence:

```bash
bash Layeerconsistentbinding/scripts/run_eval_lcb.sh \
  Layeerconsistentbinding/outputs/lcb_full_seed2017/model_best.ckpt \
  data/highlight_test_with_gt.jsonl \
  Layeerconsistentbinding/outputs/lcb_full_seed2017/eval_test
```

### 4.3 Running Tests
Run the self-contained test suite:
```bash
python -m unittest discover -s Layeerconsistentbinding/tests
```
