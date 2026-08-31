# Sim-DETR Layer-Consistent Binding (LCB Acquire → Preserve)

This folder contains the complete, self-contained implementation of **Layer-Consistent Binding (LCB Acquire → Preserve)** for Sim-DETR.

---

## 1. Overview & Motivation

Sim-DETR decoder layers process queries sequentially through self-attention, cross-attention, and FFN sub-layers with query competition. In multi-occurrence video moment retrieval, ownership established in early layers (e.g. D1) can be washed out or remixed in subsequent self-attention layers (D2–D4). 

The **Acquire → Preserve** framework explicitly decouples this into two distinct stage objectives:
$$\boxed{ \text{Acquire ownership at D1} \longrightarrow \text{Preserve ownership through D2–D4} }$$

1. **D1 Ownership Acquisition ($L_{\text{D1-bind}}$, $\lambda_{\text{D1}} = 0.5$)**:
   Completely preserves the verified NativeBind formulation on D1, providing an identical learning signal for initial occurrence acquisition:
   $$L_{\text{D1-bind}} = -\frac{1}{|\mathcal{M}|} \sum_{(j,k)\in\mathcal{M}} \log(m^{(1)}_{jk} + \epsilon)$$

2. **D2–D4 Direct Ownership Maintenance ($L_{\text{late-bind}}$, $\lambda_{\text{late}} = 0.1$)**:
   Directly supervises D2–D4 cross-attentions to prevent subsequent layers from losing track of their GT occurrence:
   $$L_{\text{late-bind}} = -\frac{1}{3|\mathcal{M}|} \sum_{\ell=2}^{4} \sum_{(j,k)\in\mathcal{M}} \log(m^{(\ell)}_{jk} + \epsilon)$$

3. **D1 → D2–D4 Ownership Consistency ($L_{\text{owner-cons}}$, $\lambda_{\text{cons}} = 0.1$)**:
   Anchors early ownership at D1 and enforces that occurrence distributions across D2–D4 stay consistent with D1 using Jensen-Shannon divergence ($\text{stopgrad}(p^{(1)})$):
   $$L_{\text{owner-cons}} = \frac{1}{3|\mathcal{M}|} \sum_{\ell=2}^{4} \sum_{(j,k)\in\mathcal{M}} JS\left(\text{stopgrad}(p^{(1)}_j), p^{(\ell)}_j\right)$$

4. **Anti-Washout Protection ($L_{\text{drop}}$, $\lambda_{\text{drop}} = 0.1$, $\delta = 0.05$)**:
   Hinge loss preventing attention mass on the matched occurrence in D2–D4 from decaying below D1's mass:
   $$L_{\text{drop}} = \frac{1}{3|\mathcal{M}|} \sum_{\ell=2}^{4} \sum_{(j,k)\in\mathcal{M}} \left[ m^{(1)}_{jk} - m^{(\ell)}_{jk} - \delta \right]^2_+, \quad \delta = 0.05$$

### Total Training Objective
$$\boxed{ L = L_{\text{Sim-DETR}} + 0.5 \cdot L_{\text{D1-bind}} + 0.1 \cdot L_{\text{late-bind}} + 0.1 \cdot L_{\text{owner-cons}} + 0.1 \cdot L_{\text{drop}} }$$

---

## 2. Directory Layout

```
Layeerconsistentbinding/
├── plan.md                                # Master specification & Acquire -> Preserve notes
├── README.md                              # Documentation & usage guide
├── __init__.py                            # Package initialization
├── controls.py                            # Loss implementations & criterion hooks
├── metrics.py                             # Layerwise AEC, ECR & D1->D4 persistence metrics
├── train_lcb.py                           # Training entrypoint for LCB Acquire -> Preserve
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

## 3. Usage

### 3.1 Running Training
To train LCB Acquire → Preserve on QVHighlights using the standard protocol (seed=2017, 200 epochs, lr=1e-4, lr_drop=100, VTC=0.3, CTC=0.5):

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
  --eval_split_name val \
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
  --lcb-d1-bind-coef 0.5 \
  --lcb-late-bind-coef 0.1 \
  --lcb-owner-cons-coef 0.1 \
  --lcb-drop-coef 0.1 \
  --lcb-drop-margin 0.05
```

### 3.2 Running Evaluation
To evaluate a checkpoint across all decoder layers and compute ownership persistence:

```bash
bash Layeerconsistentbinding/scripts/run_eval_lcb.sh \
  Layeerconsistentbinding/outputs/lcb_full_seed2017/model_best.ckpt \
  data/highlight_test_with_gt.jsonl \
  Layeerconsistentbinding/outputs/lcb_full_seed2017/eval_test
```

### 3.3 Running Tests
Run the self-contained test suite:
```bash
python -m unittest discover -s Layeerconsistentbinding/tests
```

## 4. Published Run Record

The configuration, metrics, logs, validation predictions, and layer-wise test
submissions for the full seed-2017 run are versioned in
[`results/lcb_full_seed2017`](results/lcb_full_seed2017/). Large checkpoints,
TensorBoard events, and redundant per-record analysis are intentionally excluded
from the repository.
