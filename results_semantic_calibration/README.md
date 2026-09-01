# Sim-DETR Candidate-Conditioned Semantic Calibration

## Experiment status

Completed on QVHighlights with seed 2017. Static and Full were trained from
scratch for 200 epochs. The native Sim-DETR baseline was not retrained because
the repository already contains its seed-2017 reproduction. Checkpoints were
selected only by validation `MR-full-mAP`; no test annotations were used.

## Method

This experiment asks whether a final Sim-DETR decoder candidate benefits from
being semantically evaluated with its own native temporal evidence. It adds a
late semantic residual after the decoder:

$$
\begin{aligned}
p_i(t) &= \operatorname{normalize}(M_i^{native}(t)),\\
v_i &= \sum_t p_i(t)F_t,\\
e_i &= \operatorname{LN}\left(e+\operatorname{MLP}
       ([\operatorname{LN}(e),\operatorname{LN}(v_i)])\right),\\
z_i &= \cos(W_q q_i,W_e e_i),\\
\ell'_{i,fg} &= \ell^{native}_{i,fg}+\operatorname{softplus}(\gamma)z_i,\\
\ell'_{i,bg} &= \ell^{native}_{i,bg}.
\end{aligned}
$$

Only the final foreground classification logit changes. The decoder,
background logit, spans, candidate masks, IoU scores, saliency scores,
auxiliary outputs, matcher, and original Sim-DETR losses remain unchanged.
Native mask weights are detached by default, so the semantic branch does not
retrain temporal ownership through `pred_masks`.

The two trained variants are:

- **Static**: $z_i=\cos(W_q q_i,W_e e)$; no candidate evidence is used.
- **Full / CSC**: $z_i=\cos(W_q q_i,W_e e_i)$ with candidate evidence $v_i$.

The implementation is isolated in
[`sim_detr/semantic_calibration`](../sim_detr/semantic_calibration/). Original
`sim_detr/model.py`, `transformer.py`, and `matcher.py` were not modified.

## Training protocol

| Setting | Value |
| --- | --- |
| Dataset | QVHighlights |
| Video features | SlowFast + CLIP |
| Text features | CLIP |
| Seed | 2017 |
| Batch size | 32 |
| Learning rate | 1e-4 |
| Epochs / LR drop | 200 / 100 |
| Encoder / decoder layers | 2 / 4 |
| VTC / CTC coefficients | 0.3 / 0.5 |
| Label coefficient | 4 |
| Best-checkpoint criterion | Validation MR-full-mAP |

## Main validation results

| Method | Best epoch | R1@0.5 | R1@0.7 | mAP@0.5 | mAP@0.75 | MR mAP | Δ vs baseline |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Sim-DETR baseline (existing reproduction) | — | 68.32 | 53.81 | 69.03 | 50.77 | **49.14** | — |
| + Static Semantic Calibration | 158 | **68.52** | 53.35 | **69.25** | 50.33 | 48.95 | -0.19 |
| + Candidate-Conditioned Calibration (Full) | 200 | 67.81 | **54.52** | 68.78 | 50.20 | **49.14** | +0.00 |

Full improves over separately trained Static by `+0.19` MR mAP, but only
matches the existing baseline. The metric trade-offs are mixed: Full improves
R1@0.7 while reducing R1@0.5, mAP@0.5, and mAP@0.75.

## Expanded Full-checkpoint counterfactuals

All rows below use the same Full best checkpoint (epoch 200). The validation
set contains 1,550 queries: 1,020 single-occurrence queries (at most one GT
window) and 530 multi-occurrence queries (at least two GT windows). Random
derangement uses seed 2017 and guarantees that no candidate retains its own
context. Farthest-context selects the least cosine-similar context among the
other candidates. `static` is an inference intervention on the Full
checkpoint, not the separately trained Static model.

### All queries (n=1,550)

| Intervention | R1@0.5 | R1@0.7 | mAP@0.5 | mAP@0.75 | MR mAP | Δ MR mAP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| aligned | 67.81 | **54.52** | 68.78 | 50.20 | 49.14 | +0.00 |
| roll-1 | **67.87** | 54.45 | 68.78 | **50.23** | **49.17** | +0.03 |
| roll-2 | **67.87** | 54.39 | 68.78 | 50.20 | 49.15 | +0.01 |
| roll-3 | 67.74 | 54.26 | 68.79 | 50.16 | 49.11 | -0.03 |
| random derangement | 67.81 | 54.39 | 68.79 | 50.22 | **49.17** | +0.03 |
| farthest-context | 67.81 | 54.45 | **68.81** | 50.22 | 49.15 | +0.01 |
| uniform | **67.87** | 54.45 | 68.73 | 50.20 | 49.12 | -0.02 |
| static | 67.81 | 54.39 | 68.73 | 50.13 | 49.10 | -0.04 |

### Single-occurrence queries (n=1,020)

| Intervention | R1@0.5 | R1@0.7 | mAP@0.5 | mAP@0.75 | MR mAP | Δ MR mAP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| aligned | 72.65 | 60.98 | 79.01 | 61.11 | 58.97 | +0.00 |
| roll-1 | **72.75** | **61.08** | **79.06** | **61.18** | **59.04** | +0.07 |
| roll-2 | 72.55 | 60.88 | 78.93 | 61.10 | 58.96 | -0.01 |
| roll-3 | 72.45 | 60.69 | 78.93 | 61.02 | 58.87 | -0.10 |
| random derangement | 72.65 | 60.98 | 79.00 | 61.15 | 59.00 | +0.03 |
| farthest-context | 72.55 | 60.98 | 78.98 | 61.17 | 58.99 | +0.02 |
| uniform | 72.55 | 60.88 | 78.92 | 61.12 | 58.94 | -0.03 |
| static | 72.55 | 60.88 | 78.92 | 61.03 | 58.90 | -0.07 |

### Multi-occurrence queries (n=530)

| Intervention | R1@0.5 | R1@0.7 | mAP@0.5 | mAP@0.75 | MR mAP | Δ MR mAP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| aligned | 58.49 | **42.08** | 49.08 | 29.19 | 30.24 | +0.00 |
| roll-1 | 58.49 | 41.70 | 49.01 | 29.16 | 30.18 | -0.06 |
| roll-2 | **58.87** | 41.89 | 49.25 | 29.23 | 30.28 | +0.04 |
| roll-3 | 58.68 | 41.89 | **49.29** | **29.25** | **30.33** | +0.09 |
| random derangement | 58.49 | 41.70 | 49.15 | 29.18 | 30.25 | +0.01 |
| farthest-context | 58.68 | 41.89 | 49.23 | 29.13 | 30.21 | -0.03 |
| uniform | **58.87** | **42.08** | 49.13 | 29.20 | 30.24 | +0.00 |
| static | 58.68 | 41.89 | 49.12 | 29.16 | 30.22 | -0.02 |

## Conclusion

The expanded diagnosis strengthens the seed-2017 **No-Go** conclusion:

1. Full does not outperform the reproduced baseline (`49.14 = 49.14`).
2. Across all queries, every context intervention stays within `[-0.04,+0.03]`
   MR mAP of aligned.
3. On the multi-occurrence subset, where candidate-context correspondence
   should matter most, every intervention stays within `[-0.06,+0.09]`; the
   deliberately wrong roll-3 context is the highest-scoring row (`+0.09`).
4. Random derangement and farthest-context do not degrade aligned performance
   on either all queries or multi-occurrence queries.

Thus, the earlier roll-1 result was not merely a weak overall diagnostic: more
severe correspondence-breaking interventions and occurrence-stratified
evaluation still show no meaningful aligned-context advantage. These results
do not support the claim that this trained CSC branch materially uses the
correct candidate-context correspondence. This is a single-seed inference
intervention study, so the small differences should not be interpreted as
statistically significant improvements.

## Reproduction

```bash
# The baseline is already reproduced; train only the two new controls.
bash sim_detr/semantic_calibration/scripts/train_static.sh 1 2017
bash sim_detr/semantic_calibration/scripts/train_full.sh 0 2017

# Evaluate the Full best checkpoint under all eight expanded interventions.
bash sim_detr/semantic_calibration/scripts/run_counterfactuals.sh \
  /path/to/full/model_best.ckpt 0

# Unit tests.
PYTHONPATH=. /home/guoxiangyu/miniconda3/envs/sim_detr/bin/python \
  -m unittest discover -s sim_detr/semantic_calibration/tests -v
```

## Versioned artifacts

| Run | Configuration | Training log | Evaluation history | Best / intervention metrics |
| --- | --- | --- | --- | --- |
| Static seed 2017 | [`opt.json`](static_seed2017/opt.json) | [`train_history.txt`](static_seed2017/train_history.txt) | [`eval_history.txt`](static_seed2017/eval_history.txt) | [`best_val_metrics.json`](static_seed2017/best_val_metrics.json) |
| Full seed 2017 | [`opt.json`](full_seed2017/opt.json) | [`train_history.txt`](full_seed2017/train_history.txt) | [`eval_history.txt`](full_seed2017/eval_history.txt) | [`best_val_metrics.json`](full_seed2017/best_val_metrics.json) |
| Original Full counterfactuals | — | — | — | [`aligned`](full_seed2017_counterfactual/aligned/val_metrics.json), [`native`](full_seed2017_counterfactual/native/val_metrics.json), [`static`](full_seed2017_counterfactual/static/val_metrics.json), [`roll`](full_seed2017_counterfactual/roll/val_metrics.json), [`uniform`](full_seed2017_counterfactual/uniform/val_metrics.json) |
| Expanded occurrence-stratified counterfactuals | — | — | — | [`aligned`](full_seed2017_expanded_counterfactual/aligned/stratified_metrics.json), [`roll-1`](full_seed2017_expanded_counterfactual/roll-1/stratified_metrics.json), [`roll-2`](full_seed2017_expanded_counterfactual/roll-2/stratified_metrics.json), [`roll-3`](full_seed2017_expanded_counterfactual/roll-3/stratified_metrics.json), [`random`](full_seed2017_expanded_counterfactual/random-derangement/stratified_metrics.json), [`farthest`](full_seed2017_expanded_counterfactual/farthest-context/stratified_metrics.json), [`uniform`](full_seed2017_expanded_counterfactual/uniform/stratified_metrics.json), [`static`](full_seed2017_expanded_counterfactual/static/stratified_metrics.json) |

Raw predictions and checkpoints are intentionally excluded from Git. The local
best checkpoints are approximately 144–146 MB each and can be distributed via
a GitHub Release or Git LFS separately if needed.
