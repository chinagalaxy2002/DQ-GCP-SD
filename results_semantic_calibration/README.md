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

## Full-checkpoint counterfactuals

All rows below use the same Full best checkpoint. `native` is an inference
bypass after joint training and is therefore not a separately trained
baseline.

| Intervention | R1@0.5 | R1@0.7 | mAP@0.5 | mAP@0.75 | MR mAP | Δ vs aligned |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| aligned | 67.81 | **54.52** | **68.78** | 50.20 | 49.14 | +0.00 |
| context roll | **67.87** | 54.45 | **68.78** | **50.23** | **49.17** | +0.03 |
| uniform context | **67.87** | 54.45 | 68.73 | 50.20 | 49.12 | -0.02 |
| static semantics | 67.81 | 54.39 | 68.73 | 50.13 | 49.10 | -0.04 |
| native bypass | 67.81 | 54.32 | 68.72 | 50.10 | 49.08 | -0.06 |

## Conclusion

This seed-2017 pilot is a **No-Go** under the pre-registered decision rule:

1. Full does not outperform the reproduced baseline (`49.14 = 49.14`).
2. Rolling candidate contexts does not hurt performance (`+0.03` MR mAP).
3. Uniform, static, and native interventions change MR mAP by at most `0.06`.

The result does not support the claim that Sim-DETR's final semantic decision
benefits materially from the correspondence between each candidate and its
own native temporal evidence. It also does not justify proceeding to a broad
cross-architecture semantic-calibration claim from this pilot alone.

## Reproduction

```bash
# The baseline is already reproduced; train only the two new controls.
bash sim_detr/semantic_calibration/scripts/train_static.sh 1 2017
bash sim_detr/semantic_calibration/scripts/train_full.sh 0 2017

# Evaluate the Full best checkpoint under all five interventions.
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
| Full counterfactuals | — | — | — | [`aligned`](full_seed2017_counterfactual/aligned/val_metrics.json), [`native`](full_seed2017_counterfactual/native/val_metrics.json), [`static`](full_seed2017_counterfactual/static/val_metrics.json), [`roll`](full_seed2017_counterfactual/roll/val_metrics.json), [`uniform`](full_seed2017_counterfactual/uniform/val_metrics.json) |

Raw predictions and checkpoints are intentionally excluded from Git. The local
best checkpoints are approximately 144–146 MB each and can be distributed via
a GitHub Release or Git LFS separately if needed.
