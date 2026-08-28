# Sim-DETR causal occurrence lab: Phase-1 result

This is a fixed seed-2017 existing-checkpoint analysis.  It is diagnostic
evidence only; it is not a causal-training ablation.

## Data and evaluation protocol

- Test queries: 1,542.
- Single occurrence: 1,031; multi-occurrence: 511.
- Exactly two occurrences: 266; three or more: 245.
- Clean multi-occurrence (`pairwise GT IoU < 0.1`): 511/511.
- Native evidence is captured at D1--D4 and ranked by
  `P(foreground) * sigmoid(IoU score)`.
- Binding masks now use the collated normalized `(center, width)` target spans
  and the same overlap/fallback implementation as production `L_bind`.

Checkpoints:

```text
baseline:
  test_sim_detr/hl-video_tef-exp-2026_08_20_16_41_03/model_best.ckpt
dq:
  results_dq_cgp/select_v3_parameter/beta_0p050_bind_0p20_route_0p010_center/
  runs/hl-video_tef-sim_detr_dq_cgp_v3_grid_beta_0p050_bind_0p20_route_0p010_center_seed2017-2026_08_21_08_50_54/model_best.ckpt
```

## Existing checkpoints

| mode | D4 MR-full-mAP | multi Cov@5@0.5 | D1 AEC | D4 AEC | D1 ECR | D4 ECR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 47.58 | 0.5347 | 0.6696 | 0.7928 | 0.3954 | 0.2311 |
| DQ active | 48.04 | 0.5387 | 0.6978 | 0.7657 | 0.3535 | 0.2780 |
| DQ beta-zero | 48.07 | 0.5367 | 0.6994 | 0.7683 | 0.3516 | 0.2731 |
| DQ stripped | 48.07 | 0.5367 | 0.6994 | 0.7683 | 0.3516 | 0.2731 |

The Sim-DETR baseline already reduces native evidence collision during
D1-to-D4 refinement.  The DQ checkpoint has a small D1 advantage, but does
not outperform this baseline at D4.  Therefore these existing checkpoints do
not establish the strong Soccer-GMR-style final specialization claim.

Length-normalized evidence points in the same direction.  On the multi
subset, baseline/DQ active D4 values are respectively:

```text
AEC-norm: 0.7697 / 0.8031
BM-norm:  3.6349 / 2.7974
ECR-norm: 0.2492 / 0.1993
```

The compact JSON also contains qid-macro and count-weighted micro AEC/ECR,
binding margins, own mass/enrichment, D1-own versus D4-final matching, and
official D4 MR metrics for `single`, `multi`, `two`, and `three_or_more`.

## Inference-time removal check

For the DQ checkpoint, beta-zero versus stripped comparison covers all 1,542
test queries.  The recorded deltas are exactly zero for logits, spans,
IoU-scores, saliency scores, and all D1--D4 auxiliary predictions.  Thus:

```text
DQ beta-zero == DQ stripped
```

The active-versus-beta-zero comparison has mean absolute ranking-score delta
0.00194, mean absolute span delta 0.2492 seconds, Top-5 query-ranking Jaccard
0.9868, and mean relative residual update 0.05224.

## Causal training status

The route-loss control uses the production objective
`H(W|C) - H(W)` with matched routes concatenated across the complete batch.
The first-round fixed-seed-2017 formal jobs are launched concurrently by
`scripts/run_parallel_first_round.sh` (two jobs per GPU).  They use the
release protocol and are kept separate under
`outputs/causal_training_parallel/`; no multi-seed experiment is run.
