# Sim-DETR causal occurrence lab — Phase-1 record

This is a single seed-2017 analysis record.  It is a diagnostic result, not
the result of the causal training ablations; `NoBind`, `SupervisionOnly`, and
`UnionBind` have only been smoke-tested for a few batches.

## Data and checkpoints

- Test records: 1,542; multi-occurrence: 511; clean multi-occurrence: 511.
- Two occurrences: 266; three or more: 245.
- All test GT pairs in this file have pairwise temporal IoU below 0.1.
- Baseline: `test_sim_detr/hl-video_tef-exp-2026_08_20_16_41_03/model_best.ckpt`.
- DQ: `results_dq_cgp/select_v3_parameter/beta_0p050_bind_0p20_route_0p010_center/runs/hl-video_tef-sim_detr_dq_cgp_v3_grid_beta_0p050_bind_0p20_route_0p010_center_seed2017-2026_08_21_08_50_54/model_best.ckpt`.

## Multi-occurrence summary

| mode | D4 mAP | Cov@5@0.5 | D1 AEC | D4 AEC | D1 ECR | D4 ECR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 47.58 | 0.5348 | 0.6715 | 0.7892 | 0.3982 | 0.2313 |
| DQ active | 48.04 | 0.5387 | 0.6979 | 0.7662 | 0.3564 | 0.2765 |
| DQ beta-zero | 48.07 | 0.5367 | 0.6995 | 0.7687 | 0.3545 | 0.2705 |
| DQ stripped | 48.07 | 0.5367 | 0.6995 | 0.7687 | 0.3545 | 0.2705 |

The current Sim-DETR checkpoint therefore reproduces the deployment
equivalence (`beta-zero == stripped`) but does not by itself establish the
strong occurrence-specialization pattern observed in the other backbone.
The planned controlled training is still required before making a causal
claim about `L_bind`.

## Active versus beta-zero perturbation

- Mean absolute ranking-score change: `0.00194`.
- Mean absolute foreground-probability change: `0.00281`.
- Mean absolute IoU-score change: `0.00333`.
- Mean absolute span change: `0.2492` seconds.
- Mean Top-5 query ranking Jaccard: `0.9868`.
- Mean relative residual update: `0.05224`.
- Beta-zero versus stripped: all recorded output deltas are exactly `0`.

Machine-readable files are under `outputs/phase1/` after running
`scripts/run_phase1.sh`; generated output is intentionally ignored by the
local `.gitignore` because the per-query records are large.
