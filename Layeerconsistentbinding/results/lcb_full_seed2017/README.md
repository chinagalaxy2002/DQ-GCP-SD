# LCB Full Run — Seed 2017

This directory is the publishable record for the `lcb_full` Layer-Consistent
Binding (Acquire → Preserve) run. The implementation, objective, and exact
launch commands are documented in [the parent method README](../../README.md).

## Run configuration

The run uses four decoder layers, two encoder layers, batch size 32, learning
rate `1e-4`, and seed 2017. Its four LCB loss coefficients are `0.5` (D1
binding), `0.1` (late binding), `0.1` (ownership consistency), and `0.1`
(anti-washout); the anti-washout margin is `0.05`. The complete machine-readable
configuration is in `variant.json`.

## Validation result

`model_best.ckpt` was selected at epoch 144 by validation MR-full-mAP. The
checkpoint itself is intentionally not versioned because it is a large binary
artifact. On the QVHighlights validation split, the selected checkpoint obtained:

| Metric | Value |
| --- | ---: |
| MR-full-mAP | 49.19 |
| MR-full-mAP@0.5 | 69.13 |
| MR-full-mAP@0.75 | 50.82 |
| MR-full-R1@0.5 | 68.13 |
| MR-full-R1@0.7 | 53.35 |
| HL-min-Fair mAP / Hit@1 | 77.55 / 79.74 |
| HL-min-Good mAP / Hit@1 | 66.51 / 77.61 |
| HL-min-VeryGood mAP / Hit@1 | 40.76 / 66.32 |

The complete best-checkpoint and final-checkpoint metrics are saved as
`best_hl_val_preds_metrics.json` and `latest_hl_val_preds_metrics.json`.
`best_hl_val_preds.jsonl` contains the best-checkpoint validation predictions.

## Test ownership analysis

The selected checkpoint was evaluated on 1,542 test records. Across 2,691
matched trajectories, D1 → D4 persistence is 0.9788; the D1 → D4 washout-drop
rate is 0.2404. The compact, reproducible summary is in
`eval_test/ownership_summary.json`; D1–D4 test predictions are included in the
same folder.

## Provenance and included artifacts

The files in this directory are copied unchanged from
`outputs/lcb_full_seed2017`, except `eval_test/ownership_summary.json`, which is
a compact summary derived from the original 50.7 MB per-record analysis file.
The committed archive contains:

- `variant.json`: run configuration.
- `lcb_train.log`, `train.log.txt`, and `eval.log.txt`: training and validation logs.
- `best_hl_val_preds_metrics.json`, `latest_hl_val_preds_metrics.json`, and
  `best_hl_val_preds.jsonl`: validation outputs.
- `eval_test/d1_submission.json` through `d4_submission.json`: test predictions
  from each decoder layer.
- `eval_test/ownership_summary.json`: aggregate LCB ownership metrics.

Not included: model checkpoints, TensorBoard events, tmux output, duplicate
generic evaluation files, and the full per-record analysis. These are either
large binary/redundant artifacts or contain machine-local paths. They remain
available in the original output directory.
