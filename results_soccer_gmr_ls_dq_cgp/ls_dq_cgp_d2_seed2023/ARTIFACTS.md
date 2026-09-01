# Soccer-GMR LS-DQ-CGP formal-run artifacts

This directory records the seed-2023, decoder-2 formal run. Checkpoints are
kept locally but are excluded from Git because each is about 104 MB.

- `opt.json`: complete serialized run configuration.
- `history.jsonl`: per-epoch training and official validation metrics.
- `best_val_metrics.json`: metrics for the validation-selected epoch 148.
- `latest_val_metrics.json`: metrics from the final logged epoch 198.
- `test_best/test_aligned_metrics.json`: official test metrics from the best
  validation checkpoint.
- `test_best/test_aligned_predictions.jsonl`: all 1,036 test predictions.
- `ls_dq_cgp_d2_seed2023.log`: complete formal training console log, including
  the explicit Early Stopping record.
- `test_eval.log`: test-evaluation console log.

The official test summary is mAP `18.57`, mR@1 `11.45`, mR@3 `21.40`, mR@5
`26.88`, mIoU@1 `28.40`, and AUROC `74.91`.
