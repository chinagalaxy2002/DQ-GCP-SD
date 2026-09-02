# QVHighlights Full + D1 Attention + Hungarian Binding

This isolated directory reproduces the Soccer-GMR experiment definition on
QVHighlights without modifying native Sim-DETR or the existing semantic
calibration implementation.

The formal configuration is Full candidate-conditioned semantic calibration,
D1 native cross-attention evidence pooling, Hungarian-matched D1 Binding Loss
with coefficient `0.2`, batch size `8`, learning rate `5e-5`, and seed `2023`.
The native QVHighlights feature paths, losses, validation evaluator, and
`MR-full-mAP` checkpoint selection are retained. This method is not
LS-DQ-CGP: it does not use semantic bases, prompts, or CGP routing modules.

Run on GPU 0 with:

```bash
bash sim_detr/qvhighlights_full_d1attn_bind/scripts/train.sh 0 2023 0.2 2
```

Extra training options can be appended to the command. For example, a smoke
run is:

```bash
bash sim_detr/qvhighlights_full_d1attn_bind/scripts/train.sh \
  0 2023 0.2 2 --debug --data_ratio 0.01
```

Formal outputs are written below `results_semantic_calibration/` with an
experiment ID beginning with `full_d1attn_bind0.2_bsz8_dec{2,4}_seed2023`.

To evaluate the validation-selected checkpoint on the local test-with-GT split
or generate the label-free official test submission, run:

```bash
python -m sim_detr.qvhighlights_full_d1attn_bind.inference \
  --checkpoint /path/to/model_best.ckpt \
  --eval_path data/highlight_test_with_gt.jsonl \
  --eval_split_name test_with_gt \
  --dec_layers 4 --gpu_id 0 \
  --output_dir /path/to/eval_test_with_gt

python -m sim_detr.qvhighlights_full_d1attn_bind.inference \
  --checkpoint /path/to/model_best.ckpt \
  --eval_path data/highlight_test_release.jsonl \
  --eval_split_name test \
  --dec_layers 4 --gpu_id 0 \
  --output_dir /path/to/submission_test
```

## Completed seed-2023 results

The two-layer run stopped at epoch 145 and selected epoch 95. The four-layer
run stopped at epoch 241 and selected epoch 212. Stop epochs follow the console
records; selected epochs are the one-based equivalents of checkpoint metadata
indices 94 and 211.

| Split / decoder layers | R1@0.5 | R1@0.7 | mAP@0.5 | mAP@0.75 | MR mAP | HL Fair mAP / Hit@1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation / 2 | **67.68** | 52.45 | **67.81** | 48.55 | 47.37 | **77.98 / 79.55** |
| Test-with-GT / 2 | **67.51** | **51.04** | **67.57** | 46.31 | 45.88 | **77.68 / 79.12** |
| Validation / 4 | 66.58 | **53.35** | 67.43 | **49.63** | **48.47** | 77.53 / 79.16 |
| Test-with-GT / 4 | 67.32 | 50.84 | 67.41 | **46.96** | **46.28** | 77.53 / 78.60 |

The four-layer model is the selected version by the headline validation and
test MR mAP, improving over the two-layer model by `+1.10` validation mAP and
`+0.40` test mAP. The two-layer model remains slightly higher on test R1@0.5
and highlight-detection metrics. Compared with the standard Sim-DETR baseline
(`49.14` validation and `47.60` test MR mAP), the selected four-layer model is
lower by `0.67` and `1.32`, respectively. It is therefore the better decoder
depth for this method, but not an improvement over Sim-DETR on QVHighlights.

## Artifact locations

| Artifact | Two-layer run | Four-layer run |
| --- | --- | --- |
| Run directory | `results_semantic_calibration/hl-video_tef-full_d1attn_bind0.2_bsz8_dec2_seed2023-2026_09_01_23_07_50/` | `results_semantic_calibration/hl-video_tef-full_d1attn_bind0.2_bsz8_dec4_seed2023-2026_09_01_23_49_58/` |
| Best validation metrics | `best_hl_val_preds_metrics.json` | `best_hl_val_preds_metrics.json` |
| Test-with-GT metrics | `eval_test_with_gt/hl_test_with_gt_submission_metrics.json` | `eval_test_with_gt/hl_test_with_gt_submission_metrics.json` |
| Test-with-GT predictions | `eval_test_with_gt/hl_test_with_gt_submission.jsonl` | `eval_test_with_gt/hl_test_with_gt_submission.jsonl` |
| Official test submission | `submission_test/hl_test_submission.jsonl` | `submission_test/hl_test_submission.jsonl` |
| Best checkpoint | `model_best.ckpt` | `model_best.ckpt` |
| Console log | `results_semantic_calibration/qv_full_d1attn_bind0.2_bsz8_dec2_seed2023.tmux.log` | `results_semantic_calibration/qv_full_d1attn_bind0.2_bsz8_dec4_seed2023.tmux.log` |

The small metric JSON files are versioned in Git. Model checkpoints, raw
predictions, official submissions, and console logs remain local because of
their size.
