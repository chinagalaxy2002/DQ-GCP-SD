# Sim-DETR CSC on Soccer-GMR

This directory adapts the isolated late Candidate-Conditioned Semantic
Calibration experiment to Soccer-GMR Standard without modifying native
Sim-DETR, its transformer, matcher, or the existing QVHighlights CSC code.

## Controlled comparison

The three from-scratch runs differ only in final candidate scoring:

| Run | Semantic residual | Candidate temporal evidence |
| --- | --- | --- |
| Native | No | No |
| Static | `cos(q_i, e)` | No |
| Full | `cos(q_i, e_i)` | Native candidate mask logits |

Full uses the same native mask field before its output sigmoid:

```text
p_i = masked_softmax(native_pre_sigmoid_mask_logit_i)
v_i = sum_t p_i(t) F_t
e_i = LN(e + MLP([LN(e), LN(v_i)]))
z_i = cosine(Wq q_i, We e_i)
foreground_logit_i += softplus(scale) * z_i
```

Only the final foreground logit changes. Spans, masks, IoU scores, saliency,
auxiliary outputs, decoder states, and native matching remain unchanged.

## Soccer-GMR protocol

- Standard split: 4,138 train / 465 val / 1,036 test.
- Features: CLIP 512 + SlowFast 2304; CLIP text 512; 2-second clips.
- Seed 2023, batch size 8, evaluation batch size 8, learning rate 5e-5,
  up to 400 epochs, patience 50. The reported run was stopped manually after
  the validation plateau; its validation-selected best checkpoint is used below.
- Native Sim-DETR architecture: 2 encoder and 4 decoder layers, 10 queries.
- GMR loss coefficients: Span L1 10, GIoU 1, candidate classification 4,
  existence BCE 1, and saliency 1.
- This run uses mask-IoU matching/loss coefficient 6 (`set_cost_mask=6`,
  `mask_loss_coef=6`). Shared native Sim-DETR terms remain enabled: IoU-score regression 2,
  CTC 0.5, and VTC 0.3. Foreground matched candidates retain unit weight;
  positive-sample background candidates use `eos_coef=0.1`, and all-null
  candidates use 0.05. Null IoU and CTC terms are likewise downweighted to
  0.05 and 0.1. VTC uses only positives in groups of at most eight.
- Best checkpoint is selected by official positive-query validation `mAP`.
- Full official GMR metrics are saved at every validation epoch.

The dataset contains null, single, and multi-moment queries. Following the
Moment-DETR-GMR code in DQ-CGP-main, every run receives the same max-pooled
decoder-query existence adapter and BCE supervision. Null samples use empty
Hungarian targets, so every candidate is also supervised as background; span,
GIoU, and positive saliency losses are exact zero when a batch is all-null.
Positive samples use a null-safe pairwise saliency margin over the GT temporal
union. At inference, `pred_exist_score` is the adapter probability and applies
the same soft gate below 0.3. The GMR adapter is identical in Native, Static,
and Full; CSC still changes only the candidate foreground logit. CTC uses an
all-zero temporal target for null samples, while VTC is computed only among
 positive query-video pairs to avoid treating absent events as positive
 alignment pairs.

## Seed-2023 training and test results

The following table is generated from the three runs with
`nullaware_maskloss6_bsz8` and their validation-selected `model_best.ckpt`.
Checkpoint selection uses validation `mAP`; test labels are used only for the
final test evaluation.

| Method | Best val epoch | Best val mAP | Test mAP | Test mR@1 | Test mR@3 | Test mR@5 | Test mIoU@1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Native Sim-DETR | 137 | 22.59 | 20.73 | 11.55 | 24.14 | 31.51 | 28.95 |
| Static semantic calibration | 160 | **24.30** | **21.09** | **12.41** | **24.49** | 31.10 | **31.45** |
| Candidate-conditioned calibration | 139 | 22.48 | 19.43 | 11.57 | 22.06 | 28.39 | 28.36 |

The full official test metrics are stored in each run's
`test_best/*_metrics.json`. Static calibration improves over Native on the
test split by `+0.36` mAP. Full does not improve over Static in this seed, so
this run does not support an incremental candidate-conditioned semantic gain on
Soccer-GMR.

## Full-checkpoint correspondence counterfactuals

These are inference-only interventions on the Full checkpoint. Native logits,
spans, masks, IoU scores, saliency, and existence outputs remain fixed; only
the semantic context used by the final foreground residual is changed.

| Full checkpoint intervention | Test mAP | Test mR@1 | Test mR@3 | Test mR@5 | Test mIoU@1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| aligned | 19.43 | 11.57 | 22.06 | 28.39 | 28.36 |
| roll-1 | 18.83 | 10.48 | 21.92 | 28.08 | 27.28 |
| roll-2 | 19.48 | 11.68 | 21.77 | 28.49 | 28.52 |
| roll-3 | 19.23 | 11.40 | 21.61 | 27.87 | 28.37 |
| random derangement | 19.02 | 10.86 | 21.62 | 28.26 | 27.68 |
| farthest context | 19.53 | 11.84 | 21.95 | 28.30 | 28.63 |
| uniform context | 19.34 | 11.40 | 22.08 | 28.35 | 28.39 |
| static semantics | 19.47 | 11.61 | 21.93 | 28.19 | 28.25 |
| native bypass | 19.38 | 11.42 | 22.08 | 28.28 | 28.28 |

The aligned-to-roll-1 drop is small (`-0.60` mAP), while roll-2, roll-3,
farthest-context, uniform, static, and native bypass are close to aligned.
This indicates weak sensitivity to candidate-context correspondence for this
checkpoint; the stronger observation is the Static-versus-Native comparison.

## Launch all three runs

Two RTX 3090 GPUs are divided as native on GPU 0 and static/full on GPU 1:

```bash
SOCCER_GMR_RUN_TAG=nullaware_maskloss6_bsz8 \
bash sim_detr/soccer_gmr_csc/scripts/train_all_tmux.sh 2023
```

Individual run:

```bash
bash sim_detr/soccer_gmr_csc/scripts/train_variant.sh full 0 2023
```

After Full training, run all correspondence interventions:

```bash
bash sim_detr/soccer_gmr_csc/scripts/run_counterfactuals.sh \
  results_soccer_gmr_csc/full_nullaware_maskloss6_bsz8_seed2023/model_best.ckpt 0 test
```

The implementation is isolated to this directory plus the generic
`sim_detr/semantic_calibration/` wrapper. The native `sim_detr/model.py`,
`transformer.py`, and `matcher.py` are not modified.
