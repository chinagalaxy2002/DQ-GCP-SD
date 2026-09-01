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
- Seed 2023, batch size 64, learning rate 5e-5, up to 400 epochs, patience 50.
- Native Sim-DETR architecture: 2 encoder and 4 decoder layers, 10 queries.
- GMR loss coefficients: Span L1 10, GIoU 1, candidate classification 4,
  existence BCE 1, and saliency 1.
- Shared native Sim-DETR terms remain enabled: IoU-score regression 2,
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

## Launch all three runs

Two RTX 3090 GPUs are divided as native on GPU 0 and static/full on GPU 1:

```bash
bash sim_detr/soccer_gmr_csc/scripts/train_all_tmux.sh 2023
```

Individual run:

```bash
bash sim_detr/soccer_gmr_csc/scripts/train_variant.sh full 0 2023
```

After Full training, run all correspondence interventions:

```bash
bash sim_detr/soccer_gmr_csc/scripts/run_counterfactuals.sh \
  results_soccer_gmr_csc/full_nullaware_masklogits_seed2023/model_best.ckpt 0 val
```
