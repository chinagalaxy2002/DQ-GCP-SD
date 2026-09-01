# LS-DQ-CGP on Soccer-GMR

This directory is the isolated Soccer-GMR implementation. It does not modify
or wrap the CSC model and criterion. The only shared components are the stable
Soccer-GMR NPZ data adapter and the generic LS-DQ-CGP building blocks.

## Exact feature flow

The decoder is fixed to two layers:

```text
projected pre-encoder text --masked mean--> E_static
D1 cross-attention + encoded video -------> V_q
stopgrad(V_q) + E_static --RCG/BPS/FRF----> E_adapt_q
D2 query h_q + E_adapt_q --cosine---------> semantic candidate score s_q
D2 query h_q --native span head-----------> (center, width)
max_q(D2 queries) --MLP--------------------> p_exist
```

RCG uses 16 bases by default. Every basis contains six latent prompt positions;
BPS mixes the bases and mean-pools the positions before FRF. The semantic path
cannot update `V_q` because it is detached, while Hungarian-matched Binding
Loss directly supervises D1 attention. Candidate ranking is semantic-only; the
trained native IoU head is retained exclusively for its regression loss.

The official Soccer-GMR evaluator uses `pred_exist_score > 0.4` as its hard
G-mIoU gate. Positive-query mAP is the validation checkpoint selection metric.

## Dataset and protocol

- Soccer-GMR Standard: 4,138 train / 465 val / 1,036 test.
- CLIP video 512 + SlowFast 2304 + two TEF dimensions.
- CLIP text 512, 2-second clips, at most 75 clips and 10 DETR queries.
- 2 encoder layers and exactly 2 decoder layers.
- Seed 2023, batch size 8, learning rate 5e-5, up to 400 epochs.
- Empty GT samples receive background, downweighted native IoU/CTC, and
  video-query existence supervision; span/GIoU/Binding are exactly zero.
- Mixed positive/null batches use Hungarian matching only for existing moments.

## Train

```bash
bash sim_detr/soccer_gmr_ls_dq_cgp/scripts/train.sh 0 2023
```

Equivalent Python entry point:

```bash
python -m sim_detr.soccer_gmr_ls_dq_cgp.train \
  --exp_id ls_dq_cgp_d2_seed2023 --gpu_id 0 --seed 2023
```

The trainer refuses to overwrite a non-empty run directory. Use `--resume`
with `model_latest.ckpt` to continue an interrupted run.

## Evaluate

```bash
python -m sim_detr.soccer_gmr_ls_dq_cgp.inference \
  --checkpoint results_soccer_gmr_ls_dq_cgp/ls_dq_cgp_d2_seed2023/model_best.ckpt \
  --split test --semantic_mode aligned --gpu_id 0 \
  --output_dir results_soccer_gmr_ls_dq_cgp/ls_dq_cgp_d2_seed2023/test_best
```

`static_bypass` and `context_roll` are inference counterfactuals for verifying
that any gain depends on adapted semantics and query-context correspondence.

## Current run status

At the 2026-09-01 19:31 CST log snapshot, the seed-2023 run had completed epoch
88. Its best validation mAP so far was `18.23` at epoch 57; epoch 88 obtained
`17.42`. Training was still active and no test evaluation had been run. These
are intermediate monitoring values, not final results.
