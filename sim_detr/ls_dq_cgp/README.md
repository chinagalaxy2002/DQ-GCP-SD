# Sim-DETR + LS-DQ-CGP (2-layer decoder)

This directory ports the late-semantic DQ-CGP feature flow to this repository
without editing the baseline implementation.

## Feature flow

```text
projected valid text tokens ── masked mean ───────────────► E_static

D1 native cross-attention ── A_D1 @ video memory ────────► V_q (stop-grad)
                                                               │
E_static + V_q ── RCG ── BPS ── FRF ─────────────────────► E_adapt_q
                                                               │
D2 query h_q ───────────────────── cosine matching ────────────┘
                                      │
                                      ▼
                              final relevance logits

D2 query h_q ── native span head ──► temporal span
```

The final relevance logits come from LS matching. D1's native classification
head remains in the one auxiliary output. Span, mask, IoU, saliency, DAB
reference refinement, D1→D2 competition, CTC and VTC paths remain intact.

Sim-DETR's stock evaluator normally ranks with `foreground_prob * iou_prob`.
This implementation supplies a constant public IoU factor so ranking is exactly
semantic-score ordering, while `native_iou_scores` is still used to train and
inspect the original IoU head.

## Loss

The final LS logits and spans determine the Hungarian assignment. For every
matched query, `loss_ls_bind` maximizes D1 attention mass inside its own GT
window. Its default coefficient is `0.2`. There is deliberately no routing
loss: the supplied LS-DQ-CGP reference does not define one.

## Train

The canonical script trains from scratch with `dec_layers=2` and mirrors the
repository's baseline settings:

```bash
bash sim_detr/ls_dq_cgp/scripts/train_ls_dq_cgp_scratch.sh
```

Useful overrides:

```bash
LS_DQ_CGP_GPU_ID=1 \
LS_DQ_CGP_SEED=2017 \
LS_DQ_CGP_RESULTS_ROOT=/path/to/results \
bash sim_detr/ls_dq_cgp/scripts/train_ls_dq_cgp_scratch.sh
```

For a diagnostic baseline warm-start, call the Python entry point with
`--init_from /path/to/plain_sim_detr.ckpt`. Use `--resume` for LS weights-only
resume, or `--resume --resume_all` for model/optimizer/scheduler/epoch resume.

## Evaluate and causal checks

```bash
LS_DQ_CGP_CKPT=/path/to/model_best.ckpt \
bash sim_detr/ls_dq_cgp/scripts/eval_ls_dq_cgp.sh
```

The script evaluates:

- `active`: candidate-specific adapted semantics;
- `static_bypass`: match D2 queries to shared `E_static`;
- `context_roll`: query `q` receives the local visual context of query `q-1`.

Strict checkpoint loading is used in all modes.

## Existence head boundary

`--use_ls_exist_head` creates a max-pooled, video-query-level existence head,
and the criterion supports BCE when `targets['exist_label']` is present. The
stock QVHighlights loader has no such label and is organized around positive
moment queries, so the canonical training entry rejects this flag instead of
silently training a degenerate all-positive classifier. Enable it only through
a GMR dataset adapter that includes empty video-query pairs and existence labels.

## Verification

```bash
/home/guoxiangyu/miniconda3/envs/sim_detr/bin/python -m compileall -q sim_detr/ls_dq_cgp
/home/guoxiangyu/miniconda3/envs/sim_detr/bin/python -m unittest discover -s sim_detr/ls_dq_cgp/tests -v
```

The suite uses only Python's standard test runner and covers the CGP equations,
stop-gradient, attention masking, binding coordinates,
counterfactuals, two-layer enforcement, state-dict cleanliness and gradients.

## Local run status

The seed-2017 QVHighlights scratch run present locally is incomplete. It
finished eight logged epochs and stopped while starting the ninth. The best
intermediate validation result is MR-full-mAP `29.45` at the seventh
evaluation; no test evaluation was run. This number is retained for run
accounting only and must not be compared with completed 200-epoch results.
