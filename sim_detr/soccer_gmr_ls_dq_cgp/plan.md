# Soccer-GMR LS-DQ-CGP implementation plan and audit

## Scope boundary

The implementation lives only in `sim_detr/soccer_gmr_ls_dq_cgp/` and reuses
the generic `sim_detr/ls_dq_cgp/` algorithm. It does not change the existing
`sim_detr/soccer_gmr_csc/` experiment. Decoder depth is not exposed as a run
choice: the configuration always writes `dec_layers = 2`, and the builder and
model independently reject any other value.

## Algorithm contract

1. Project video and text to 256 dimensions.
2. Compute `E_static` by masked mean over projected, pre-encoder text tokens.
3. Capture normalized D1 cross-attention over valid video clips.
4. Compute `V_q = A_q @ M_v` from D1 attention and encoded video memory.
5. Detach `V_q` on the semantic path.
6. Route `[V_q, E_static]` over 16 learned bases, each with six prompt slots.
7. Mean-pool the mixed prompt and use FRF to produce `E_adapt_q`.
8. Rank D2 candidates with learned-scale cosine similarity between projected
   `h_q` and `E_adapt_q`. Do not mix the native IoU score into inference rank.
9. Keep the native D2 span head unchanged.
10. Predict one existence logit from max-pooled D2 queries and expose it as
    `pred_exist_score` to the official Soccer-GMR hard gate.

## Supervision contract

- Hungarian span/GIoU/mask matching is unchanged for positive moments.
- Binding minimizes negative log D1 attention mass inside matched GT spans.
- Candidate classification supervises matched candidates as foreground and all
  remaining candidates as background.
- The trained native IoU head uses max IoU against any GT, but is never used to
  alter semantic candidate ordering.
- Existence uses video-query-level BCE.
- All-null batches have exact-zero span, GIoU, mask and Binding losses; their
  background, IoU, CTC and existence losses remain finite.
- Mixed batches retain positive matching and null-sample background/existence
  supervision in the same forward pass.

## Verification gates

- Generic LS-DQ-CGP unit tests: attention normalization, Binding, stop-gradient,
  RCG/BPS/FRF shapes, counterfactuals, two-layer enforcement and existence shape.
- Real Soccer-GMR CUDA positive+null forward/backward.
- Real Soccer-GMR CUDA all-null forward/backward.
- One-percent one-epoch training followed by the official GMR evaluator.
- Formal full-data run must finish epoch 1, write latest/best checkpoints, and
  enter epoch 2 before it is considered successfully launched.
