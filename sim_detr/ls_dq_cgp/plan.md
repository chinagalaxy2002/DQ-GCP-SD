# LS-DQ-CGP → Sim-DETR migration plan

## Verified contracts

| Feature / operation | Reference implementation | Sim-DETR migration |
|---|---|---|
| Static text | masked mean of projected, pre-encoder tokens | same; padding and zeroed text-drop rows excluded |
| Candidate binding | native D1 cross-attention | hook `decoder.layers[0].cross_attn`; no decoder replacement |
| Local context | `A_D1 @ M_video` | same; Sim-DETR decoder memory is already video-only |
| Semantic adaptation | stopgrad(`V_q`) → RCG → BPS → FRF | same equations and default dimensions |
| Candidate matching | D2 query cosine adapted text | replaces only final `pred_logits` |
| Localization | native final span head | unchanged, including DAB reference refinement |
| Decoder depth | two layers | enforced at construction and CLI |
| Binding supervision | final Hungarian match supervises D1 attention | top-level-only `loss_ls_bind`; no routing loss |
| Ranking | semantic score | native evaluator's IoU multiplier is made constant; native IoU stays trained separately |
| Existence | optional video-query scalar | optional interface; disabled for stock QVHighlights because it has no existence labels |

## Implementation stages

1. Implement independent late-semantic module and D1 attention capture.
2. Subclass Sim-DETR without modifying baseline source or checkpoint namespaces.
3. Reproduce the native forward from projected inputs onward and retain every
   localization, saliency, auxiliary, CTC and VTC output.
4. Extend the criterion with matched D1 binding and optional supervised existence.
5. Add strict resume, safe baseline warm-start, active/static-bypass/context-roll
   inference, and canonical two-layer experiment scripts.
6. Verify tensor shapes, masks, stop-gradient, state dict, counterfactual effects,
   backward gradients, CLI parsing and compilation.

## Scientific comparison

The primary comparison must use the same Sim-DETR data, seed, two-layer decoder,
training budget and optimizer settings. The only additions are LS-DQ-CGP
parameters and `0.2 * loss_ls_bind`. Do not compare a warm-started LS model with
a baseline trained from scratch as the primary effectiveness result.
