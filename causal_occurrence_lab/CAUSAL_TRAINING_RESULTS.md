# Sim-DETR causal training results

The later parameter-free Baseline vs NativeBind `lambda=0.5` experiment is
reported separately in
[`NATIVE_BINDING_RESULTS.md`](NATIVE_BINDING_RESULTS.md). Its delivery includes
the completed validation/test results, a one-command paired training launcher,
the common evaluation launcher, and machine-readable artifacts. Incomplete
NativeBind coefficient runs are not included.

This is the first-round controlled training on QVHighlights using one fixed
seed (`2017`).  No multi-seed experiment was run.  The four jobs used the
same release protocol: four decoder layers, two encoder layers, batch size
32, learning rate `1e-4`, learning-rate drop at epoch 100, 200 maximum
epochs, VTC `0.3`, CTC `0.5`, and checkpoint selection by validation
`MR-full-mAP`.

The mechanism metrics below use the production normalized target-span
geometry, D4-final Hungarian assignment tracked back through D1, and the
official Sim-DETR ranking/post-processing protocol.  All 511 multi-occurrence
queries are also clean under pairwise GT IoU `< 0.1`.

## First-round comparison

| Variant | Bind | Route | Inject | Test D4 mAP | Cov@5@0.5 | D1 AEC | D4 AEC | D1 ECR | D4 ECR |
| --- | ---: | ---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FullRepro | 0.20 | 0.01 | yes | 47.41 | 0.5347 | 0.6678 | 0.7891 | 0.4241 | 0.2559 |
| NoBind | 0.00 | 0.01 | yes | 47.65 | 0.5305 | 0.6281 | 0.7311 | 0.4650 | 0.3290 |
| SupervisionOnly | 0.20 | 0.00 | no | 46.68 | 0.5320 | 0.6167 | 0.7619 | 0.5114 | 0.2863 |
| UnionBind | 0.20 | 0.01 | yes | 46.76 | 0.5315 | 0.7017 | 0.7890 | 0.3492 | 0.2311 |

For reference, the existing Phase-1 baseline checkpoint reports test D4 mAP
`47.58`, multi Cov@5@0.5 `0.5347`, D1/D4 AEC `0.6696/0.7928`, and D1/D4 ECR
`0.3954/0.2311`.  It is retained as an existing-checkpoint diagnostic and
is not presented as a retrained causal-harness baseline.

## Interpretation

- `FullRepro` versus `NoBind` supports a useful effect of the binding-related
  training path in this fixed run: removing the binding term reduces D1/D4
  AEC and increases D1/D4 ECR.
- `SupervisionOnly` is not equivalent to `FullRepro` on Sim-DETR.  Its
  residual is disabled during both training and evaluation, and its final
  evidence metrics and test mAP are lower.  Therefore this run does not
  establish that matched binding supervision alone is sufficient for the
  complete Full result.
- `UnionBind` is strong on the reported D4 evidence metrics and slightly
  lowers D4 ECR relative to `FullRepro`.  Consequently, this first round does
  **not** prove that candidate-specific matched targets outperform a global GT
  union on Sim-DETR.  The candidate-specific assignment claim should remain
  qualified until an additional control or repeated protocol resolves this
  result.
- Multi-occurrence coverage changes are small (`0.5305`--`0.5347`), and the
  benchmark mAP ordering does not match the evidence-binding ordering.  The
  current results therefore separate the internal-evidence mechanism from a
  simple claim that DQ improves multi-occurrence coverage or overall mAP.

## Reproducibility artifacts

- Training logs and `variant.json` files are in `results/causal_training/`.
- The machine-readable summary is `results/causal_training/summary.json`.
- The full uncommitted checkpoints remain under
  `outputs/causal_training_parallel/` and are excluded from Git by design.
- The four jobs were run concurrently in tmux: `dqgcp_full`,
  `dqgcp_no_bind`, `dqgcp_supervision`, and `dqgcp_union`.
