# Published experiment artifacts

These files are the reproducible artifacts from the fixed seed-2017 Phase-1
analysis and the first-round causal training run.  No multi-seed experiment is
run.

## Phase-1

The four checkpoint analyses cover all 1,542 test queries. The compact JSON
files contain the dataset census, qid-paired bootstrap comparisons, active vs
beta-zero perturbation statistics, beta-zero vs stripped equivalence, and
CLIP-similarity strata. `phase1/phase1_summary.json` additionally contains
production-mask binding statistics (macro and micro AEC/ECR, BM, own mass and
enrichment) and official D4 MR metrics for occurrence-count subsets. Full
per-query records remain ignored because they are large.

Checkpoint modes:

```text
baseline
dq_active
dq_beta_zero
dq_stripped
```

## Causal training logs

`causal_training/` contains compact logs, variant manifests, and the complete
first-round fixed-seed-2017 comparison. The machine-readable summary is
`causal_training/summary.json`; the narrative interpretation is in
`../CAUSAL_TRAINING_RESULTS.md`. All four variants were evaluated using
`val MR-full-mAP` checkpoint selection.

## Completed Native Binding pair

`native_binding/summary.json` contains the machine-readable validation, test,
occurrence-subset, and D1/D4 mechanism comparison for the two completed runs:
the fixed Sim-DETR baseline and NativeBind `lambda=0.5`. It also records the
selected checkpoint epochs and SHA-256 identifiers. Incomplete coefficient
runs are excluded. The narrative report and paired training/evaluation
commands are in `../NATIVE_BINDING_RESULTS.md`.

`native_binding/inference_contract.json` records the matching state-dict
contract and exact output equivalence of the attention-capture wrapper.
