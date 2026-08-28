# Published experiment artifacts

These files are the reproducible artifacts from the fixed seed-2017 Phase-1
analysis and short implementation smoke tests.

## Phase-1

The four checkpoint analyses cover all 1,542 test queries. The compact JSON
files contain the dataset census, qid-paired bootstrap comparisons, active vs
beta-zero perturbation statistics, beta-zero vs stripped equivalence, and
CLIP-similarity strata. The full per-query JSON records remain in the local
working tree under `outputs/phase1/` and are intentionally excluded from the
GitHub commit because they are large.

Checkpoint modes:

```text
baseline
dq_active
dq_beta_zero
dq_stripped
```

## Smoke logs

`smoke/` contains the real fixed-seed-2017 four-batch forward/backward logs for
`full` and `no_bind`. `SupervisionOnly` and `UnionBind` were also smoke-tested
locally; no formal 200-epoch causal-ablation result is claimed in this
delivery.

No multi-seed experiment was run.
