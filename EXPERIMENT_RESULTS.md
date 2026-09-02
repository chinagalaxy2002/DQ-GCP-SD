# Consolidated Experiment Record

This document reconciles the versioned reports with the local training logs,
metric JSON files, checkpoints, and interrupted runs available through
2026-09-02. It separates completed results from diagnostics and unfinished
experiments so that the latter are not cited as final numbers.

The matching machine-readable snapshot is
[`results/experiment_inventory_2026_09_01.json`](results/experiment_inventory_2026_09_01.json).

## Reporting conventions

- `Completed` means that the planned training protocol finished, or that a
  validation-selected checkpoint received the intended final evaluation.
- `Diagnostic` means an inference intervention or mechanism analysis on a
  fixed checkpoint. It is not an independently trained effectiveness result.
- `Incomplete` means that a run produced intermediate metrics but did not
  finish its planned budget.
- `Running` values are timestamped snapshots and must not be reported as final.
- Earlier QVHighlights experiments use seed 2017. The Full CSC + D1-attention
  + Hungarian-binding depth comparison uses seed 2023 on both datasets. No
  multi-seed claim is made.

## Evaluation provenance

Two QVHighlights test paths are present locally. The standard submission
evaluator reports the released Sim-DETR baseline at `47.60` MR-full-mAP and
DQ-CGP V3 at `48.06`. The causal attention-capture analyzer reports the same
baseline checkpoint at `47.58` and the DQ checkpoint at `48.04`. The small
differences come from different saved prediction/evaluation paths; deltas must
be computed within one path rather than mixing the two tables.

## QVHighlights: completed effectiveness runs

All DQ-CGP rows below use two encoder layers, four decoder layers, seed 2017,
batch size 32, and a 200-epoch from-scratch budget unless noted otherwise.

| Method | Configuration | Best val MR-mAP | Test MR-mAP | Status |
| --- | --- | ---: | ---: | --- |
| Sim-DETR baseline | Native model | 49.14 | 47.60 | Completed reference |
| DQ-CGP V3 | D1 insertion; beta 0.05; bind 0.20; route 0.01 | 49.66 | **48.06** | Completed release |
| Dual independent | Independent adapters after D1 and D3 | 49.53 | 46.85 | Completed |
| Tied dual | One shared adapter after D1 and D3 | 49.32 | 46.88 | Completed |
| Tied all | One shared adapter after D1, D2, and D3 | 49.64 | 46.80 | Completed |
| NativeBind | Native D1 attention binding, lambda 0.5 | **49.67** | 47.25 | Completed mechanism control |
| LCB Full | Acquire at D1, preserve through D2-D4 | 49.19 | 47.25 | Completed mechanism control |
| Static semantic calibration | Shared text semantics | 48.95 | not run | Completed validation study |
| Candidate-conditioned calibration | Candidate mask-pooled semantics | 49.14 | not run | Completed validation study |
| Full CSC + D1 attention + binding, D2 | D1 evidence; bind 0.20; seed 2023; batch 8 | 47.37 | 45.88 | Completed depth comparison |
| Full CSC + D1 attention + binding, D4 | D1 evidence; bind 0.20; seed 2023; batch 8 | 48.47 | 46.28 | Completed depth comparison |

The DQ-CGP V3 checkpoint was selected at epoch 103 within its run. It was the
pre-designated release configuration. A later exploratory grid contained a
configuration with a higher validation score, so V3 should not be described as
the validation winner of that later grid.

For the new Full CSC depth comparison, D2 stopped early at epoch 145 and its
best checkpoint was selected at epoch 95. D4 stopped early at **epoch 241**
and its best checkpoint was selected at epoch 212. D4 improves over D2 by
`1.10` validation MR-mAP and `0.40` test MR-mAP, so D4 is the preferred depth
for this method on QVHighlights. However, its `48.47/46.28` validation/test
MR-mAP remains below the standard Sim-DETR baseline (`49.14/47.60`), so this
run does not establish a QVHighlights improvement.

### DQ-CGP screening grid

The local directory contains an interrupted first wave ending around epochs
49-59 and a second wave in which all nine configurations reached 200 epochs.
Only the completed second wave is listed here.

| beta | Binding weight | Routing weight | Best val MR-mAP | Test MR-mAP |
| ---: | ---: | ---: | ---: | ---: |
| 0.025 | 0.10 | 0.005 | 49.31 | 47.68 |
| 0.025 | 0.10 | 0.020 | 49.59 | 46.73 |
| 0.025 | 0.40 | 0.005 | 49.16 | 47.63 |
| 0.025 | 0.40 | 0.020 | 49.72 | 47.49 |
| 0.050 | 0.20 | 0.010 | 49.66 | **48.06** |
| 0.100 | 0.10 | 0.005 | 49.03 | 47.40 |
| 0.100 | 0.10 | 0.020 | 49.69 | 47.82 |
| 0.100 | 0.40 | 0.005 | **50.02** | 47.88 |
| 0.100 | 0.40 | 0.020 | 48.93 | 47.66 |

All nine completed configurations were evaluated on the local test-with-GT
split. These test results are therefore exploratory comparisons, not a clean
held-out model-selection study. The highest validation result is `50.02`,
whereas the highest test result is the pre-designated V3 configuration at
`48.06`.

### Inference-time beta sweep

This diagnostic changes only the residual coefficient of the trained V3
checkpoint at inference time.

| Inference beta | Val MR-mAP |
| ---: | ---: |
| 0.00 | 49.59 |
| 0.05 | **49.66** |
| 0.10 | 49.53 |
| 0.20 | 49.45 |
| 0.50 | 49.16 |
| 1.00 | 48.43 |
| 1.50 | 47.87 |
| 2.00 | 47.56 |
| 2.50 | 47.29 |
| 3.00 | 47.20 |

The trained value `0.05` is best in this sweep. Increasing the inference
residual beyond 0.5 progressively reduces validation MR-mAP.

## QVHighlights: occurrence-binding diagnostics

### Existing-checkpoint Phase 1

| Mode | Test D4 mAP | Multi Cov@5@0.5 | D1 AEC | D4 AEC | D1 ECR | D4 ECR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 47.58 | 0.5347 | 0.6696 | 0.7928 | 0.3954 | 0.2311 |
| DQ active | 48.04 | 0.5387 | 0.6978 | 0.7657 | 0.3535 | 0.2780 |
| DQ beta-zero | 48.07 | 0.5367 | 0.6994 | 0.7683 | 0.3516 | 0.2731 |
| DQ stripped | 48.07 | 0.5367 | 0.6994 | 0.7683 | 0.3516 | 0.2731 |

Beta-zero and stripped outputs are exactly equal. The active DQ checkpoint has
a small D1 ownership advantage but does not exceed the baseline at D4.

### Controlled causal training

| Variant | Bind | Route | Residual injection | Test D4 mAP | D1/D4 AEC | D1/D4 ECR |
| --- | ---: | ---: | :---: | ---: | ---: | ---: |
| FullRepro | 0.20 | 0.01 | yes | 47.41 | 0.6678 / 0.7891 | 0.4241 / 0.2559 |
| NoBind | 0.00 | 0.01 | yes | **47.65** | 0.6281 / 0.7311 | 0.4650 / 0.3290 |
| SupervisionOnly | 0.20 | 0.00 | no | 46.68 | 0.6167 / 0.7619 | 0.5114 / 0.2863 |
| UnionBind | 0.20 | 0.01 | yes | 46.76 | 0.7017 / 0.7890 | 0.3492 / 0.2311 |

These fixed-seed results show that the binding path changes internal ownership,
but they do not establish that candidate-specific matching is better than a GT
union target. Mechanism ordering and final retrieval ordering also differ.

### NativeBind and LCB

NativeBind at lambda 0.5 raises validation MR-mAP from `49.14` to `49.67` but
reduces test MR-mAP from `47.58` to `47.25` in the causal analyzer. On the 511
multi-occurrence test queries, D1 AEC improves from `0.6696` to `0.8480` and D1
ECR falls from `0.3954` to `0.1490`. At D4, however, AEC falls to `0.7322` and
ECR rises to `0.3250`.

LCB extends the objective across D2-D4. It reaches D4 AEC `0.9555`, D4 ECR
`0.1480`, and D1-to-D4 persistence `0.9788`, but its test MR-mAP remains
`47.25`. NativeBind and LCB therefore support the ownership mechanism without
supporting a final test-retrieval improvement.

Additional NativeBind coefficients are incomplete: lambda 2.0 stopped while
entering epoch 45 and has an intermediate best validation MR-mAP of `46.23`;
lambda 1, 5, 10, and 20 stopped during epoch 1 and have no usable result.

## QVHighlights: semantic calibration

Static semantic calibration obtains `48.95` validation MR-mAP. The Full
candidate-conditioned model obtains `49.14`, matching the existing baseline.
On the same Full checkpoint, eight aligned or correspondence-breaking context
interventions remain within `49.10-49.17` on all validation queries and within
`30.18-30.33` on the multi-occurrence subset. These results do not support the
claim that the trained branch materially uses the correct candidate-context
correspondence.

## Soccer-GMR: completed comparison

The formal comparison uses seed 2023, batch size 8, two encoder and four
decoder layers, mask loss/matching coefficient 6, null-aware supervision, and
validation mAP for checkpoint selection.

| Method | Best val epoch | Best val mAP | Test mAP | Test mR@1 | Test mR@3 | Test mR@5 | Test mIoU@1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Native | 137 | 22.59 | 20.73 | 11.55 | 24.14 | **31.51** | 28.95 |
| Static | 160 | **24.30** | **21.09** | **12.41** | **24.49** | 31.10 | **31.45** |
| Candidate-conditioned Full | 139 | 22.48 | 19.43 | 11.57 | 22.06 | 28.39 | 28.36 |

Static improves test mAP over Native by `0.36`. Full does not improve over
either Static or Native. Full-checkpoint context interventions range from
`18.83` to `19.53` test mAP; the farthest-context intervention is slightly
higher than aligned (`19.53` versus `19.43`). The evidence for correct
candidate-context correspondence is therefore weak in this run.

### Full CSC + D1-attention evidence + Hungarian binding

This is a separate seed-2023 depth comparison and is **not LS-DQ-CGP**. Both
runs use batch size 8 and binding coefficient 0.2.

| Decoder depth | Stop epoch | Selected epoch | Val mAP | Test mAP | Test mR@1 | Test mR@3 | Test mR@5 | Test mIoU@1 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 198 | 148 | 23.64 | **22.03** | **12.79** | **26.00** | **32.51** | **30.89** |
| 4 | 196 | 146 | **23.72** | 20.75 | 11.85 | 23.87 | 31.20 | 29.66 |

D4 is only `0.08` higher on validation mAP but `1.28` lower on test mAP, so D2
is the preferred Soccer-GMR depth. Its `22.03` test mAP is the highest result
in the current seed-2023 Soccer-GMR table, but the gain cannot be attributed to
one component because evidence pooling, binding, and decoder depth were not
varied in a controlled factorial ablation.

### Protocol-development runs

These validation-only runs document how the Soccer-GMR protocol evolved. They
must not be compared as if only the semantic method changed.

| Protocol | Native val mAP | Static val mAP | Full val mAP | Note |
| --- | ---: | ---: | ---: | --- |
| Early, batch 64, mask loss 0, no null-aware weighting | 11.32 | 6.41 | 7.71 | Stopped early |
| Null-aware, batch 64, mask loss 0 | 8.02 | 4.85 | 7.05 | Stopped early |
| Null-aware, batch 64, mask loss 6 | 3.25 | 3.43 | 3.96 | Only about 10-12 epochs |
| Formal, batch 8, four decoder layers, mask loss 6 | 22.59 | 24.30 | 22.48 | Completed comparison |
| Batch 8, two decoder layers, mask loss 6 | not run | 22.41 | 20.88 | Validation only; no matched Native run |

For the short batch-64 Static mask-loss-6 run, the history reaches `3.43` but
the saved `best_val_metrics.json` contains `2.68`. The history value is kept
only as a diagnostic, and the run is excluded from formal conclusions.

## Incomplete and running LS experiments

### QVHighlights LS-DQ-CGP

The two-layer LS-DQ-CGP run completed eight logged epochs and stopped while
starting the ninth. Its intermediate best validation MR-mAP is `29.45` at the
seventh evaluation. It has no test result and is not a completed comparison.

### Soccer-GMR LS-DQ-CGP completed run and counterfactuals

The two-layer Soccer-GMR LS-DQ-CGP run completed 198 epochs (early stopped with patience 50) and selected epoch 148. Its official validation mAP is `21.06` and test mAP is `18.57`.

| Semantic intervention | Test mAP | Test mR@1 | Test mR@3 | Test mR@5 | Test mIoU@1 | G-mIoU@1 | Delta vs aligned |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Aligned (standard)** | **18.57** | **11.45** | **21.40** | **26.88** | **28.40** | **38.96** | — |
| **Static Bypass** | 17.89 | 10.74 | 21.31 | 25.87 | 27.52 | 38.14 | $-0.68$ ($-3.7\%$) |
| **Context Roll** | **12.70** | **5.14** | **15.07** | **22.43** | **15.75** | **33.37** | **$-5.87$ ($-31.6\%$)** |

The results confirm that:
1. Static bypass drops performance, demonstrating active use of candidate-specific semantic adaptation.
2. Context roll causes a massive `5.87` test mAP drop ($-31.6\%$) and `6.31` mR@1 drop ($-55.1\%$), establishing that grounding accuracy strictly depends on correct localized evidence binding.

Artifacts and compact summaries are versioned in `results_soccer_gmr_ls_dq_cgp/ls_dq_cgp_d2_seed2023/` and `results/soccer_gmr_ls_dq_cgp_counterfactuals_summary.json`.

## Artifact policy

Large checkpoints, TensorBoard event files, raw predictions, and full console
logs remain excluded from Git. The repository versions compact metrics,
configuration records, the completed causal summaries, and this reconciled
inventory. This avoids publishing machine-local paths and multi-gigabyte run
directories while retaining the claims and their numerical provenance.

## Versioned evidence index

- Released V3 validation/test metrics:
  [`results/v3_d1_val_metrics.json`](results/v3_d1_val_metrics.json) and
  [`results/v3_d1_test_metrics.json`](results/v3_d1_test_metrics.json).
- Existing-checkpoint causal analysis:
  [`causal_occurrence_lab/PHASE1_RESULTS.md`](causal_occurrence_lab/PHASE1_RESULTS.md)
  and
  [`causal_occurrence_lab/results/phase1/phase1_summary.json`](causal_occurrence_lab/results/phase1/phase1_summary.json).
- Controlled causal training:
  [`causal_occurrence_lab/CAUSAL_TRAINING_RESULTS.md`](causal_occurrence_lab/CAUSAL_TRAINING_RESULTS.md)
  and
  [`causal_occurrence_lab/results/causal_training/summary.json`](causal_occurrence_lab/results/causal_training/summary.json).
- NativeBind:
  [`causal_occurrence_lab/NATIVE_BINDING_RESULTS.md`](causal_occurrence_lab/NATIVE_BINDING_RESULTS.md)
  and
  [`causal_occurrence_lab/results/native_binding/summary.json`](causal_occurrence_lab/results/native_binding/summary.json).
- LCB:
  [`Layeerconsistentbinding/results/lcb_full_seed2017/README.md`](Layeerconsistentbinding/results/lcb_full_seed2017/README.md)
  and
  [`ownership_summary.json`](Layeerconsistentbinding/results/lcb_full_seed2017/eval_test/ownership_summary.json).
- QVHighlights semantic calibration:
  [`results_semantic_calibration/README.md`](results_semantic_calibration/README.md).
- Soccer-GMR semantic calibration:
  [`sim_detr/soccer_gmr_csc/README.md`](sim_detr/soccer_gmr_csc/README.md).
- Completed Full CSC + D1-attention + Hungarian-binding depth comparison:
  [`results/full_d1attn_binding_seed2023_summary.json`](results/full_d1attn_binding_seed2023_summary.json).

The beta sweeps, early Soccer-GMR runs, interrupted LS run, and historical
snapshots were reconstructed from ignored local output directories. Their
compact values are versioned in the machine-readable inventory, while the raw
logs remain local.
