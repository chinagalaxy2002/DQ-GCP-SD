# DQ-CGP V3 screening grid

This directory contains nine isolated copies of the DQ-CGP V3 training code.
The architecture is fixed to one D1 insertion.  The first-stage grid is a
two-level factorial design over residual strength `beta`, binding-loss weight,
and routing-loss weight, plus the original V3 centre point.

The intended protocol is to select a configuration exclusively by validation
`MR-full-mAP` and reserve `highlight_test_with_gt.jsonl` for one final
checkpoint. The historical local artifacts do not satisfy that clean protocol:
all nine completed configurations were subsequently evaluated on the local
test-with-GT split. Their test table must therefore be reported as exploratory,
not as held-out model selection.

The completed grid's highest validation MR-mAP is `50.02` at
`beta=0.100`, binding `0.40`, and routing `0.005`. The pre-designated V3 centre
point obtains `49.66` validation MR-mAP but the highest exploratory test result
at `48.06`. The complete table and the interrupted first-wave runs are recorded
in [`EXPERIMENT_RESULTS.md`](../../../EXPERIMENT_RESULTS.md#dq-cgp-screening-grid).
