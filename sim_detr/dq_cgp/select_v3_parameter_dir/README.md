# DQ-CGP V3 screening grid

This directory contains nine isolated copies of the DQ-CGP V3 training code.
The architecture is fixed to one D1 insertion.  The first-stage grid is a
two-level factorial design over residual strength `beta`, binding-loss weight,
and routing-loss weight, plus the original V3 centre point.

Select the next configuration exclusively by validation `MR-full-mAP`; reserve
`highlight_test_with_gt.jsonl` for the final selected checkpoint only.
