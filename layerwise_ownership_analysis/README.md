# Inference-time beta sweep

`beta_sweep_eval.py` evaluates one trained DQ-CGP checkpoint repeatedly while
changing only the fixed residual coefficient `model.query_cgp.beta`. It calls
the repository's strict checkpoint loader and official evaluation pipeline.

Example:

```bash
conda run -n sim_detr python -m layerwise_ownership_analysis.beta_sweep_eval \
  --checkpoint /path/to/model_best.ckpt \
  --betas 0 0.05 0.1 0.2 \
  --output-root layerwise_ownership_outputs/beta_sweep_val
```

This is a test-time sensitivity/causal ablation. It is not equivalent to
training four independent models with the corresponding beta values.

## Completed validation sweep

The local sweep uses the released V3 checkpoint, whose trained beta is 0.05.

| Inference beta | Val MR-full-mAP |
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

The trained value is best in this inference-only sweep. Larger residuals
progressively reduce validation MR-mAP; this does not establish how models
trained independently at those beta values would behave.
