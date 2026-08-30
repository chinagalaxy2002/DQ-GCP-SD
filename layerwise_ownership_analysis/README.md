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
