# Sim-DETR Candidate-Conditioned Semantic Calibration

This directory contains an isolated late-calibration experiment. The native
Sim-DETR source files, matcher, criterion, decoder, localization heads and
auxiliary outputs are not modified.

The wrapper captures the native transformer's decoder states and video memory,
then uses native `pred_masks` as candidate-specific evidence:

```text
native pred_masks -> normalized support -> video-memory pooling -> v_i
e_i = LayerNorm(e + MLP([LayerNorm(e), LayerNorm(v_i)]))
z_i = cosine(Wq q_i, We e_i)
pred_logits[..., 0] += softplus(scale) * z_i
```

`native`, `static`, and `full` modes are supported. The expanded context
counterfactuals are `aligned`, `roll-1`, `roll-2`, `roll-3`, deterministic
`random-derangement`, `farthest-context`, and `uniform`. The inference entry
point also evaluates `static` and writes official full-range MR metrics for
all, single-occurrence, and multi-occurrence queries. Support weights are
detached by default.

The already-reproduced native baseline should not be retrained for this pilot.
Run only:

```bash
bash sim_detr/semantic_calibration/scripts/train_static.sh 1 2017
bash sim_detr/semantic_calibration/scripts/train_full.sh 0 2017
```

After Full finishes, evaluate its best checkpoint with:

```bash
bash sim_detr/semantic_calibration/scripts/run_counterfactuals.sh /path/to/full_best.ckpt 0
```

The training entry point reuses the original QVHighlights protocol and the
original validation `MR-full-mAP` checkpoint selection.

Completed seed-2017 results, counterfactuals, logs, and the final No-Go
interpretation are recorded in
[`results_semantic_calibration/README.md`](../../results_semantic_calibration/README.md).
