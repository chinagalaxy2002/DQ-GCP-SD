# Native Hungarian Binding on Sim-DETR

This report compares only two fully trained, validation-selected runs:

1. Vanilla Sim-DETR (the completed seed-2017 baseline).
2. The same Vanilla Sim-DETR trained with Native Hungarian Binding at
   `lambda_native_bind=0.5` for 200 epochs.

Incomplete coefficient runs are intentionally excluded from every table and
conclusion below.

## Method

Sim-DETR passes video-only memory to its decoder, so the native D1
cross-attention already has the required query-to-clip form
$A^{D1}\in\mathbb{R}^{B\times Q\times T}$. During training, the final D4
predictions are matched to ground-truth windows with Sim-DETR's unchanged
Hungarian matcher. For each final match $(j,k)$, NativeBind measures the D1
attention mass assigned by query $j$ to the clips overlapping ground-truth
window $k$:

$$
m_{jk}=\sum_{t\in GT_k} A^{D1}_{j,t}, \qquad
L_{\text{native-bind}}=-\frac{1}{|\mathcal{M}|}
\sum_{(j,k)\in\mathcal{M}}\log(m_{jk}+\epsilon).
$$

The completed configuration uses

$$
L=L_{\text{Sim-DETR}}+0.5L_{\text{native-bind}}.
$$

The implementation captures the weights returned by the existing D1
`cross_attn` module without detaching them, normalizes over valid video clips,
and applies the matched loss in the criterion. It does not introduce a basis,
router, FRF, residual update, prediction head, or alternative matcher.

Both saved checkpoints have 287 state-dict keys and 11,934,706 state
elements, with identical key sets and tensor shapes and no `query_cgp.*`
keys. NativeBind therefore adds zero model parameters. On a real two-example
batch, installing the attention-capture wrapper produced four tensors of
shape `[2,10,75]` and changed none of 22 compared output tensors
(`max_abs_difference=0.0`). The binding machinery is training-only; evaluation
uses `--mode baseline`.

## Training protocol

The two runs use seed 2017, four decoder layers, two encoder layers, batch
size 32, learning rate `1e-4`, learning-rate drop at epoch 100, 200 epochs,
VTC coefficient `0.3`, CTC coefficient `0.5`, and validation
`MR-full-mAP` for checkpoint selection. The baseline best checkpoint is epoch
114; the NativeBind best checkpoint is epoch 143.

### Validation results

| Method | R1@0.5 | R1@0.7 | mAP@0.5 | mAP@0.75 | mAP Avg. | Fair HL | Good HL | VeryGood HL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Sim-DETR baseline | 68.32 | 53.81 | 69.03 | 50.77 | 49.14 | 77.90 | 66.74 | 40.97 |
| NativeBind (`lambda=0.5`) | **69.16** | **54.39** | **70.03** | **50.88** | **49.67** | **78.21** | **67.04** | **41.19** |
| NativeBind - baseline | +0.84 | +0.58 | +1.00 | +0.11 | **+0.53** | +0.31 | +0.30 | +0.22 |

## Test results

Both best checkpoints were re-evaluated on the same 1,542-query local
test-with-GT split with `analyze_checkpoints.py`, native D1-D4 attention
capture, identical ranking and post-processing, and Vanilla Sim-DETR mode.

| Method | R1@0.5 | R1@0.7 | mAP@0.5 | mAP@0.75 | mAP Avg. | Fair HL | Good HL | VeryGood HL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Sim-DETR baseline | 66.86 | **51.56** | 67.70 | **48.70** | **47.58** | 77.12 | 65.95 | 39.86 |
| NativeBind (`lambda=0.5`) | **67.06** | 51.10 | **67.85** | 48.07 | 47.25 | **77.79** | **66.45** | **40.19** |
| NativeBind - baseline | +0.20 | -0.46 | +0.15 | -0.63 | **-0.33** | +0.67 | +0.50 | +0.33 |

NativeBind improves all three test highlight-detection mAP values, but it does
not improve the headline test moment-retrieval average mAP.

### Occurrence-count subsets

| Test subset | Queries | Metric | Baseline | NativeBind | Delta |
| --- | ---: | --- | ---: | ---: | ---: |
| Single occurrence | 1,031 | D4 mAP Avg. | **57.24** | 56.70 | -0.54 |
| Multi occurrence | 511 | D4 mAP Avg. | 28.09 | **28.18** | +0.09 |
| Multi occurrence | 511 | D4 R1@0.5 | 58.51 | **59.30** | +0.79 |
| Multi occurrence | 511 | D4 R1@0.7 | **39.73** | 39.14 | -0.59 |
| Exactly two | 266 | D4 mAP Avg. | 33.72 | **33.99** | +0.27 |
| Three or more | 245 | D4 mAP Avg. | **21.98** | 21.88 | -0.10 |
| Multi occurrence | 511 | Coverage@5@0.5 | 0.5347 | **0.5415** | +0.0067 |

### Native-attention mechanism metrics

AEC is the fraction of matched queries whose dominant occurrence agrees with
the final Hungarian owner. ECR is the collision rate between queries matched
to different occurrences. Both metrics use the final D4 Hungarian assignment
traced back to each native attention layer and are averaged over the 511
multi-occurrence queries.

| Metric | Baseline | NativeBind | Delta | Desired direction |
| --- | ---: | ---: | ---: | :---: |
| D1 AEC | 0.6696 | **0.8480** | +0.1784 | up |
| D1 ECR | 0.3954 | **0.1490** | -0.2463 | down |
| D4 AEC | **0.7928** | 0.7322 | -0.0606 | up |
| D4 ECR | **0.2311** | 0.3250 | +0.0938 | down |

The directly supervised D1 attention shows a large, directionally correct
ownership improvement. That effect does not persist to D4: final-layer AEC
falls and collision rises. This explains why the experiment provides strong
evidence that the loss controls early evidence ownership, but does not support
a claim that D1-only NativeBind improves final test MR performance. The clean
current conclusion is therefore:

> Native Hungarian Binding at `lambda=0.5` strongly improves the intended D1
> occurrence-binding mechanism and slightly improves validation mAP, test HL,
> and loose-threshold multi-occurrence retrieval, but the improvement does not
> propagate reliably to D4 and test MR mAP decreases by 0.33.

## Reproduction

Launch the complete Baseline vs NativeBind `lambda=0.5` pair with one command.
By default the two jobs run concurrently on GPUs 0 and 1 and write to separate
output directories under one pair root:

```bash
bash causal_occurrence_lab/scripts/run_native_binding_pair.sh
```

For one GPU, run the same pair sequentially:

```bash
CAUSAL_PARALLEL=0 \
BASELINE_GPU_ID=0 \
NATIVE_GPU_ID=0 \
bash causal_occurrence_lab/scripts/run_native_binding_pair.sh
```

The paired outputs are:

```text
causal_occurrence_lab/outputs/native_binding_pair_seed2017/
  baseline_seed2017/model_best.ckpt
  native_bind_lambda_0p5_seed2017/model_best.ckpt
```

To train only NativeBind when the completed baseline checkpoint is already
available:

```bash
bash causal_occurrence_lab/scripts/run_native_binding_lambda_0p5.sh
```

Evaluate the two completed best checkpoints and generate the compact summary:

```bash
PAIR_ROOT="$PWD/causal_occurrence_lab/outputs/native_binding_pair_seed2017" \
CAUSAL_BASELINE_CHECKPOINT="$PAIR_ROOT/baseline_seed2017/model_best.ckpt" \
NATIVE_BIND_CHECKPOINT="$PAIR_ROOT/native_bind_lambda_0p5_seed2017/model_best.ckpt" \
CAUSAL_DATA=/path/to/highlight_test_with_gt.jsonl \
bash causal_occurrence_lab/scripts/eval_native_binding_completed.sh
```

Verify that the NativeBind checkpoint has the Vanilla Sim-DETR state contract
and that attention capture is exactly inference-neutral:

```bash
python causal_occurrence_lab/verify_native_binding_inference.py \
  --checkpoint /path/to/native_bind_lambda_0p5/model_best.ckpt \
  --baseline-checkpoint /path/to/baseline/model_best.ckpt \
  --data /path/to/highlight_test_with_gt.jsonl \
  --device cuda:0
```

The machine-readable completed-run summary is
[`results/native_binding/summary.json`](results/native_binding/summary.json),
and the saved architecture/capture check is
[`results/native_binding/inference_contract.json`](results/native_binding/inference_contract.json).
