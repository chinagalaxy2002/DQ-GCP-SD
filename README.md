# [ICCV2025] Sim-DETR: Unlock DETR for Temporal Sentence Grounding

## DQ-CGP V3 (D1) extension

This repository includes the complete DQ-CGP implementation for
QVHighlights. DQ-CGP performs candidate-specific temporal binding, basis
routing, and feature refinement for each native DETR query. The selected V3
configuration inserts DQ-CGP once between decoder layers D1 and D2 and is
trained from scratch under the same protocol as the Sim-DETR baseline.

- Complete training and evaluation guide: [sim_detr/dq_cgp/README.md](sim_detr/dq_cgp/README.md)
- Released checkpoint: [DQ-CGP V3 (D1) release](https://github.com/chinagalaxy2002/DQ-GCP-SD/releases/tag/v3-d1-qvhighlights)
- Selection metric: validation `MR-full-mAP`, without using test GT for model selection

### QVHighlights results

| Method | test R1@0.5 | test R1@0.7 | test mAP@0.5 | test mAP@0.75 | test mAP Avg. | val R1@0.5 | val R1@0.7 | val mAP@0.5 | val mAP@0.75 | val mAP Avg. |
| ------ | ----------: | ----------: | -----------: | ------------: | ------------: | ---------: | ---------: | ----------: | -----------: | -----------: |
| Sim-DETR baseline | 66.93 | **51.56** | 67.75 | 48.89 | 47.60 | **68.32** | 53.81 | **69.03** | 50.77 | 49.14 |
| **DQ-CGP V3 (D1)** | **67.96** | 51.36 | **68.94** | **49.01** | **48.06** | 67.81 | **54.06** | 68.81 | **51.01** | **49.66** |
| Improvement | +1.03 | -0.20 | +1.19 | +0.12 | **+0.46** | -0.51 | +0.25 | -0.22 | +0.24 | **+0.52** |

The released V3 checkpoint was selected at epoch 103. Its main parameters are
seed 2017, `beta=0.05`, binding-loss coefficient `0.20`, routing-loss
coefficient `0.01`, 16 bases, and prompt length 6. Test metrics are computed
only after validation-based checkpoint selection using the local
`data/highlight_test_with_gt.jsonl` annotations.

The complete checkpoint is stored as one Git LFS object at
`checkpoints/model_best.ckpt`. Clone it together with the source and verify the
downloaded file:

```bash
git lfs pull
sha256sum checkpoints/model_best.ckpt
# cb0df35b25397e34b8da27e0dd9a266d4fca00c0584cfbd45b5be8639ebc3e19
```

### Exploratory DQ-CGP variants on test

| Method | R1@0.5 | R1@0.7 | mAP@0.5 | mAP@0.75 | mAP Avg. |
| --- | ---: | ---: | ---: | ---: | ---: |
| DQ-CGP V3 (D1) | **67.96** | 51.36 | **68.94** | **49.01** | **48.06** |
| Grid: beta=0.100, bind=0.40, route=0.005 | 66.80 | 51.23 | 67.48 | 48.96 | 47.88 |
| Grid: beta=0.100, bind=0.10, route=0.020 | 68.35 | **52.40** | 68.75 | 48.06 | 47.82 |
| Grid: beta=0.025, bind=0.10, route=0.005 | 66.93 | 51.62 | 67.89 | 48.31 | 47.68 |
| Grid: beta=0.100, bind=0.40, route=0.020 | 67.90 | 50.97 | 68.17 | 48.25 | 47.66 |
| Grid: beta=0.025, bind=0.40, route=0.005 | 67.25 | 52.01 | 67.90 | 48.20 | 47.63 |
| Grid: beta=0.025, bind=0.40, route=0.020 | 67.51 | 51.23 | 67.80 | 47.76 | 47.49 |
| Grid: beta=0.100, bind=0.10, route=0.005 | 66.93 | 49.94 | 67.59 | 47.90 | 47.40 |
| Grid: beta=0.025, bind=0.10, route=0.020 | 66.80 | 50.52 | 67.55 | 46.89 | 46.73 |
| DQ-CGP tied dual | 66.15 | 50.26 | 67.21 | 47.67 | 46.88 |
| DQ-CGP dual independent | 66.21 | 51.36 | 67.21 | 47.76 | 46.85 |
| DQ-CGP tied all | 66.21 | 49.81 | 67.30 | 47.10 | 46.80 |

by Jiajin Tang*, Zhengxuan Wei*, Yuchen Zhu, Cheng Shi, Guanbin Li, Liang Lin, Sibei Yang†

*Equal contribution; †Corresponding Author


[![arXiv:2509.23867](https://img.shields.io/badge/arXiv-2509.23867-red)](https://arxiv.org/abs/2509.23867)

----------
## Abstract

Temporal sentence grounding aims to identify exact moments in a video that correspond to a given textual query, typically addressed with detection transformer (DETR) solutions. However, we find that typical strategies designed to enhance DETR do not improve, and may even degrade, its performance in this task. We systematically analyze and identify the root causes of this abnormal behavior: (1) conflicts between queries from similar target moments and (2) internal query conflicts due to the tension between global semantics and local localization. Building on these insights, we propose a simple yet powerful baseline, Sim-DETR, which extends the standard DETR with two minor modifications in the decoder layers: (1) constraining self-attention between queries based on their semantic and positional overlap and (2) adding query-to-frame alignment to bridge the global and local contexts. Experiments demonstrate that Sim-DETR unlocks the full potential of DETR for temporal sentence grounding, offering a strong baseline for future research.

----------
## Framework
<p align="center">
  <img src="assets/framework.png" width="700"/>
</p>

----------

## Prerequisites

### 0. Clone this repository

```
git clone https://github.com/SooLab/Sim-DETR.git
cd Sim-DETR
```

### 1. Prepare datasets
#### QVHighlights

We use video features (CLIP and SlowFast) and text features (CLIP) as inputs. For CLIP, we utilize the features extracted by [R2-Tuning](https://github.com/yeliudev/R2-Tuning) (from the last four layers), but we retain only the `[CLS]` token per frame to ensure efficiency. You can download our prepared feature files from [qvhighlights\_features](https://drive.google.com/drive/folders/1rRVID6OO5arVR1vL5SP5fcCFAJ35B-IK?usp=sharing) and unzip them to your data root directory.


### 2. Install dependencies

For Anaconda setup, refer to the official [Moment-DETR GitHub](https://github.com/jayleicn/moment_detr).

----------

## QVHighlights

### Training

Update `feat_root` in `sim_detr/scripts/train.sh` to the path where you saved the features, then run:

```bash
bash sim_detr/scripts/train.sh  
```


### Inference Evaluation and Codalab Submission

After training, you can generate `hl_val_submission.jsonl` and `hl_test_submission.jsonl` for validation and test sets by running:

```
bash sim_detr/scripts/inference.sh results/{direc}/model_best.ckpt 'val'
bash sim_detr/scripts/inference.sh results/{direc}/model_best.ckpt 'test'
```

Replace `{direc}` with the path to your saved checkpoint. For more details on submission, see [standalone_eval/README.md](standalone_eval/README.md).

----------

## Citation

If you find this repository useful, please cite our work:

```
@inproceedings{tang2025sim,
  title={Sim-DETR: Unlock DETR for Temporal Sentence Grounding},
  author={Tang, Jiajin and Wei, Zhengxuan and Zhu, Yuchen and Shi, Cheng and Li, Guanbin and Lin, Liang and Yang, Sibei},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={22760--22771},
  year={2025}
}
```

----------

## License

The annotation files and parts of the implementation are borrowed from Moment-DETR and TR-DETR. Consequently, our code is also released under the [MIT License](https://opensource.org/licenses/MIT).
