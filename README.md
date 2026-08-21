# [ICCV2025] Sim-DETR: Unlock DETR for Temporal Sentence Grounding

> **DQ-CGP extension.** The complete implementation and exact reproduction
> commands for the DQ-CGP V3 (D1) result are in
> [`sim_detr/dq_cgp/README.md`](sim_detr/dq_cgp/README.md).  The matching
> trained checkpoint is provided as a GitHub Release asset; download it into
> the run directory together with the release's `opt.json` before inference.

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

