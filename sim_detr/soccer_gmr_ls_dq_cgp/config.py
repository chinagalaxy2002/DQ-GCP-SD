"""Standalone Soccer-GMR options for the LS-DQ-CGP experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


DEFAULT_GMR_ROOT = Path(
    "/home/guoxiangyu/VLMbasedIter_momentretrival/generalized-moment-retrieval"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Two-layer Sim-DETR + LS-DQ-CGP on Soccer-GMR")
    parser.add_argument("--exp_id", required=True)
    parser.add_argument("--gmr_root", type=Path, default=DEFAULT_GMR_ROOT)
    parser.add_argument(
        "--results_root", type=Path, default=Path("results_soccer_gmr_ls_dq_cgp")
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--n_epoch", type=int, default=400)
    parser.add_argument("--max_es_cnt", type=int, default=50)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--lr_drop", type=int, default=400)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--bsz", type=int, default=8)
    parser.add_argument("--eval_bsz", type=int, default=8)
    parser.add_argument("--grad_clip", type=float, default=0.1)
    parser.add_argument("--eval_epoch_interval", type=int, default=1)
    parser.add_argument("--max_q_l", type=int, default=32)
    parser.add_argument("--max_v_l", type=int, default=75)
    parser.add_argument("--max_windows", type=int, default=8)
    parser.add_argument("--clip_length", type=int, default=2)

    parser.add_argument("--ls_num_basis", type=int, default=16)
    parser.add_argument("--ls_prompt_length", type=int, default=6)
    parser.add_argument("--ls_router_hidden_dim", type=int, default=256)
    parser.add_argument("--ls_frf_hidden_dim", type=int, default=512)
    parser.add_argument("--ls_temperature", type=float, default=1.0)
    parser.add_argument("--ls_initial_scale", type=float, default=10.0)
    parser.add_argument("--ls_initial_bias", type=float, default=-2.0)
    parser.add_argument("--ls_binding_loss_coef", type=float, default=0.2)
    parser.add_argument("--ls_exist_loss_coef", type=float, default=1.0)
    parser.add_argument("--exist_gate_thd", type=float, default=0.4)
    parser.add_argument("--ls_static_bypass", action="store_true")
    parser.add_argument("--ls_context_roll", action="store_true")
    parser.add_argument("--init_from", type=Path)
    return parser


def parse_options(argv=None):
    opt = build_parser().parse_args(argv)
    root = opt.gmr_root.expanduser().resolve()
    feature_root = root / "Soccergmr"
    opt.gmr_root = root
    opt.train_path = root / "data/label/Standard/train.jsonl"
    opt.eval_path = root / "data/label/Standard/val.jsonl"
    opt.test_path = root / "data/label/Standard/test.jsonl"
    opt.v_feat_dirs = [feature_root / "clip", feature_root / "slowfast"]
    opt.t_feat_dir = feature_root / "clip_text"

    opt.dset_name = "soccer_gmr"
    opt.eval_split_name = "val"
    opt.ctx_mode = "video_tef"
    opt.v_feat_dim = 512 + 2304 + 2
    opt.t_feat_dim = 512
    opt.a_feat_dir = None
    opt.a_feat_dim = None
    opt.device = torch.device(
        f"cuda:{opt.gpu_id}" if torch.cuda.is_available() and opt.gpu_id >= 0 else "cpu"
    )
    opt.pin_memory = opt.device.type == "cuda"

    # Fixed native Sim-DETR architecture. LS-DQ-CGP consumes D1 attention and D2 states.
    opt.position_embedding = "sine"
    opt.enc_layers = 2
    opt.dec_layers = 2
    opt.dim_feedforward = 1024
    opt.hidden_dim = 256
    opt.input_dropout = 0.5
    opt.dropout = 0.1
    opt.txt_drop_ratio = 0.0
    opt.use_txt_pos = False
    opt.nheads = 8
    opt.num_queries = 10
    opt.pre_norm = False
    opt.n_input_proj = 2
    opt.contrastive_hdim = 64
    opt.temperature = 0.07
    opt.lw_saliency = 1.0
    opt.saliency_margin = 0.2
    opt.aux_loss = True
    opt.span_loss_type = "l1"
    opt.contrastive_align_loss = False
    opt.contrastive_align_loss_coef = 0.0

    # Soccer-GMR protocol and null-aware supervision.
    opt.set_cost_span = 10.0
    opt.set_cost_giou = 1.0
    opt.set_cost_class = 4.0
    opt.set_cost_mask = 6.0
    opt.span_loss_coef = 10.0
    opt.giou_loss_coef = 1.0
    opt.label_loss_coef = 4.0
    opt.eos_coef = 0.1
    opt.mask_loss_coef = 6.0
    opt.iou_scores_loss_coef = 2.0
    opt.VTC_loss_coef = 0.3
    opt.CTC_loss_coef = 0.5
    opt.background_focal_weight = 0.1
    opt.null_background_focal_weight = 0.05
    opt.null_iou_loss_weight = 0.05
    opt.null_ctc_loss_weight = 0.1
    opt.vtc_group_size = 8
    opt.use_ls_exist_head = True

    required = [opt.train_path, opt.eval_path, opt.test_path, opt.t_feat_dir, *opt.v_feat_dirs]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing Soccer-GMR inputs: {missing}")
    if opt.resume is not None and opt.init_from is not None:
        raise ValueError("--resume and --init_from are mutually exclusive")
    return opt


def serializable_options(opt):
    result = {}
    for key, value in vars(opt).items():
        if isinstance(value, Path):
            result[key] = str(value)
        elif isinstance(value, torch.device):
            result[key] = str(value)
        elif isinstance(value, list):
            result[key] = [str(item) if isinstance(item, Path) else item for item in value]
        else:
            result[key] = value
    return result


def save_options(opt, output_dir):
    with (Path(output_dir) / "opt.json").open("w", encoding="utf-8") as handle:
        json.dump(serializable_options(opt), handle, indent=2, ensure_ascii=False)
