"""Evaluate a Soccer-GMR LS-DQ-CGP checkpoint and semantic counterfactuals."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .dataset import SoccerGMRDataset
from .evaluation import evaluate_model
from .model_builder import build_soccer_gmr_ls_dq_cgp, load_checkpoint_strict


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument(
        "--semantic_mode",
        choices=("aligned", "static_bypass", "context_roll"),
        default="aligned",
    )
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args(argv)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    opt = checkpoint["opt"]
    opt.device = torch.device(
        "cuda:0" if torch.cuda.is_available() and args.gpu_id >= 0 else "cpu"
    )
    opt.gpu_id = args.gpu_id
    opt.init_from = None
    opt.ls_static_bypass = args.semantic_mode == "static_bypass"
    opt.ls_context_roll = args.semantic_mode == "context_roll"
    model, _ = build_soccer_gmr_ls_dq_cgp(opt)
    load_checkpoint_strict(model, args.checkpoint)
    model.static_bypass = opt.ls_static_bypass
    model.context_roll = opt.ls_context_roll
    model.to(opt.device)
    path = opt.eval_path if args.split == "val" else opt.test_path
    dataset = SoccerGMRDataset(
        path, opt.v_feat_dirs, opt.t_feat_dir,
        max_q_l=opt.max_q_l, max_v_l=opt.max_v_l,
        max_windows=opt.max_windows, clip_len=opt.clip_length,
        load_labels=True,
    )
    try:
        evaluate_model(
            model, dataset, opt, args.output_dir,
            f"{args.split}_{args.semantic_mode}",
        )
    finally:
        model.d1_capture.close()


if __name__ == "__main__":
    main()
