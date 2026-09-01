"""Evaluate a Soccer-GMR CSC checkpoint and apply inference interventions."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .dataset import SoccerGMRDataset
from .evaluation import evaluate_model
from .model_builder import build_soccer_gmr_model, load_checkpoint_strict


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--semantic_variant", choices=("native", "static", "full"))
    parser.add_argument("--context_variant", choices=(
        "aligned", "roll-1", "roll-2", "roll-3", "random-derangement",
        "farthest-context", "uniform",
    ), default="aligned")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args(argv)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    opt = checkpoint["opt"]
    opt.device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu_id >= 0 else "cpu")
    opt.gpu_id = args.gpu_id
    if args.semantic_variant:
        opt.semantic_variant = args.semantic_variant
    opt.semantic_context_variant = args.context_variant
    model, _ = build_soccer_gmr_model(opt)
    load_checkpoint_strict(model, args.checkpoint)
    model.set_counterfactual(opt.semantic_variant, args.context_variant)
    model.to(opt.device)
    path = opt.eval_path if args.split == "val" else opt.test_path
    dataset = SoccerGMRDataset(
        path, opt.v_feat_dirs, opt.t_feat_dir, max_q_l=opt.max_q_l,
        max_v_l=opt.max_v_l, max_windows=opt.max_windows,
        clip_len=opt.clip_length, load_labels=True,
    )
    try:
        evaluate_model(
            model, dataset, opt, args.output_dir,
            f"{args.split}_{opt.semantic_variant}_{args.context_variant}",
        )
    finally:
        model.close()


if __name__ == "__main__":
    main()
