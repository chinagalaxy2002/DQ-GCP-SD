"""Evaluate QVHighlights Full + D1 attention + Hungarian Binding checkpoints."""

from __future__ import annotations

import argparse
import logging
import os
import pprint
import sys
import torch

from sim_detr.inference import eval_epoch
from sim_detr.start_end_dataset import StartEndDataset
from sim_detr.start_end_dataset_audio import StartEndDataset_audio
from utils.basic_utils import load_jsonl, save_json

from .config import ExperimentOptions
from .model_builder import build_model, load_checkpoint

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


def _escape_argparse_percent(parser) -> None:
    for action in parser._actions:
        if isinstance(action.help, str):
            action.help = action.help.replace("%", "%%")


def parse_eval_options(argv=None):
    options = ExperimentOptions()
    options.initialize()
    _escape_argparse_percent(options.parser)
    old_argv = sys.argv
    if argv is not None:
        sys.argv = [old_argv[0], *argv]
    try:
        opt = options.parse()
    finally:
        sys.argv = old_argv
    opt.semantic_variant = "full"
    opt.semantic_evidence_source = "d1_attention"
    opt.semantic_context_variant = "aligned"
    if opt.device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        torch.cuda.set_device(opt.gpu_id)
    return opt


def build_eval_dataset(opt):
    if opt.eval_path is None:
        raise ValueError("--eval_path is required")
    load_labels = opt.eval_split_name != "test"
    common = dict(
        dset_name=opt.dset_name,
        data_path=opt.eval_path,
        v_feat_dirs=opt.v_feat_dirs,
        q_feat_dir=opt.t_feat_dir,
        q_feat_type="last_hidden_state",
        max_q_l=opt.max_q_l,
        max_v_l=opt.max_v_l,
        ctx_mode=opt.ctx_mode,
        data_ratio=opt.data_ratio,
        normalize_v=not opt.no_norm_vfeat,
        normalize_t=not opt.no_norm_tfeat,
        clip_len=opt.clip_length,
        max_windows=opt.max_windows,
        load_labels=load_labels,
        span_loss_type=opt.span_loss_type,
        txt_drop_ratio=0,
        dset_domain=getattr(opt, "dset_domain", None),
    )
    if opt.a_feat_dir is None:
        return StartEndDataset(**common)
    return StartEndDataset_audio(a_feat_dir=opt.a_feat_dir, **common)


def run_evaluation(opt, checkpoint_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    opt.results_dir = output_dir
    model, criterion = build_model(opt)
    checkpoint = load_checkpoint(model, checkpoint_path)
    model.to(opt.device)
    criterion.to(opt.device)

    submission_filename = f"hl_{opt.eval_split_name}_submission.jsonl"
    try:
        with torch.no_grad():
            dataset = build_eval_dataset(opt)
            metrics, metrics_nms, _, paths = eval_epoch(
                model, dataset, opt, submission_filename, criterion=criterion
            )
    finally:
        model.close()

    result = {
        "checkpoint_path": checkpoint_path,
        "checkpoint_epoch": checkpoint.get("epoch", None),
        "metrics": metrics,
        "metrics_nms": metrics_nms,
        "paths": paths,
    }
    torch.save(result, os.path.join(output_dir, "eval_result.pt"))
    if metrics is not None:
        logger.info("Evaluation metrics: %s", pprint.pformat(metrics["brief"]))
    return result


def main():
    parser = argparse.ArgumentParser(description="Evaluate Full D1 Binding on QVHighlights")
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--eval_path", required=True, type=str)
    parser.add_argument("--eval_split_name", required=True, type=str)
    parser.add_argument("--dec_layers", type=int, default=4)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--binding_loss_coef", type=float, default=0.2)
    parser.add_argument("--eval_bsz", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2023)
    args = parser.parse_args()

    argv = [
        "--resume", args.checkpoint,
        "--eval_path", args.eval_path,
        "--eval_split_name", args.eval_split_name,
        "--dec_layers", str(args.dec_layers),
        "--gpu_id", str(args.gpu_id),
        "--device", str(args.gpu_id),
        "--results_root", os.path.dirname(args.output_dir),
        "--exp_id", os.path.basename(args.output_dir),
        "--binding_loss_coef", str(args.binding_loss_coef),
        "--eval_bsz", str(args.eval_bsz),
        "--seed", str(args.seed),
        "--dset_name", "hl",
        "--ctx_mode", "video_tef",
        "--train_path", "data/highlight_train_release.jsonl",
        "--v_feat_dirs", "data/slowfast_features", "data/clip_b32_vid_k4",
        "--v_feat_dim", "5376",
        "--t_feat_dir", "data/clip_b32_txt_k4",
        "--t_feat_dim", "2048",
        "--max_windows", "8",
        "--enc_layers", "2",
    ]
    opt = parse_eval_options(argv)
    opt.eval_bsz = args.eval_bsz
    run_evaluation(opt, args.checkpoint, args.output_dir)


if __name__ == "__main__":
    main()
