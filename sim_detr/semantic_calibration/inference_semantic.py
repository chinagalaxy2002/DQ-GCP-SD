"""Inference and counterfactual evaluation for semantic-calibration checkpoints."""

from __future__ import annotations

import argparse
import logging
import os
import pprint
import sys

import torch
import torch.backends.cudnn as cudnn

from sim_detr.inference import eval_epoch
from sim_detr.start_end_dataset import StartEndDataset
from sim_detr.start_end_dataset_audio import StartEndDataset_audio
from utils.basic_utils import load_jsonl, save_json

from .config import SemanticTestOptions
from .model_builder import build_model_semantic, load_semantic_checkpoint
from .stratified_eval import stratified_occurrence_metrics

logger = logging.getLogger(__name__)


def _escape_argparse_percent(parser):
    for action in parser._actions:
        if isinstance(action.help, str):
            action.help = action.help.replace("%", "%%")


def parse_inference_options(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if "--resume" not in argv:
        raise ValueError("--resume must point to a semantic-calibration checkpoint")
    options = SemanticTestOptions()
    options.initialize()
    _escape_argparse_percent(options.parser)
    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0], *argv]
        opt = options.parse()
    finally:
        sys.argv = old_argv
    if opt.semantic_variant == "native":
        logger.warning("native bypass selected; this is a counterfactual, not a new baseline")
    if opt.device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        torch.cuda.set_device(opt.gpu_id)
    return opt


def build_eval_dataset(opt):
    if opt.eval_path is None:
        raise ValueError("--eval_path is required")
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
        load_labels=opt.eval_split_name != "test",
        span_loss_type=opt.span_loss_type,
        txt_drop_ratio=0,
        dset_domain=getattr(opt, "dset_domain", None),
    )
    if opt.a_feat_dir is None:
        return StartEndDataset(**common)
    return StartEndDataset_audio(a_feat_dir=opt.a_feat_dir, **common)


def run_one(opt, semantic_variant, context_variant, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    opt.results_dir = output_dir
    opt.semantic_variant = semantic_variant
    opt.semantic_context_variant = context_variant
    model, criterion = build_model_semantic(opt)
    checkpoint = load_semantic_checkpoint(model, opt.resume)
    model.to(opt.device)
    criterion.to(opt.device)
    submission_filename = f"hl_{opt.eval_split_name}_{semantic_variant}_{context_variant}.jsonl"
    try:
        with torch.no_grad():
            metrics, metrics_nms, _, paths = eval_epoch(
                model, build_eval_dataset(opt), opt, submission_filename, criterion=criterion
            )
    finally:
        model.close()
    submission_path = os.path.join(output_dir, submission_filename)
    stratified = stratified_occurrence_metrics(
        load_jsonl(submission_path),
        load_jsonl(opt.eval_path),
        max_windows=opt.max_windows,
    )
    save_json(stratified, os.path.join(output_dir, "stratified_metrics.json"), save_pretty=True)
    result = {
        "semantic_variant": semantic_variant,
        "context_variant": context_variant,
        "checkpoint_epoch": checkpoint.get("epoch", None),
        "metrics": metrics,
        "metrics_nms": metrics_nms,
        "stratified_metrics": stratified,
        "paths": paths,
    }
    torch.save(result, os.path.join(output_dir, "counterfactual_result.pt"))
    if metrics is not None:
        logger.info("%s/%s: %s", semantic_variant, context_variant, pprint.pformat(metrics["brief"]))
    return result


def main(argv=None):
    opt = parse_inference_options(argv)
    root = opt.eval_results_dir or os.path.join(opt.results_dir, "expanded_counterfactuals")
    variants = {
        "aligned": ("full", "aligned"),
        "roll-1": ("full", "roll-1"),
        "roll-2": ("full", "roll-2"),
        "roll-3": ("full", "roll-3"),
        "random-derangement": ("full", "random-derangement"),
        "farthest-context": ("full", "farthest-context"),
        "uniform": ("full", "uniform"),
        "static": ("static", "aligned"),
    }
    for name, (semantic_variant, context_variant) in variants.items():
        run_one(opt, semantic_variant, context_variant, os.path.join(root, name))


if __name__ == "__main__":
    main()
