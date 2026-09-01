"""Strict inference and causal counterfactuals for LS-DQ-CGP checkpoints."""

from __future__ import annotations

import argparse
import logging
import os
import pprint
import sys

import torch
import torch.backends.cudnn as cudnn

from sim_detr.config import TestOptions
from sim_detr.inference import eval_epoch
from sim_detr.dq_cgp.inference_dq_cgp import build_eval_dataset
from sim_detr.dq_cgp.train_dq_cgp import _escape_argparse_percent
from .model_builder import add_ls_dq_cgp_args, build_model_ls_dq_cgp


logger = logging.getLogger(__name__)


def _runtime_overrides(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--resume")
    parser.add_argument("--eval_split_name")
    parser.add_argument("--eval_path")
    parser.add_argument("--eval_results_dir")
    parser.add_argument("--gpu_id", type=int)
    parser.add_argument("--device", type=int)
    parser.add_argument("--ls_static_bypass", action="store_true")
    parser.add_argument("--ls_context_roll", action="store_true")
    return parser.parse_known_args(argv)[0]


def parse_options(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    runtime = _runtime_overrides(argv)
    if runtime.resume is None and not any(flag in argv for flag in ("-h", "--help")):
        raise ValueError("--resume is required")
    options = TestOptions()
    options.initialize()
    add_ls_dq_cgp_args(options.parser)
    _escape_argparse_percent(options.parser)
    previous = sys.argv
    try:
        sys.argv = [previous[0], *argv]
        opt = options.parse()
    finally:
        sys.argv = previous
    for name in ("eval_split_name", "eval_path", "gpu_id"):
        value = getattr(runtime, name)
        if value is not None:
            setattr(opt, name, value)
    if runtime.eval_results_dir is not None:
        opt.eval_results_dir = runtime.eval_results_dir
        opt.results_dir = runtime.eval_results_dir
        os.makedirs(opt.results_dir, exist_ok=True)
    if runtime.device is not None:
        opt.device = torch.device("cuda" if runtime.device >= 0 else "cpu")
    opt.ls_static_bypass = runtime.ls_static_bypass
    opt.ls_context_roll = runtime.ls_context_roll
    opt.init_from = None
    return opt


def start_inference(argv=None):
    opt = parse_options(argv)
    if opt.dec_layers != 2:
        raise ValueError("checkpoint configuration must use dec_layers=2")
    if opt.device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.cuda.set_device(opt.gpu_id)
    model, criterion = build_model_ls_dq_cgp(opt)
    checkpoint = torch.load(opt.resume, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=True)
    model.static_bypass = opt.ls_static_bypass
    model.context_roll = opt.ls_context_roll
    model.to(opt.device).eval()
    criterion.to(opt.device).eval()
    dataset = build_eval_dataset(opt)
    cudnn.benchmark = True
    cudnn.deterministic = False
    mode = "static_bypass" if model.static_bypass else "context_roll" if model.context_roll else "active"
    logger.info("Strict LS-DQ-CGP inference mode=%s checkpoint=%s", mode, opt.resume)
    with torch.no_grad():
        result = eval_epoch(
            model,
            dataset,
            opt,
            f"ls_dq_cgp_{mode}_{opt.eval_split_name}.jsonl",
            criterion=criterion,
        )
    metrics, metrics_nms, _, paths = result
    if metrics is not None:
        logger.info("metrics %s", pprint.pformat(metrics["brief"], indent=4))
    return metrics, metrics_nms, paths


if __name__ == "__main__":
    start_inference()
