"""Strict checkpoint inference for Sim-DETR + DQ-CGP.

The original ``sim_detr.inference`` evaluator is reused unchanged.  This entry
point only replaces model construction/loading so that DQ-CGP checkpoints are
rebuilt from their saved ``opt.json`` and loaded with ``strict=True``.
"""

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
from sim_detr.start_end_dataset import StartEndDataset
from sim_detr.start_end_dataset_audio import StartEndDataset_audio
from sim_detr.dq_cgp.model_builder import add_dq_cgp_args, build_model_dq_cgp
from utils.model_utils import count_parameters


logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


def _escape_argparse_percent(parser):
    """Work around literal percent signs in the baseline help strings."""
    for action in parser._actions:
        if isinstance(action.help, str):
            action.help = action.help.replace("%", "%%")


def _requested_eval_overrides(argv):
    """Capture eval overrides before TestOptions reloads training opt.json."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--eval_split_name", type=str, default=None)
    parser.add_argument("--eval_path", type=str, default=None)
    parser.add_argument("--eval_results_dir", type=str, default=None)
    parser.add_argument("--gpu_id", type=int, default=None)
    parser.add_argument("--device", type=int, default=None)
    known, _ = parser.parse_known_args(argv)
    return known


def parse_inference_options(argv=None):
    """Load architecture options from the checkpoint directory's opt.json."""
    argv = sys.argv[1:] if argv is None else list(argv)
    overrides = _requested_eval_overrides(argv)
    if overrides.resume is None and not any(flag in argv for flag in ("-h", "--help")):
        raise ValueError("--resume must point to a DQ-CGP checkpoint")

    options = TestOptions()
    options.initialize()
    add_dq_cgp_args(options.parser)
    _escape_argparse_percent(options.parser)

    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0], *argv]
        opt = options.parse()
    finally:
        sys.argv = old_argv

    # TestOptions intentionally reloads training settings.  Evaluation split,
    # annotation path, output directory and physical GPU are runtime overrides.
    if overrides.eval_split_name is not None:
        opt.eval_split_name = overrides.eval_split_name
    if overrides.eval_path is not None:
        opt.eval_path = overrides.eval_path
    if overrides.eval_results_dir is not None:
        opt.eval_results_dir = overrides.eval_results_dir
        opt.results_dir = overrides.eval_results_dir
        os.makedirs(opt.results_dir, exist_ok=True)
    if overrides.gpu_id is not None:
        opt.gpu_id = overrides.gpu_id
    if overrides.device is not None:
        opt.device = torch.device("cuda" if overrides.device >= 0 else "cpu")

    # A saved warm-start path describes training provenance, not an inference
    # action.  Prevent model_builder from loading it a second time.
    opt.init_from = None
    return opt


def setup_model_dq_cgp_inference(opt):
    """Build the saved DQ architecture and strictly restore its checkpoint."""
    if not getattr(opt, "use_query_cgp", False):
        raise ValueError(
            "The checkpoint opt.json does not enable DQ-CGP. Evaluate the "
            "original Sim-DETR baseline with sim_detr/inference.py instead."
        )
    if opt.device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        torch.cuda.set_device(opt.gpu_id)

    model, criterion = build_model_dq_cgp(opt)
    checkpoint = torch.load(opt.resume, map_location="cpu")
    if "model" not in checkpoint:
        raise RuntimeError(f"Checkpoint has no 'model' state: {opt.resume}")
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(opt.device)
    criterion.to(opt.device)
    model_params, _ = count_parameters(model, verbose=False)
    logger.info(
        "Strictly loaded %s (epoch %s, %d parameters)",
        opt.resume,
        checkpoint.get("epoch", "unknown"),
        model_params,
    )
    return model, criterion


def build_eval_dataset(opt):
    """Construct the original Sim-DETR evaluation dataset unchanged."""
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


def start_inference_dq_cgp(argv=None):
    opt = parse_inference_options(argv)
    logger.info("Evaluation split=%s annotations=%s", opt.eval_split_name, opt.eval_path)

    cudnn.benchmark = True
    cudnn.deterministic = False
    eval_dataset = build_eval_dataset(opt)
    model, criterion = setup_model_dq_cgp_inference(opt)

    submission_filename = f"hl_{opt.eval_split_name}_submission.jsonl"
    with torch.no_grad():
        metrics_no_nms, metrics_nms, _, latest_file_paths = eval_epoch(
            model,
            eval_dataset,
            opt,
            submission_filename,
            criterion=criterion,
        )

    if opt.eval_split_name != "test":
        logger.info(
            "metrics_no_nms %s", pprint.pformat(metrics_no_nms["brief"], indent=4)
        )
    if metrics_nms is not None:
        logger.info("metrics_nms %s", pprint.pformat(metrics_nms["brief"], indent=4))
    return metrics_no_nms, metrics_nms, latest_file_paths


if __name__ == "__main__":
    start_inference_dq_cgp()
