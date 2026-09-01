"""Train Static/Full semantic calibration with the native Sim-DETR protocol."""

from __future__ import annotations

import logging
import os
import random
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from sim_detr.config import BaseOptions
from sim_detr.start_end_dataset import StartEndDataset
from sim_detr.start_end_dataset_audio import StartEndDataset_audio
from sim_detr.train import set_seed, train, train_hl
from utils.model_utils import count_parameters

from .config import SemanticBaseOptions, add_semantic_args
from .model_builder import build_model_semantic, load_native_checkpoint

logger = logging.getLogger(__name__)


def _escape_argparse_percent(parser):
    for action in parser._actions:
        if isinstance(action.help, str):
            action.help = action.help.replace("%", "%%")


def _build_datasets(opt):
    dataset_config = dict(
        dset_name=opt.dset_name,
        data_path=opt.train_path,
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
        span_loss_type=opt.span_loss_type,
        txt_drop_ratio=opt.txt_drop_ratio,
        dset_domain=opt.dset_domain,
    )
    dataset_cls = StartEndDataset if opt.a_feat_dir is None else StartEndDataset_audio
    if opt.a_feat_dir is not None:
        dataset_config["a_feat_dir"] = opt.a_feat_dir
    train_dataset = dataset_cls(**dataset_config)

    eval_dataset = None
    if opt.eval_path is not None:
        eval_config = dict(dataset_config)
        eval_config["data_path"] = opt.eval_path
        eval_config["txt_drop_ratio"] = 0
        # Keep the same pretraining compatibility behaviour as train.py.
        if "sub_features" in opt.t_feat_dir:
            eval_config["q_feat_dir"] = opt.t_feat_dir.replace("sub_features", "text_features")
        eval_dataset = dataset_cls(**eval_config)
    return train_dataset, eval_dataset


def setup_model_semantic(opt):
    model, criterion = build_model_semantic(opt)
    model.to(opt.device)
    criterion.to(opt.device)
    optimizer = torch.optim.AdamW(
        [{"params": [p for p in model.parameters() if p.requires_grad]}],
        lr=opt.lr,
        weight_decay=opt.wd,
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, opt.lr_drop)

    if opt.init_from_native is not None:
        load_native_checkpoint(model, opt.init_from_native)
    if opt.resume is not None:
        checkpoint = torch.load(opt.resume, map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=True)
        if opt.resume_all:
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            if opt.start_epoch is None:
                opt.start_epoch = checkpoint["epoch"] + 1
    return model, criterion, optimizer, lr_scheduler


def start_training(argv=None):
    options = SemanticBaseOptions()
    options.initialize()
    _escape_argparse_percent(options.parser)
    old_argv = sys.argv
    if argv is not None:
        sys.argv = [old_argv[0], *argv]
    try:
        opt = options.parse()
    finally:
        sys.argv = old_argv

    if opt.semantic_variant == "native":
        raise ValueError(
            "Use sim_detr/train.py for the already-reproduced native baseline; "
            "this entry point is for static/full semantic calibration."
        )
    if opt.device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        torch.cuda.set_device(opt.gpu_id)
    set_seed(opt.seed)
    logger.info("Semantic variant=%s, seed=%d", opt.semantic_variant, opt.seed)

    train_dataset, eval_dataset = _build_datasets(opt)
    model, criterion, optimizer, lr_scheduler = setup_model_semantic(opt)
    count_parameters(model)
    if opt.dset_name == "tvsum":
        train_hl(model, criterion, optimizer, lr_scheduler, train_dataset, eval_dataset, opt)
    else:
        train(model, criterion, optimizer, lr_scheduler, train_dataset, eval_dataset, opt)
    return opt.ckpt_filepath.replace(".ckpt", "_best.ckpt")


if __name__ == "__main__":
    start_training()
