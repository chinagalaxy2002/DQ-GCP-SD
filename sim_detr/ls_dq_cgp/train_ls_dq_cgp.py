"""Training entry point for the two-layer Sim-DETR + LS-DQ-CGP experiment."""

from __future__ import annotations

import logging

import torch

from sim_detr.config import BaseOptions
from sim_detr.start_end_dataset import StartEndDataset
from sim_detr.start_end_dataset_audio import StartEndDataset_audio
from sim_detr.dq_cgp.train_dq_cgp import (
    _escape_argparse_percent,
    set_seed,
    train,
)
from utils.model_utils import count_parameters
from .model_builder import add_ls_dq_cgp_args, build_model_ls_dq_cgp


logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


def _load_baseline(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    source = checkpoint.get("model", checkpoint)
    source = {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in source.items()
    }
    if any(key.startswith(("ls_cgp.", "ls_exist_head.")) for key in source):
        raise RuntimeError("--init_from expects a plain Sim-DETR checkpoint")
    target_keys = set(model.state_dict())
    unexpected_source = sorted(set(source) - target_keys)
    if unexpected_source:
        raise RuntimeError(f"unmapped baseline keys: {unexpected_source[:20]}")
    missing, unexpected = model.load_state_dict(source, strict=False)
    bad_missing = [
        key for key in missing
        if not key.startswith(("ls_cgp.", "ls_exist_head."))
    ]
    if bad_missing or unexpected:
        raise RuntimeError(
            f"unsafe baseline warm-start; missing={bad_missing[:20]}, "
            f"unexpected={unexpected[:20]}"
        )
    logger.info("Warm-started baseline; %d LS tensors newly initialized", len(missing))


def _dataset(opt, path, train_split):
    common = dict(
        dset_name=opt.dset_name,
        data_path=path,
        v_feat_dirs=opt.v_feat_dirs,
        q_feat_dir=(
            opt.t_feat_dir if train_split
            else opt.t_feat_dir.replace("sub_features", "text_features")
        ),
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
        txt_drop_ratio=opt.txt_drop_ratio if train_split else 0,
        dset_domain=getattr(opt, "dset_domain", None),
    )
    if opt.a_feat_dir is None:
        return StartEndDataset(**common)
    return StartEndDataset_audio(a_feat_dir=opt.a_feat_dir, **common)


def setup(opt):
    if opt.resume is not None and opt.init_from is not None:
        raise ValueError("--resume and --init_from are mutually exclusive")
    if opt.resume_all and opt.resume is None:
        raise ValueError("--resume_all requires --resume")
    model, criterion = build_model_ls_dq_cgp(opt)
    model.to(opt.device)
    criterion.to(opt.device)
    total, trainable = count_parameters(model, verbose=False)
    ls_total, _ = count_parameters(model.ls_cgp, verbose=False)
    logger.info("Model parameters=%d trainable=%d LS-DQ-CGP=%d", total, trainable, ls_total)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=opt.lr, weight_decay=opt.wd
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, opt.lr_drop)
    resume_metadata = {}
    if opt.init_from is not None:
        _load_baseline(model, opt.init_from)
    if opt.resume is not None:
        checkpoint = torch.load(opt.resume, map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=True)
        if opt.resume_all:
            for key in ("optimizer", "lr_scheduler", "epoch"):
                if key not in checkpoint:
                    raise RuntimeError(f"full resume checkpoint lacks {key}")
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["lr_scheduler"])
            if opt.start_epoch is None:
                opt.start_epoch = int(checkpoint["epoch"]) + 1
            resume_metadata = {
                "best_metric": checkpoint.get("best_metric", checkpoint.get("metric", 0.0)),
                "es_cnt": checkpoint.get("es_cnt", 0),
            }
    return model, criterion, optimizer, scheduler, resume_metadata


def start_training():
    options = BaseOptions()
    options.initialize()
    add_ls_dq_cgp_args(options.parser)
    _escape_argparse_percent(options.parser)
    opt = options.parse()
    if opt.dec_layers != 2:
        raise ValueError("canonical LS-DQ-CGP requires --dec_layers 2")
    if opt.use_ls_exist_head:
        raise ValueError(
            "The stock Sim-DETR dataset has no exist_label. Use a GMR dataset "
            "adapter that supplies targets['exist_label'] before enabling this head."
        )
    if opt.device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.cuda.set_device(opt.gpu_id)
    set_seed(opt.seed, use_cuda=opt.device.type == "cuda")
    train_dataset = _dataset(opt, opt.train_path, train_split=True)
    eval_dataset = _dataset(opt, opt.eval_path, train_split=False) if opt.eval_path else None
    model, criterion, optimizer, scheduler, metadata = setup(opt)
    return train(
        model, criterion, optimizer, scheduler, train_dataset, eval_dataset, opt, metadata
    )


if __name__ == "__main__":
    start_training()
