"""Build the isolated two-layer Soccer-GMR LS-DQ-CGP model."""

from __future__ import annotations

import logging

import torch

from sim_detr.ls_dq_cgp.model_builder import build_model_ls_dq_cgp

from .criterion import SoccerGMRLSDQCGPCriterion


logger = logging.getLogger(__name__)


def _load_plain_baseline(model, path):
    checkpoint = torch.load(path, map_location="cpu")
    source = checkpoint.get("model", checkpoint)
    source = {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in source.items()
    }
    if any(key.startswith(("ls_cgp.", "ls_exist_head.")) for key in source):
        raise RuntimeError("--init_from expects a plain Sim-DETR checkpoint")
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
    logger.info("Warm-started baseline; %d LS/Exist tensors initialized", len(missing))


def build_soccer_gmr_ls_dq_cgp(opt):
    if int(opt.dec_layers) != 2:
        raise ValueError("Soccer-GMR LS-DQ-CGP requires exactly two decoder layers")
    model, native_criterion = build_model_ls_dq_cgp(opt)
    model.static_bypass = bool(opt.ls_static_bypass)
    model.context_roll = bool(opt.ls_context_roll)
    if opt.init_from is not None:
        _load_plain_baseline(model, opt.init_from)
    criterion = SoccerGMRLSDQCGPCriterion(
        native_criterion,
        binding_loss_coef=opt.ls_binding_loss_coef,
        exist_loss_coef=opt.ls_exist_loss_coef,
        saliency_loss_coef=opt.lw_saliency,
        saliency_margin=opt.saliency_margin,
        background_focal_weight=opt.background_focal_weight,
        null_background_focal_weight=opt.null_background_focal_weight,
        null_iou_loss_weight=opt.null_iou_loss_weight,
    )
    return model, criterion


def load_checkpoint_strict(model, path):
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=True)
    return checkpoint
