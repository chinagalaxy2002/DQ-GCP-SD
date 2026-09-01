"""Build unchanged Sim-DETR under the independent CSC wrapper."""

from __future__ import annotations

import torch

from sim_detr.semantic_calibration.semantic_model import SimDETRWithSemanticCalibration

from .criterion import NullSafeCriterion
from .gmr_adapter import SoccerGMRModel


def build_soccer_gmr_model(opt):
    from sim_detr.model import build_model

    base_model, native_criterion = build_model(opt)
    semantic_model = SimDETRWithSemanticCalibration(
        base_model=base_model,
        hidden_dim=opt.semantic_hidden_dim,
        dropout=opt.semantic_dropout,
        semantic_scale_init=opt.semantic_scale_init,
        semantic_variant=opt.semantic_variant,
        detach_support=opt.semantic_detach_support,
        diagnostic_mode=opt.semantic_diagnostic_mode,
    )
    semantic_model.semantic_context_variant = opt.semantic_context_variant
    semantic_model.semantic_counterfactual_seed = opt.semantic_counterfactual_seed
    model = SoccerGMRModel(
        semantic_model, hidden_dim=opt.hidden_dim, exist_pool=opt.exist_pool
    )
    return model, NullSafeCriterion(
        native_criterion, exist_loss_coef=opt.exist_loss_coef,
        saliency_loss_coef=opt.lw_saliency, saliency_margin=opt.saliency_margin,
    )


def load_checkpoint_strict(model, path):
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=True)
    return checkpoint
