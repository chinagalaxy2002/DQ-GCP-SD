"""Build unchanged Sim-DETR under the independent CSC wrapper with D1 Binding support."""

from __future__ import annotations

import torch

from sim_detr.semantic_calibration.semantic_model import SimDETRWithSemanticCalibration

from .criterion import NullSafeCriterion
from .gmr_adapter import SoccerGMRModel


def build_soccer_gmr_model(opt):
    from sim_detr.model import build_model

    # Use native_mask_logits for base wrapper if d1_attention is chosen so base doesn't crash on unrecognized choice
    base_evidence_source = (
        "native_mask_logits"
        if opt.semantic_evidence_source == "d1_attention"
        else opt.semantic_evidence_source
    )
    base_model, native_criterion = build_model(opt)
    semantic_model = SimDETRWithSemanticCalibration(
        base_model=base_model,
        hidden_dim=opt.semantic_hidden_dim,
        dropout=opt.semantic_dropout,
        semantic_scale_init=opt.semantic_scale_init,
        semantic_variant=opt.semantic_variant,
        detach_support=opt.semantic_detach_support,
        diagnostic_mode=opt.semantic_diagnostic_mode,
        evidence_source=base_evidence_source,
    )
    semantic_model.semantic_context_variant = opt.semantic_context_variant
    semantic_model.semantic_counterfactual_seed = opt.semantic_counterfactual_seed
    model = SoccerGMRModel(
        semantic_model,
        hidden_dim=opt.hidden_dim,
        exist_pool=opt.exist_pool,
        evidence_source=opt.semantic_evidence_source,
        binding_loss_coef=opt.binding_loss_coef,
    )
    return model, NullSafeCriterion(
        native_criterion,
        exist_loss_coef=opt.exist_loss_coef,
        saliency_loss_coef=opt.lw_saliency,
        saliency_margin=opt.saliency_margin,
        background_focal_weight=opt.background_focal_weight,
        null_background_focal_weight=opt.null_background_focal_weight,
        null_iou_loss_weight=opt.null_iou_loss_weight,
        binding_loss_coef=opt.binding_loss_coef,
    )


def load_checkpoint_strict(model, path):
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=True)
    return checkpoint
