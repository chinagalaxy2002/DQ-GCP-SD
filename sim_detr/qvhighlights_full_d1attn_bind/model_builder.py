"""Build the isolated QVHighlights Full + D1-attention + Binding model."""

from __future__ import annotations

import torch

from sim_detr.semantic_calibration.semantic_model import SimDETRWithSemanticCalibration

from .criterion import BindingCriterion
from .model import QVHighlightsFullD1BindingModel


def build_model(opt):
    from sim_detr.model import build_model as build_native_model

    base_model, native_criterion = build_native_model(opt)
    semantic_model = SimDETRWithSemanticCalibration(
        base_model=base_model,
        hidden_dim=opt.semantic_hidden_dim,
        dropout=opt.semantic_dropout,
        semantic_scale_init=opt.semantic_scale_init,
        semantic_variant="full",
        detach_support=opt.semantic_detach_support,
        diagnostic_mode=False,
        evidence_source="native_mask_logits",
    )
    semantic_model.semantic_context_variant = "aligned"
    semantic_model.semantic_counterfactual_seed = opt.seed
    model = QVHighlightsFullD1BindingModel(semantic_model)
    criterion = BindingCriterion(native_criterion, opt.binding_loss_coef)
    return model, criterion


def load_checkpoint(model, path):
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=True)
    return checkpoint
