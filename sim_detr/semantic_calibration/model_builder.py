"""Build and load the independent semantic-calibration model."""

from __future__ import annotations

import logging

import torch

from .semantic_model import SimDETRWithSemanticCalibration

logger = logging.getLogger(__name__)


def build_model_semantic(args):
    # Lazy import keeps module-level unit tests independent of optional native
    # torchvision installation; actual training still uses the exact baseline
    # builder.
    from sim_detr.model import build_model as build_native_model

    base_model, criterion = build_native_model(args)
    model = SimDETRWithSemanticCalibration(
        base_model=base_model,
        hidden_dim=getattr(args, "semantic_hidden_dim", args.hidden_dim),
        dropout=getattr(args, "semantic_dropout", 0.1),
        semantic_scale_init=getattr(args, "semantic_scale_init", 1.0),
        semantic_variant=getattr(args, "semantic_variant", "full"),
        detach_support=getattr(args, "semantic_detach_support", True),
        diagnostic_mode=getattr(args, "semantic_diagnostic_mode", False),
    )
    model.semantic_context_variant = getattr(args, "semantic_context_variant", "aligned")
    model.semantic_scale_override = getattr(args, "semantic_scale_override", None)
    return model, criterion


def _state_dict_from_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu")
    return checkpoint.get("model", checkpoint)


def load_native_checkpoint(model, checkpoint_path):
    """Strictly load a native Sim-DETR checkpoint into the wrapped base model."""
    source = _state_dict_from_checkpoint(checkpoint_path)
    source = {
        key[7:] if key.startswith("module.") else key: value
        for key, value in source.items()
    }
    target_keys = set(model.base_model.state_dict().keys())
    unexpected = sorted(set(source) - target_keys)
    missing = sorted(target_keys - set(source))
    if unexpected or missing:
        raise RuntimeError(
            "Native checkpoint is not architecture-compatible. "
            f"unexpected={unexpected[:10]}, missing={missing[:10]}"
        )
    model.base_model.load_state_dict(source, strict=True)
    logger.info("Loaded native Sim-DETR checkpoint: %s", checkpoint_path)


def load_semantic_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "model" not in checkpoint:
        raise RuntimeError(f"Checkpoint has no model state: {checkpoint_path}")
    model.load_state_dict(checkpoint["model"], strict=True)
    return checkpoint
