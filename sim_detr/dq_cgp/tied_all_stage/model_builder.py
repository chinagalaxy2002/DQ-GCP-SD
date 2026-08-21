"""Model builder for Sim-DETR + tied all-interlayer DQ-CGP.

This module mirrors ``sim_detr.model.build_model`` but returns a
``SimDETRWithTiedAllStageDQCGP`` instead of ``SimDETR``. It also extends the criterion's
``loss_map`` and ``weight_dict`` with the DQ-CGP losses.

Usage in a training script::

    from sim_detr.dq_cgp.tied_all_stage.model_builder import build_model_dq_cgp
    model, criterion = build_model_dq_cgp(args)

The ``args`` object must be the usual Sim-DETR ``opt`` namespace, optionally
extended with the DQ-CGP flags added by ``add_dq_cgp_args(parser)``.
"""

from __future__ import annotations

import argparse
from typing import Tuple

import torch

from sim_detr.model import SetCriterion, MLP, LinearLayer
from sim_detr.matcher import build_matcher
from sim_detr.transformer import build_transformer
from sim_detr.position_encoding import build_position_encoding
from .sim_detr_dq_cgp import SimDETRWithTiedAllStageDQCGP
from .loss import loss_query_cgp


# ---------------------------------------------------------------------------
# Argument extension
# ---------------------------------------------------------------------------

def add_dq_cgp_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add all DQ-CGP configuration flags to an existing argument parser.

    Call this immediately after constructing the base ``BaseOptions`` parser.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The Sim-DETR argument parser (already initialised via
        ``BaseOptions.initialize()``).

    Returns
    -------
    argparse.ArgumentParser
        The same parser with DQ-CGP flags added.
    """
    g = parser.add_argument_group("DQ-CGP")
    g.add_argument(
        "--use_query_cgp",
        action="store_true",
        help="Enable DQ-CGP inter-decoder-layer adapter.",
    )
    g.add_argument(
        "--query_cgp_num_basis",
        type=int,
        default=16,
        help="Number of shared prompt bases K (default 16).",
    )
    g.add_argument(
        "--query_cgp_prompt_length",
        type=int,
        default=6,
        help="Number of tokens per basis prompt (default 6).",
    )
    g.add_argument(
        "--query_cgp_router_hidden_dim",
        type=int,
        default=256,
        help="RCG router MLP hidden dimension (default 256).",
    )
    g.add_argument(
        "--query_cgp_frf_hidden_dim",
        type=int,
        default=512,
        help="FRF MLP hidden dimension (default 512).",
    )
    g.add_argument(
        "--query_cgp_temperature",
        type=float,
        default=1.0,
        help="Routing softmax temperature (default 1.0).",
    )
    g.add_argument(
        "--query_cgp_beta",
        type=float,
        default=0.05,
        help="Fixed residual injection strength β (default 0.05).",
    )
    g.add_argument(
        "--query_cgp_after_layer",
        type=int,
        default=0,
        help=(
            "Insert DQ-CGP after this decoder layer (0-indexed). "
            "Default 0 → D1→DQ→D2→D3→D4."
        ),
    )
    g.add_argument(
        "--query_cgp_middle_after_layer",
        type=int,
        default=1,
        help=(
            "Reuse the tied DQ-CGP after this decoder layer "
            "(0-indexed). Default 1 -> D2->DQ-middle->D3."
        ),
    )
    g.add_argument(
        "--query_cgp_late_after_layer",
        type=int,
        default=2,
        help=(
            "Reuse the tied DQ-CGP after this decoder layer "
            "(0-indexed). Default 2 -> D3->DQ-late->D4."
        ),
    )
    g.add_argument(
        "--query_cgp_binding_loss_coef",
        type=float,
        default=0.2,
        help="Weight for the temporal binding loss (default 0.2).",
    )
    g.add_argument(
        "--query_cgp_route_loss_coef",
        type=float,
        default=0.01,
        help="Weight for the routing diversity loss (default 0.01).",
    )
    g.add_argument(
        "--init_from",
        type=str,
        default=None,
        help=(
            "Path to a baseline Sim-DETR checkpoint to warm-start DQ-CGP. "
            "Only query_cgp.* keys are allowed to be absent."
        ),
    )
    g.add_argument(
        "--eval_disable_query_cgp_early",
        action="store_true",
        help="Inference only: skip the tied adapter call after the early layer.",
    )
    g.add_argument(
        "--eval_disable_query_cgp_middle",
        action="store_true",
        help="Inference only: skip the tied adapter call after the middle layer.",
    )
    g.add_argument(
        "--eval_disable_query_cgp_late",
        action="store_true",
        help="Inference only: skip the tied adapter call after the late layer.",
    )
    return parser


# ---------------------------------------------------------------------------
# Criterion extension (adds DQ-CGP loss to SetCriterion)
# ---------------------------------------------------------------------------

class DQCGPSetCriterion(SetCriterion):
    """SetCriterion extended with DQ-CGP binding and routing losses.

    Parameters
    ----------
    clip_len : int
        Clip length in seconds, forwarded to the binding loss.
    All other parameters are identical to ``SetCriterion``.
    """

    def __init__(self, *args, clip_len: int = 2, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._clip_len = int(clip_len)

    def loss_query_cgp(self, outputs, targets, indices, log=True):
        """Compute DQ-CGP binding + routing losses."""
        return loss_query_cgp(
            outputs=outputs,
            targets=targets,
            indices=indices,
            clip_len=self._clip_len,
            span_loss_type=self.span_loss_type,
        )

    def get_loss(self, loss, outputs, targets, indices, **kwargs):
        loss_map = {
            "spans": self.loss_spans,
            "labels": self.loss_labels,
            "contrastive_align": self.loss_contrastive_align,
            "saliency": self.loss_saliency,
            "mask_iou": self.loss_mask_iou,
            "iou_scores": self.loss_iou_scores,
            "query_cgp": self.loss_query_cgp,
        }
        assert loss in loss_map, f"Unknown loss: {loss}"
        return loss_map[loss](outputs, targets, indices, **kwargs)

    def forward(self, outputs, targets):
        """Loss forward with DQ-CGP losses skipped for aux outputs."""
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        if self.use_matcher:
            indices = self.matcher(outputs_without_aux, targets)
            losses_target = self.losses
        else:
            indices = None
            losses_target = ["saliency"]

        losses = {}
        for loss in losses_target:
            losses.update(self.get_loss(loss, outputs, targets, indices))

        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                if self.use_matcher:
                    indices_aux = self.matcher(aux_outputs, targets)
                    losses_target_aux = self.losses
                else:
                    indices_aux = None
                    losses_target_aux = ["saliency"]

                for loss in losses_target_aux:
                    # saliency and query_cgp are top-level only
                    if loss in {"saliency", "query_cgp"}:
                        continue
                    kwargs = {}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices_aux, **kwargs)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_model_dq_cgp(args) -> Tuple[SimDETRWithTiedAllStageDQCGP, DQCGPSetCriterion]:
    """Build Sim-DETR + DQ-CGP model and criterion.

    Mirrors ``sim_detr.model.build_model`` but produces ``SimDETRWithTiedAllStageDQCGP``
    and ``DQCGPSetCriterion``.

    Parameters
    ----------
    args : namespace
        Parsed argument namespace.  Must contain all standard Sim-DETR args
        plus the DQ-CGP args added by ``add_dq_cgp_args``.  Missing DQ-CGP
        args fall back to their defaults via ``getattr(..., default)``.

    Returns
    -------
    model : SimDETRWithTiedAllStageDQCGP
    criterion : DQCGPSetCriterion
    """
    device = torch.device(args.device)

    transformer = build_transformer(args)
    position_embedding, txt_position_embedding = build_position_encoding(args)

    # Helper to read DQ-CGP args gracefully (works even if the arg is absent
    # because an old opt.json is being reloaded).
    def _get(name, default):
        return getattr(args, name, default)

    common_kwargs = dict(
        transformer=transformer,
        position_embed=position_embedding,
        txt_position_embed=txt_position_embedding,
        txt_dim=args.t_feat_dim,
        vid_dim=args.v_feat_dim,
        num_queries=args.num_queries,
        input_dropout=args.input_dropout,
        aux_loss=args.aux_loss,
        contrastive_align_loss=args.contrastive_align_loss,
        contrastive_hdim=args.contrastive_hdim,
        max_v_l=args.max_v_l,
        span_loss_type=args.span_loss_type,
        use_txt_pos=args.use_txt_pos,
        n_input_proj=args.n_input_proj,
        clip_len=args.clip_length,
        # DQ-CGP flags
        use_query_cgp=_get("use_query_cgp", False),
        query_cgp_num_basis=_get("query_cgp_num_basis", 16),
        query_cgp_prompt_length=_get("query_cgp_prompt_length", 6),
        query_cgp_router_hidden_dim=_get("query_cgp_router_hidden_dim", 256),
        query_cgp_frf_hidden_dim=_get("query_cgp_frf_hidden_dim", 512),
        query_cgp_temperature=_get("query_cgp_temperature", 1.0),
        query_cgp_beta=_get("query_cgp_beta", 0.05),
        query_cgp_after_layer=_get("query_cgp_after_layer", 0),
        query_cgp_middle_after_layer=_get("query_cgp_middle_after_layer", 1),
        query_cgp_late_after_layer=_get("query_cgp_late_after_layer", 2),
        query_cgp_binding_loss_coef=_get("query_cgp_binding_loss_coef", 0.2),
        query_cgp_route_loss_coef=_get("query_cgp_route_loss_coef", 0.01),
    )

    if args.a_feat_dir is None:
        model = SimDETRWithTiedAllStageDQCGP(**common_kwargs)
    else:
        model = SimDETRWithTiedAllStageDQCGP(
            aud_dim=args.a_feat_dim,
            **common_kwargs,
        )

    # ---- Matcher ----
    matcher = build_matcher(args)

    # ---- Weight dict ----
    weight_dict = {
        "loss_span": args.span_loss_coef,
        "loss_giou": args.giou_loss_coef,
        "loss_label": args.label_loss_coef,
        "loss_saliency": args.lw_saliency,
        "loss_mask_iou": args.mask_loss_coef,
        "loss_iou_scores": args.iou_scores_loss_coef,
    }
    if args.contrastive_align_loss:
        weight_dict["loss_contrastive_align"] = args.contrastive_align_loss_coef

    if _get("use_query_cgp", False):
        weight_dict["loss_query_cgp_bind"] = _get("query_cgp_binding_loss_coef", 0.2)
        weight_dict["loss_query_cgp_route"] = _get("query_cgp_route_loss_coef", 0.01)

    # Auxiliary losses
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update(
                {
                    k + f"_{i}": v
                    for k, v in weight_dict.items()
                    if k not in {"loss_saliency", "loss_query_cgp_bind", "loss_query_cgp_route"}
                }
            )
        weight_dict.update(aux_weight_dict)

    weight_dict["loss_mask_iou"] = 0  # original Sim-DETR sets this to 0

    # ---- Losses list ----
    losses = ["spans", "labels", "saliency", "mask_iou", "iou_scores"]
    if args.contrastive_align_loss:
        losses.append("contrastive_align")
    if _get("use_query_cgp", False):
        losses.append("query_cgp")

    use_matcher = not (args.dset_name == "tvsum")

    criterion = DQCGPSetCriterion(
        matcher=matcher,
        weight_dict=weight_dict,
        losses=losses,
        eos_coef=args.eos_coef,
        temperature=args.temperature,
        span_loss_type=args.span_loss_type,
        max_v_l=args.max_v_l,
        saliency_margin=args.saliency_margin,
        use_matcher=use_matcher,
        clip_len=args.clip_length,
    )
    criterion.to(device)

    # ---- Optional: warm-start from baseline checkpoint ----
    init_from = _get("init_from", None)
    if init_from is not None:
        SimDETRWithTiedAllStageDQCGP.load_from_baseline_checkpoint(
            init_from, model, device="cpu"
        )

    return model, criterion


__all__ = [
    "add_dq_cgp_args",
    "build_model_dq_cgp",
    "DQCGPSetCriterion",
]
