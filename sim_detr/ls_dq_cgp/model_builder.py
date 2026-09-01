"""Arguments, model construction and criterion for Sim-DETR + LS-DQ-CGP."""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from sim_detr.matcher import build_matcher
from sim_detr.model import SetCriterion
from sim_detr.position_encoding import build_position_encoding
from sim_detr.transformer import build_transformer
from .loss import binding_loss
from .model import SimDETRWithLSDQCGP


def add_ls_dq_cgp_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("LS-DQ-CGP")
    group.add_argument("--ls_num_basis", type=int, default=16)
    group.add_argument("--ls_prompt_length", type=int, default=6)
    group.add_argument("--ls_router_hidden_dim", type=int, default=256)
    group.add_argument("--ls_frf_hidden_dim", type=int, default=512)
    group.add_argument("--ls_temperature", type=float, default=1.0)
    group.add_argument("--ls_initial_scale", type=float, default=10.0)
    group.add_argument("--ls_initial_bias", type=float, default=-2.0)
    group.add_argument("--ls_binding_loss_coef", type=float, default=0.2)
    group.add_argument("--use_ls_exist_head", action="store_true")
    group.add_argument("--ls_exist_loss_coef", type=float, default=1.0)
    group.add_argument(
        "--init_from",
        type=str,
        default=None,
        help="Warm-start from a plain Sim-DETR checkpoint (weights only).",
    )
    counterfactual = parser.add_mutually_exclusive_group()
    counterfactual.add_argument("--ls_static_bypass", action="store_true")
    counterfactual.add_argument("--ls_context_roll", action="store_true")
    return parser


class LSDQCGPSetCriterion(SetCriterion):
    def loss_iou_scores(self, outputs, targets, indices):
        if "native_iou_scores" not in outputs:
            return super().loss_iou_scores(outputs, targets, indices)
        native_outputs = dict(outputs)
        native_outputs["iou_scores"] = outputs["native_iou_scores"]
        return super().loss_iou_scores(native_outputs, targets, indices)

    def loss_ls_binding(self, outputs, targets, indices, **kwargs):
        del kwargs
        if "ls_d1_attention" not in outputs:
            return {"loss_ls_bind": outputs["pred_logits"].sum() * 0.0}
        return {"loss_ls_bind": binding_loss(
            outputs["ls_d1_attention"],
            outputs["video_mask"],
            targets,
            indices,
            span_loss_type=self.span_loss_type,
        )}

    def loss_ls_exist(self, outputs, targets, indices, **kwargs):
        del indices, kwargs
        if "pred_exist_logits" not in outputs:
            return {"loss_ls_exist": outputs["pred_logits"].sum() * 0.0}
        if "exist_label" not in targets:
            raise KeyError(
                "use_ls_exist_head requires video-query-level targets['exist_label']; "
                "the stock QVHighlights loader does not provide them"
            )
        labels = targets["exist_label"].to(outputs["pred_exist_logits"]).float()
        return {"loss_ls_exist": F.binary_cross_entropy_with_logits(
            outputs["pred_exist_logits"], labels
        )}

    def get_loss(self, loss, outputs, targets, indices, **kwargs):
        loss_map = {
            "spans": self.loss_spans,
            "labels": self.loss_labels,
            "contrastive_align": self.loss_contrastive_align,
            "saliency": self.loss_saliency,
            "mask_iou": self.loss_mask_iou,
            "iou_scores": self.loss_iou_scores,
            "ls_binding": self.loss_ls_binding,
            "ls_exist": self.loss_ls_exist,
        }
        if loss not in loss_map:
            raise ValueError(f"unknown loss: {loss}")
        return loss_map[loss](outputs, targets, indices, **kwargs)

    def forward(self, outputs, targets):
        final_outputs = {k: v for k, v in outputs.items() if k != "aux_outputs"}
        if self.use_matcher:
            indices = self.matcher(final_outputs, targets)
            active_losses = self.losses
        else:
            indices = None
            active_losses = ["saliency"]
        losses = {}
        for loss_name in active_losses:
            losses.update(self.get_loss(loss_name, outputs, targets, indices))

        for layer_index, auxiliary in enumerate(outputs.get("aux_outputs", [])):
            aux_indices = self.matcher(auxiliary, targets) if self.use_matcher else None
            aux_losses = self.losses if self.use_matcher else ["saliency"]
            for loss_name in aux_losses:
                if loss_name in {"saliency", "ls_binding", "ls_exist"}:
                    continue
                values = self.get_loss(loss_name, auxiliary, targets, aux_indices)
                losses.update({f"{key}_{layer_index}": value for key, value in values.items()})
        return losses


def build_model_ls_dq_cgp(args):
    if int(args.dec_layers) != 2:
        raise ValueError("LS-DQ-CGP experiments require --dec_layers 2")
    transformer = build_transformer(args)
    position, text_position = build_position_encoding(args)
    kwargs = dict(
        transformer=transformer,
        position_embed=position,
        txt_position_embed=text_position,
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
        ls_num_basis=getattr(args, "ls_num_basis", 16),
        ls_prompt_length=getattr(args, "ls_prompt_length", 6),
        ls_router_hidden_dim=getattr(args, "ls_router_hidden_dim", 256),
        ls_frf_hidden_dim=getattr(args, "ls_frf_hidden_dim", 512),
        ls_temperature=getattr(args, "ls_temperature", 1.0),
        ls_initial_scale=getattr(args, "ls_initial_scale", 10.0),
        ls_initial_bias=getattr(args, "ls_initial_bias", -2.0),
        use_ls_exist_head=getattr(args, "use_ls_exist_head", False),
    )
    if args.a_feat_dir is not None:
        kwargs["aud_dim"] = args.a_feat_dim
    model = SimDETRWithLSDQCGP(**kwargs)
    matcher = build_matcher(args)
    weight_dict = {
        "loss_span": args.span_loss_coef,
        "loss_giou": args.giou_loss_coef,
        "loss_label": args.label_loss_coef,
        "loss_saliency": args.lw_saliency,
        "loss_mask_iou": 0.0,
        "loss_iou_scores": args.iou_scores_loss_coef,
        "loss_ls_bind": getattr(args, "ls_binding_loss_coef", 0.2),
    }
    if args.contrastive_align_loss:
        weight_dict["loss_contrastive_align"] = args.contrastive_align_loss_coef
    if getattr(args, "use_ls_exist_head", False):
        weight_dict["loss_ls_exist"] = getattr(args, "ls_exist_loss_coef", 1.0)
    if args.aux_loss:
        for key, value in list(weight_dict.items()):
            if key not in {"loss_saliency", "loss_ls_bind", "loss_ls_exist"}:
                weight_dict[f"{key}_0"] = value

    losses = ["spans", "labels", "saliency", "mask_iou", "iou_scores", "ls_binding"]
    if args.contrastive_align_loss:
        losses.append("contrastive_align")
    if getattr(args, "use_ls_exist_head", False):
        losses.append("ls_exist")
    criterion = LSDQCGPSetCriterion(
        matcher=matcher,
        weight_dict=weight_dict,
        losses=losses,
        eos_coef=args.eos_coef,
        temperature=args.temperature,
        span_loss_type=args.span_loss_type,
        max_v_l=args.max_v_l,
        saliency_margin=args.saliency_margin,
        use_matcher=args.dset_name != "tvsum",
    )
    criterion.to(torch.device(args.device))
    return model, criterion
