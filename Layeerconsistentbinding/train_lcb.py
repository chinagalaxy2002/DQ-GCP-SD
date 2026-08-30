"""Train Sim-DETR with Layer-Consistent Binding (LCB-Full).

This training entrypoint implements the official Layer-Consistent Binding protocol:
- Native Sim-DETR architecture without prompt/router/residual modifications.
- Native cross-attention captured across decoder layers D1–D4.
- Hungarian matching on final D4 predictions only.
- Multi-layer matched binding loss (L_layer_bind, coef=0.5).
- Occurrence-level consistency loss (L_owner_cons, coef=0.1).
- Anti-washout hinge loss (L_drop, coef=0.1, margin=0.05).
- Validation-selected 200 epochs run on seed 2017.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from causal_occurrence_lab.capture_attention import NativeCrossAttentionCapture
from causal_occurrence_lab.common import REPO_ROOT, build_dataset, save_json, set_seed
from Layeerconsistentbinding.controls import (
    install_layer_consistent_binding_control,
    remove_layer_consistent_binding_control,
)


LCB_VARIANTS = {
    "lcb_full": {
        "description": "Layer-Consistent Binding on native D1–D4 Sim-DETR attention",
        "use_dq": False,
        "native": "layer_consistent",
        "layer_bind_coef": 0.5,
        "owner_cons_coef": 0.1,
        "drop_coef": 0.1,
        "drop_margin": 0.05,
    },
    "native_bind_d1": {
        "description": "D1-only native binding (reference comparison)",
        "use_dq": False,
        "native": "d1_only",
        "layer_bind_coef": 0.5,
        "owner_cons_coef": 0.0,
        "drop_coef": 0.0,
        "drop_margin": 0.05,
    },
    "baseline": {
        "description": "Clean Sim-DETR baseline without binding regularizers",
        "use_dq": False,
        "native": "none",
        "layer_bind_coef": 0.0,
        "owner_cons_coef": 0.0,
        "drop_coef": 0.0,
        "drop_margin": 0.05,
    },
}


def _escape_percent_help(parser):
    for action in parser._actions:
        if isinstance(action.help, str):
            action.help = action.help.replace("%", "%%")


def build_parser():
    from sim_detr.config import BaseOptions
    from sim_detr.dq_cgp.model_builder import add_dq_cgp_args

    base = BaseOptions()
    base.initialize()
    add_dq_cgp_args(base.parser)
    _escape_percent_help(base.parser)
    parser = base.parser

    group = parser.add_argument_group("Layer-Consistent Binding (LCB)")
    group.add_argument(
        "--variant",
        choices=sorted(LCB_VARIANTS),
        default="lcb_full",
        help="LCB training variant (default: lcb_full)",
    )
    group.add_argument(
        "--output-dir",
        required=True,
        help="Directory to save checkpoints, logs, and variant config",
    )
    group.add_argument(
        "--causal-dec-layers",
        type=int,
        default=4,
        help="Number of decoder layers (default: 4)",
    )
    group.add_argument(
        "--lcb-layer-bind-coef",
        type=float,
        default=None,
        help="Override L_layer_bind coefficient (default: 0.5 for lcb_full)",
    )
    group.add_argument(
        "--lcb-owner-cons-coef",
        type=float,
        default=None,
        help="Override L_owner_cons coefficient (default: 0.1 for lcb_full)",
    )
    group.add_argument(
        "--lcb-drop-coef",
        type=float,
        default=None,
        help="Override L_drop coefficient (default: 0.1 for lcb_full)",
    )
    group.add_argument(
        "--lcb-drop-margin",
        type=float,
        default=None,
        help="Override drop margin delta (default: 0.05)",
    )
    return parser


def finalize_options(args):
    cfg = LCB_VARIANTS[args.variant]
    args.dec_layers = int(args.causal_dec_layers)
    args.seed = int(args.seed)
    args.use_query_cgp = False

    args.lcb_layer_bind_coef = (
        cfg["layer_bind_coef"]
        if args.lcb_layer_bind_coef is None
        else float(args.lcb_layer_bind_coef)
    )
    args.lcb_owner_cons_coef = (
        cfg["owner_cons_coef"]
        if args.lcb_owner_cons_coef is None
        else float(args.lcb_owner_cons_coef)
    )
    args.lcb_drop_coef = (
        cfg["drop_coef"]
        if args.lcb_drop_coef is None
        else float(args.lcb_drop_coef)
    )
    args.lcb_drop_margin = (
        cfg["drop_margin"]
        if args.lcb_drop_margin is None
        else float(args.lcb_drop_margin)
    )

    args.device = torch.device(
        "cuda" if int(args.device) >= 0 and torch.cuda.is_available() else "cpu"
    )
    if args.device.type == "cuda":
        torch.cuda.set_device(int(args.gpu_id))

    args.pin_memory = not args.no_pin_memory
    args.use_tef = "tef" in args.ctx_mode
    args.use_video = "video" in args.ctx_mode
    if not args.use_video:
        args.v_feat_dim = 0
    if args.use_tef:
        args.v_feat_dim += 2

    out_dir = Path(args.output_dir).expanduser()
    if not out_dir.is_absolute():
        out_dir = (Path.cwd() / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir = str(out_dir)
    args.ckpt_filepath = str(out_dir / "model.ckpt")
    args.train_log_filepath = str(out_dir / "train.log.txt")
    args.eval_log_filepath = str(out_dir / "eval.log.txt")
    args.tensorboard_log_dir = str(out_dir / "tensorboard_log")
    args.exp_id = args.variant
    args.eval_bsz = int(args.eval_bsz)
    return args


def build_lcb_model(args):
    """Build standard Sim-DETR model and install LCB controls on the criterion."""
    from sim_detr.model import build_model

    model, criterion = build_model(args)
    capture = None
    cfg = LCB_VARIANTS[args.variant]

    if cfg["native"] == "layer_consistent":
        capture = NativeCrossAttentionCapture(model).install()
        install_layer_consistent_binding_control(
            criterion,
            capture,
            layer_bind_coef=args.lcb_layer_bind_coef,
            owner_cons_coef=args.lcb_owner_cons_coef,
            drop_coef=args.lcb_drop_coef,
            drop_margin=args.lcb_drop_margin,
            layers=tuple(range(args.dec_layers)),
        )
    elif cfg["native"] == "d1_only":
        capture = NativeCrossAttentionCapture(model).install()
        install_layer_consistent_binding_control(
            criterion,
            capture,
            layer_bind_coef=args.lcb_layer_bind_coef,
            owner_cons_coef=0.0,
            drop_coef=0.0,
            drop_margin=args.lcb_drop_margin,
            layers=(0,),
        )

    if getattr(args, "resume", None):
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"], strict=True)

    model.to(args.device)
    criterion.to(args.device)
    return model, criterion, capture


def run(args):
    args = finalize_options(args)
    set_seed(args.seed)
    variant_cfg = LCB_VARIANTS[args.variant]

    save_json(args.results_dir + "/variant.json", {
        "variant": args.variant,
        "description": variant_cfg["description"],
        "seed": args.seed,
        "decoder_layers": args.dec_layers,
        "encoder_layers": args.enc_layers,
        "batch_size": args.bsz,
        "lr": args.lr,
        "lr_drop": args.lr_drop,
        "max_es_cnt": args.max_es_cnt,
        "vtc_loss_coef": getattr(args, "VTC_loss_coef", getattr(args, "vtc_loss_coef", 0.3)),
        "ctc_loss_coef": getattr(args, "CTC_loss_coef", getattr(args, "ctc_loss_coef", 0.5)),
        "layer_bind_coef": args.lcb_layer_bind_coef,
        "owner_cons_coef": args.lcb_owner_cons_coef,
        "drop_coef": args.lcb_drop_coef,
        "drop_margin": args.lcb_drop_margin,
        "native_mode": variant_cfg["native"],
        "checkpoint_selection": "val MR-full-mAP",
    })

    logging.basicConfig(
        filename=str(Path(args.results_dir) / "lcb_train.log"),
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logger = logging.getLogger("layer_consistent_binding")
    logger.info("Starting LCB training variant %s with options %s", args.variant, vars(args))

    model, criterion, capture = build_lcb_model(args)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.wd,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    resume_metadata = {}

    if getattr(args, "resume", None) and getattr(args, "resume_all", False):
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        if "optimizer" not in checkpoint or "lr_scheduler" not in checkpoint:
            raise RuntimeError("--resume_all requires optimizer and scheduler state")
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["lr_scheduler"])
        args.start_epoch = int(checkpoint["epoch"]) + 1
        resume_metadata = {
            "best_metric": checkpoint.get("best_metric", checkpoint.get("metric", 0.0)),
            "es_cnt": checkpoint.get("es_cnt", 0),
        }

    from sim_detr.dq_cgp.train_dq_cgp import train as production_train

    train_dataset = build_dataset(args, args.train_path, load_labels=True)
    eval_dataset = (
        build_dataset(args, args.eval_path, load_labels=True)
        if args.eval_path
        else None
    )

    try:
        best = production_train(
            model,
            criterion,
            optimizer,
            scheduler,
            train_dataset,
            eval_dataset,
            args,
            resume_metadata=resume_metadata,
        )
        logger.info("Finished LCB variant %s, best validation MR-full-mAP=%.4f", args.variant, best)
        return best
    finally:
        if capture is not None:
            capture.remove()


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
