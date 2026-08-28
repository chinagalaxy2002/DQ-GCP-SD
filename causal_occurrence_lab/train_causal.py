"""Train controlled Sim-DETR variants with the release protocol.

The training loop is imported from ``sim_detr.dq_cgp.train_dq_cgp`` so the
optimizer, VTC/CTC terms, clipping, validation selection, and checkpoint
format remain the production ones.  Only the model/criterion construction and
runtime causal controls are selected here.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from causal_occurrence_lab.capture_attention import NativeCrossAttentionCapture
from causal_occurrence_lab.common import REPO_ROOT, build_dataset, save_json, set_seed
from causal_occurrence_lab.controls import (
    install_injection_control,
    install_native_binding_control,
    install_query_cgp_loss_control,
)


VARIANTS = {
    "baseline": {
        "use_dq": False, "binding": 0.0, "route": 0.0,
        "inject": False, "target": None, "native": False,
    },
    "full": {
        "use_dq": True, "binding": 0.2, "route": 0.01,
        "inject": True, "target": "matched", "native": False,
    },
    "no_bind": {
        "use_dq": True, "binding": 0.0, "route": 0.01,
        "inject": True, "target": "matched", "native": False,
    },
    "supervision_only": {
        "use_dq": True, "binding": 0.2, "route": 0.0,
        "inject": False, "target": "matched", "native": False,
    },
    "union_bind": {
        "use_dq": True, "binding": 0.2, "route": 0.01,
        "inject": True, "target": "union", "native": False,
    },
    "no_route": {
        "use_dq": True, "binding": 0.2, "route": 0.0,
        "inject": True, "target": "matched", "native": False,
    },
    "architecture_only": {
        "use_dq": True, "binding": 0.0, "route": 0.0,
        "inject": True, "target": "matched", "native": False,
    },
    "wrong_bind": {
        "use_dq": True, "binding": 0.2, "route": 0.01,
        "inject": True, "target": "rolled", "native": False,
    },
    "native_bind": {
        "use_dq": False, "binding": 0.2, "route": 0.0,
        "inject": False, "target": None, "native": True,
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
    group = parser.add_argument_group("causal occurrence lab")
    group.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    group.add_argument("--output-dir", required=True)
    group.add_argument("--causal-dec-layers", type=int, default=4)
    group.add_argument("--clean-iou", type=float, default=0.1)
    group.add_argument(
        "--init-from",
        "--causal-init-from",
        dest="causal_init_from",
        default=None,
        help="Baseline checkpoint used to initialise a newly built DQ model.",
    )
    # The production DQ arguments are added above for architecture
    # compatibility.  These three controls are intentionally re-exposed as
    # causal-lab overrides so a run can be described by one command line.
    parser.set_defaults(
        query_cgp_binding_loss_coef=None,
        query_cgp_route_loss_coef=None,
    )
    group.add_argument(
        "--query_cgp_inject",
        dest="causal_inject_override",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override whether the DQ residual is injected (default: variant).",
    )
    group.add_argument(
        "--query_cgp_binding_target",
        dest="causal_binding_target_override",
        choices=("matched", "union", "rolled"),
        default=None,
        help="Override the causal binding target (default: variant).",
    )
    return parser


def finalize_options(args):
    cfg = VARIANTS[args.variant]
    args.dec_layers = int(args.causal_dec_layers)
    args.seed = int(args.seed)
    args.use_query_cgp = bool(cfg["use_dq"])
    binding_coef = (
        cfg["binding"]
        if args.query_cgp_binding_loss_coef is None
        else args.query_cgp_binding_loss_coef
    )
    route_coef = (
        cfg["route"]
        if args.query_cgp_route_loss_coef is None
        else args.query_cgp_route_loss_coef
    )
    args.query_cgp_binding_loss_coef = float(binding_coef)
    args.query_cgp_route_loss_coef = float(route_coef)
    args.causal_inject = (
        bool(cfg["inject"])
        if args.causal_inject_override is None
        else bool(args.causal_inject_override)
    )
    args.causal_binding_target = (
        cfg["target"]
        if args.causal_binding_target_override is None
        else args.causal_binding_target_override
    )
    args.device = torch.device(
        "cuda" if int(args.device) >= 0 else "cpu"
    )
    if args.device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
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


def build_variant(args):
    """Build one model and install only the requested runtime controls."""

    cfg = VARIANTS[args.variant]
    capture = None
    if cfg["use_dq"]:
        from sim_detr.dq_cgp.model_builder import build_model_dq_cgp

        causal_init = args.causal_init_from or getattr(args, "init_from", None)
        # The production builder also understands init_from.  We apply it
        # below after construction so every causal variant follows one path.
        original_init_from = getattr(args, "init_from", None)
        args.init_from = None
        model, criterion = build_model_dq_cgp(args)
        args.init_from = original_init_from
        if causal_init:
            from sim_detr.dq_cgp.train_dq_cgp import _load_baseline_weights

            _load_baseline_weights(model, causal_init)
        if args.variant != "full" or not args.causal_inject:
            install_injection_control(model, args.causal_inject)
        if args.variant != "full" or args.causal_binding_target != "matched":
            install_query_cgp_loss_control(
                criterion, binding_target=args.causal_binding_target or "matched"
            )
    else:
        from sim_detr.model import build_model

        model, criterion = build_model(args)
        if cfg["native"]:
            capture = NativeCrossAttentionCapture(model).install()
            install_native_binding_control(
                criterion, capture,
                binding_coef=args.query_cgp_binding_loss_coef,
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
    variant = VARIANTS[args.variant]
    save_json(args.results_dir + "/variant.json", {
        "variant": args.variant,
        "seed": args.seed,
        "decoder_layers": args.dec_layers,
        "use_query_cgp": variant["use_dq"],
        "binding_coef": args.query_cgp_binding_loss_coef,
        "route_coef": args.query_cgp_route_loss_coef,
        "binding_target": args.causal_binding_target,
        "inject_residual": args.causal_inject,
        "native_binding": variant["native"],
        "checkpoint_selection": "val MR-full-mAP",
    })
    logging.basicConfig(
        filename=str(Path(args.results_dir) / "causal_train.log"),
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logger = logging.getLogger("causal_occurrence_lab")
    logger.info("Starting variant %s with options %s", args.variant, vars(args))

    model, criterion, capture = build_variant(args)
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

    # Import the production loop lazily so `--help` and unit tests do not
    # initialize TensorBoard or any dataset worker.
    from sim_detr.dq_cgp.train_dq_cgp import train as production_train

    train_dataset = build_dataset(args, args.train_path, load_labels=True)
    eval_dataset = build_dataset(args, args.eval_path, load_labels=True) if args.eval_path else None
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
    logger.info("Finished variant %s, best validation MR-full-mAP=%.4f", args.variant, best)
    if capture is not None:
        capture.remove()
    return best


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
