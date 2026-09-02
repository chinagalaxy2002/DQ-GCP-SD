"""Train Full + D1-attention evidence + Hungarian Binding on QVHighlights."""

from __future__ import annotations

import logging
import sys

import torch

from sim_detr.semantic_calibration.train_semantic import _build_datasets
from sim_detr.train import set_seed, train
from utils.model_utils import count_parameters

from .config import ExperimentOptions
from .model_builder import build_model, load_checkpoint

logger = logging.getLogger(__name__)


def _escape_argparse_percent(parser) -> None:
    for action in parser._actions:
        if isinstance(action.help, str):
            action.help = action.help.replace("%", "%%")


def parse_options(argv=None):
    options = ExperimentOptions()
    options.initialize()
    _escape_argparse_percent(options.parser)
    old_argv = sys.argv
    if argv is not None:
        sys.argv = [old_argv[0], *argv]
    try:
        opt = options.parse()
    finally:
        sys.argv = old_argv
    opt.semantic_variant = "full"
    opt.semantic_evidence_source = "d1_attention"
    opt.semantic_context_variant = "aligned"
    if opt.dec_layers < 2:
        raise ValueError("This experiment requires at least two decoder layers")
    return opt


def setup_model(opt):
    model, criterion = build_model(opt)
    model.to(opt.device)
    criterion.to(opt.device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=opt.lr,
        weight_decay=opt.wd,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, opt.lr_drop)
    if opt.resume is not None:
        checkpoint = load_checkpoint(model, opt.resume)
        if opt.resume_all:
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["lr_scheduler"])
            if opt.start_epoch is None:
                opt.start_epoch = int(checkpoint["epoch"]) + 1
    return model, criterion, optimizer, scheduler


def start_training(argv=None):
    opt = parse_options(argv)
    if opt.device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        torch.cuda.set_device(opt.gpu_id)
    set_seed(opt.seed)
    logger.info(
        "QVHighlights Full + D1 attention + Binding: seed=%d coef=%.4f dec=%d",
        opt.seed,
        opt.binding_loss_coef,
        opt.dec_layers,
    )
    train_dataset, eval_dataset = _build_datasets(opt)
    model, criterion, optimizer, scheduler = setup_model(opt)
    count_parameters(model)
    try:
        train(
            model,
            criterion,
            optimizer,
            scheduler,
            train_dataset,
            eval_dataset,
            opt,
        )
    finally:
        model.close()
    return opt.results_dir


if __name__ == "__main__":
    start_training()
