"""DQ-CGP training entry-point for Sim-DETR.

This script is a drop-in replacement for ``sim_detr/train.py`` that:
1. Adds the DQ-CGP argument group via ``add_dq_cgp_args``.
2. Uses ``build_model_dq_cgp`` instead of ``build_model``.
3. Keeps **all** original Sim-DETR training logic (VTC/CTC losses,
   gradient clipping, evaluation, checkpoint saving) unchanged.

No original Sim-DETR file is imported-and-overwritten; we import the original
helpers and reuse them verbatim.
"""

import os
import time
import json
import pprint
import random
import shutil
import numpy as np
from tqdm import tqdm, trange
from collections import defaultdict

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# -- Original Sim-DETR imports (unchanged) --
from sim_detr.config import BaseOptions
from sim_detr.start_end_dataset import (
    StartEndDataset, start_end_collate, prepare_batch_inputs,
)
from sim_detr.start_end_dataset_audio import (
    StartEndDataset_audio, start_end_collate_audio, prepare_batch_inputs_audio,
)
from sim_detr.inference import eval_epoch
from utils.basic_utils import AverageMeter, dict_to_markdown
from utils.model_utils import count_parameters
from sim_detr.loss_fun.VTCLoss import VTCLoss
from sim_detr.loss_fun.CTCLoss import CTC_Loss

# -- DQ-CGP imports --
from .model_builder import add_dq_cgp_args, build_model_dq_cgp
from .sim_detr_dq_cgp import ARCHITECTURE_ID
from .checkpoint_contract import validate_tied_checkpoint

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


def _escape_argparse_percent(parser):
    """Make the original Sim-DETR help strings safe for argparse.

    ``BaseOptions`` contains literal percent signs in two help strings.  Python's
    argparse treats help strings as %-format templates, so ``--help`` otherwise
    raises ``ValueError: unsupported format character``.  Keeping this local
    avoids modifying the baseline configuration module.
    """
    for action in parser._actions:
        if isinstance(action.help, str):
            action.help = action.help.replace("%", "%%")


def _checkpoint_payload(model, optimizer, lr_scheduler, opt, epoch, best_metric, es_cnt):
    """Return a checkpoint that supports strict inference and full resume."""
    return {
        "architecture_id": ARCHITECTURE_ID,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "epoch": int(epoch),
        "metric": float(best_metric),
        "best_metric": float(best_metric),
        "es_cnt": int(es_cnt),
        "model_cfg": dict(vars(opt)),
        # Retain the key used by the original Sim-DETR checkpoints.
        "opt": opt,
    }


def _load_baseline_weights(model, checkpoint_path):
    """Warm-start from an official Sim-DETR checkpoint, with strict validation.

    The native decoder hook retains every baseline state-dict key.  Therefore
    no decoder-prefix remapping is allowed: all source keys must map directly,
    and only the newly introduced ``query_cgp.*`` prefix may be absent.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    source_state = checkpoint.get("model", checkpoint)
    normalized_source_keys = [
        key[7:] if key.startswith("module.") else key
        for key in source_state
    ]
    dq_prefix = "query_cgp."
    if any(key.startswith(dq_prefix) for key in normalized_source_keys):
        raise RuntimeError(
            "--init_from expects a baseline Sim-DETR checkpoint without "
            "DQ-CGP tensors. Use --resume for a tied dual-stage checkpoint."
        )
    target_state = model.state_dict()
    remapped = {}
    unmapped = []

    for source_key, value in source_state.items():
        key = source_key[7:] if source_key.startswith("module.") else source_key
        if key in target_state:
            remapped[key] = value
        else:
            unmapped.append(source_key)

    if unmapped:
        raise RuntimeError(
            "Baseline checkpoint contains keys that do not map to Sim-DETR + "
            f"DQ-CGP (first 20): {unmapped[:20]}"
        )

    missing, unexpected = model.load_state_dict(remapped, strict=False)
    disallowed_missing = [k for k in missing if not k.startswith(dq_prefix)]
    if disallowed_missing or unexpected:
        raise RuntimeError(
            "Unsafe baseline initialisation. "
            f"Missing non-DQ keys: {disallowed_missing[:20]}; "
            f"unexpected keys: {unexpected[:20]}"
        )
    logger.info(
        "Loaded baseline checkpoint %s; %d DQ-CGP tensors are newly initialised.",
        checkpoint_path,
        len(missing),
    )


def set_seed(seed, use_cuda=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)
        cudnn.benchmark = False
        cudnn.deterministic = True


def train_epoch(model, criterion, train_loader, optimizer, opt, epoch_i, tb_writer):
    logger.info(f"[Epoch {epoch_i+1}]")
    model.train()
    criterion.train()

    time_meters = defaultdict(AverageMeter)
    loss_meters = defaultdict(AverageMeter)

    num_training_examples = len(train_loader)
    timer_dataloading = time.time()

    for batch_idx, batch in tqdm(
        enumerate(train_loader), desc="Training Iteration", total=num_training_examples
    ):
        time_meters["dataloading_time"].update(time.time() - timer_dataloading)

        timer_start = time.time()
        if opt.a_feat_dir is None:
            model_inputs, targets = prepare_batch_inputs(
                batch[1], opt.device, non_blocking=opt.pin_memory
            )
        else:
            model_inputs, targets = prepare_batch_inputs_audio(
                batch[1], opt.device, non_blocking=opt.pin_memory
            )

        # Mask labels (same as original train.py)
        B, L, _ = model_inputs["src_vid"].shape
        mask_labels = []
        for b_idx in range(B):
            relevant_windows = batch[0][b_idx]["relevant_windows"]
            num_windows = len(relevant_windows)
            masks = (
                torch.zeros(num_windows, L).to(opt.device)
                if num_windows < opt.max_windows
                else torch.zeros(opt.max_windows, L).to(opt.device)
            )
            for w_idx, w in enumerate(relevant_windows):
                if w_idx >= opt.max_windows:
                    break
                start = int(w[0]) // opt.clip_length
                end = int(w[1]) // opt.clip_length
                masks[w_idx, start:end] = 1
            mask_labels.append(masks)
        targets["mask_labels"] = mask_labels

        time_meters["prepare_inputs_time"].update(time.time() - timer_start)
        timer_start = time.time()

        outputs = model(**model_inputs)
        loss_dict = criterion(outputs, targets)

        weight_dict = criterion.weight_dict
        losses = sum(
            loss_dict[k] * weight_dict[k]
            for k in loss_dict.keys()
            if k in weight_dict
        )

        # -- VTC / CTC (identical to original) --
        src_txt_mask = model_inputs["src_txt_mask"]
        src_vid_mask = model_inputs["src_vid_mask"]
        pos_mask = targets["src_pos_mask"]

        src_txt_ed = outputs["src_txt_ed"]
        src_vid_ed = outputs["src_vid_ed"]
        loss_align = CTC_Loss()
        loss_vid_txt_align = loss_align(
            src_vid_ed, src_txt_ed, pos_mask, src_vid_mask, src_txt_mask
        )

        src_vid_cls_ed = outputs["src_vid_cls_ed"]
        src_txt_cls_ed = outputs["src_txt_cls_ed"]
        loss_align_VTC = VTCLoss()
        loss_vid_txt_align_VTC = loss_align_VTC(src_txt_cls_ed, src_vid_cls_ed)

        losses = (
            losses
            + opt.VTC_loss_coef * loss_vid_txt_align_VTC
            + opt.CTC_loss_coef * loss_vid_txt_align
        )
        time_meters["model_forward_time"].update(time.time() - timer_start)

        timer_start = time.time()
        optimizer.zero_grad()
        losses.backward()
        if opt.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
        optimizer.step()
        time_meters["model_backward_time"].update(time.time() - timer_start)

        loss_dict["loss_overall"] = float(losses)
        for k, v in loss_dict.items():
            loss_meters[k].update(
                float(v) * weight_dict[k] if k in weight_dict else float(v)
            )

        timer_dataloading = time.time()
        if opt.debug and batch_idx == 3:
            break

    # -- TensorBoard logging --
    tb_writer.add_scalar("Train/lr", float(optimizer.param_groups[0]["lr"]), epoch_i + 1)
    for k, v in loss_meters.items():
        tb_writer.add_scalar(f"Train/{k}", v.avg, epoch_i + 1)

    # -- DQ-CGP-specific diagnostics --
    if "loss_query_cgp_bind" in loss_meters:
        tb_writer.add_scalar(
            "Train/DQCGP_binding_loss", loss_meters["loss_query_cgp_bind"].avg, epoch_i + 1
        )
    if "loss_query_cgp_route" in loss_meters:
        tb_writer.add_scalar(
            "Train/DQCGP_route_loss", loss_meters["loss_query_cgp_route"].avg, epoch_i + 1
        )

    logger.info(
        f"Epoch[{epoch_i+1}] "
        + "  ".join(f"{k} {v.avg:.4f}" for k, v in sorted(loss_meters.items()))
    )
    log_line = opt.train_log_txt_formatter.format(
        time_str=time.strftime("%Y_%m_%d_%H_%M_%S"),
        epoch=epoch_i + 1,
        loss_str=" ".join(f"{k} {v.avg:.4f}" for k, v in loss_meters.items()),
    )
    with open(opt.train_log_filepath, "a") as log_file:
        log_file.write(log_line)
    return loss_meters


def train(
    model,
    criterion,
    optimizer,
    lr_scheduler,
    train_dataset,
    eval_dataset,
    opt,
    resume_metadata=None,
):
    """Train with the official Sim-DETR validation/checkpoint protocol."""
    model.to(opt.device)
    criterion.to(opt.device)

    tb_writer = SummaryWriter(opt.tensorboard_log_dir)
    tb_writer.add_text("hyperparameters", dict_to_markdown(vars(opt), max_str_len=None))
    opt.train_log_txt_formatter = (
        "{time_str} [Epoch] {epoch:03d} [Loss] {loss_str}\n"
    )
    opt.eval_log_txt_formatter = (
        "{time_str} [Epoch] {epoch:03d} [Loss] {loss_str} "
        "[Metrics] {eval_metrics_str}\n"
    )

    metadata = resume_metadata or {}
    best_metric = float(metadata.get("best_metric", 0.0))
    es_cnt = int(metadata.get("es_cnt", 0))
    if opt.start_epoch is None:
        start_epoch = -1 if opt.eval_untrained else 0
    else:
        start_epoch = int(opt.start_epoch)

    best_ckpt_path = opt.ckpt_filepath.replace(".ckpt", "_best.ckpt")
    latest_ckpt_path = opt.ckpt_filepath.replace(".ckpt", "_latest.ckpt")
    save_submission_filename = (
        f"latest_{opt.dset_name}_{opt.eval_split_name}_preds.jsonl"
    )

    train_loader = DataLoader(
        train_dataset,
        collate_fn=start_end_collate if opt.a_feat_dir is None else start_end_collate_audio,
        batch_size=opt.bsz,
        num_workers=opt.num_workers,
        shuffle=True,
        pin_memory=opt.pin_memory,
    )

    for epoch_i in trange(start_epoch, opt.n_epoch, desc="Epoch"):
        if epoch_i > -1:
            train_epoch(model, criterion, train_loader, optimizer, opt, epoch_i, tb_writer)
            lr_scheduler.step()

        should_stop = False
        if eval_dataset is not None:
            with torch.no_grad():
                metrics_no_nms, metrics_nms, eval_loss_meters, latest_file_paths = eval_epoch(
                    model,
                    eval_dataset,
                    opt,
                    save_submission_filename=save_submission_filename,
                    epoch_i=epoch_i,
                    criterion=criterion,
                    tb_writer=tb_writer,
                )

            logger.info("metrics_no_nms %s", pprint.pformat(metrics_no_nms["brief"], indent=4))
            if metrics_nms is not None:
                logger.info("metrics_nms %s", pprint.pformat(metrics_nms["brief"], indent=4))

            for key, value in metrics_no_nms["brief"].items():
                tb_writer.add_scalar(f"Eval/{key}", float(value), epoch_i + 1)

            eval_log_line = opt.eval_log_txt_formatter.format(
                time_str=time.strftime("%Y_%m_%d_%H_%M_%S"),
                epoch=epoch_i,
                loss_str=" ".join(
                    f"{key} {meter.avg:.4f}"
                    for key, meter in eval_loss_meters.items()
                ),
                eval_metrics_str=json.dumps(metrics_no_nms),
            )
            with open(opt.eval_log_filepath, "a") as log_file:
                log_file.write(eval_log_line)

            metric_key = "mAP" if opt.dset_name == "tvsum" else "MR-full-mAP"
            stop_score = float(metrics_no_nms["brief"][metric_key])
            if stop_score > best_metric:
                best_metric = stop_score
                es_cnt = 0
                torch.save(
                    _checkpoint_payload(
                        model, optimizer, lr_scheduler, opt, epoch_i, best_metric, es_cnt
                    ),
                    best_ckpt_path,
                )
                for source_path in latest_file_paths:
                    best_path = source_path.replace("latest", "best")
                    if source_path != best_path:
                        shutil.copy2(source_path, best_path)
                logger.info(
                    "[Best] epoch %d, %s %.4f", epoch_i + 1, metric_key, best_metric
                )
            else:
                es_cnt += 1
                should_stop = opt.max_es_cnt != -1 and es_cnt > opt.max_es_cnt

            # Every validation epoch is resumable, matching the official
            # Sim-DETR latest-checkpoint behaviour.
            torch.save(
                _checkpoint_payload(
                    model, optimizer, lr_scheduler, opt, epoch_i, best_metric, es_cnt
                ),
                latest_ckpt_path,
            )

        # Retain the official periodic archival checkpoints as well.
        save_interval = 10 if "subs_train" in opt.train_path else 50
        if (
            epoch_i > -1
            and ((epoch_i + 1) % save_interval == 0 or (epoch_i + 1) % opt.lr_drop == 0)
        ):
            periodic_path = opt.ckpt_filepath.replace(".ckpt", f"_e{epoch_i:04d}.ckpt")
            torch.save(
                _checkpoint_payload(
                    model, optimizer, lr_scheduler, opt, epoch_i, best_metric, es_cnt
                ),
                periodic_path,
            )

        if should_stop:
            with open(opt.train_log_filepath, "a") as log_file:
                log_file.write(f"Early Stop at epoch {epoch_i}\n")
            logger.info("Early stop at epoch %d; best metric %.4f", epoch_i, best_metric)
            break
        if opt.debug:
            break

    tb_writer.close()
    return best_metric


def setup_model_dq_cgp(opt):
    """Build model/criterion/optimizer and apply one checkpoint mode."""
    if opt.resume is not None and opt.init_from is not None:
        raise ValueError("--resume and --init_from are mutually exclusive")
    if opt.resume_all and opt.resume is None:
        raise ValueError("--resume_all requires --resume")

    logger.info("Building Sim-DETR + DQ-CGP model …")
    # Keep checkpoint modes under this entry point so init_from is applied once
    # and remains mutually exclusive with strict DQ resume.
    init_from = opt.init_from
    opt.init_from = None
    try:
        model, criterion = build_model_dq_cgp(opt)
    finally:
        opt.init_from = init_from

    model.to(opt.device)
    criterion.to(opt.device)
    model_params, trainable_params = count_parameters(model, verbose=False)
    if getattr(opt, "use_query_cgp", False) and model.query_cgp is not None:
        cgp_params, _ = count_parameters(model.query_cgp, verbose=False)
        logger.info(
            "Model: %d parameters (%d trainable; %d in DQ-CGP)",
            model_params, trainable_params, cgp_params,
        )
    else:
        logger.info(
            "Model: %d parameters (%d trainable)", model_params, trainable_params
        )
    optimizer = torch.optim.AdamW(
        [{"params": [p for p in model.parameters() if p.requires_grad]}],
        lr=opt.lr, weight_decay=opt.wd,
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, opt.lr_drop)
    resume_metadata = {}

    if init_from is not None:
        logger.info("Initialising from baseline checkpoint %s", init_from)
        _load_baseline_weights(model, init_from)

    if opt.resume is not None:
        logger.info(f"Resuming from {opt.resume}")
        checkpoint = torch.load(opt.resume, map_location="cpu")
        validate_tied_checkpoint(checkpoint, opt, operation="--resume")
        model.load_state_dict(checkpoint["model"], strict=True)
        if opt.resume_all:
            missing_training_state = [
                key for key in ("optimizer", "lr_scheduler", "epoch")
                if key not in checkpoint
            ]
            if missing_training_state:
                raise RuntimeError(
                    "Checkpoint cannot be fully resumed; missing keys: "
                    f"{missing_training_state}. Use --resume without --resume_all "
                    "for weights-only loading."
                )
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            if opt.start_epoch is None:
                opt.start_epoch = int(checkpoint["epoch"]) + 1
            resume_metadata = {
                "best_metric": checkpoint.get(
                    "best_metric", checkpoint.get("metric", 0.0)
                ),
                "es_cnt": checkpoint.get("es_cnt", 0),
            }

    return model, criterion, optimizer, lr_scheduler, resume_metadata


def start_training():
    # -----------------------------------------------------------------------
    # Parse args: base Sim-DETR options + DQ-CGP extensions
    # -----------------------------------------------------------------------
    base = BaseOptions()
    base.initialize()
    add_dq_cgp_args(base.parser)
    _escape_argparse_percent(base.parser)
    opt = base.parse()
    if not getattr(opt, "use_query_cgp", False):
        raise ValueError(
            "The DQ-CGP training entry requires --use_query_cgp. "
            "Use sim_detr/train.py for the original Sim-DETR baseline."
        )

    if opt.device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        torch.cuda.set_device(opt.gpu_id)
        logger.info("Using CUDA device %d", opt.gpu_id)
    set_seed(opt.seed)
    logger.info(f"Using seed {opt.seed}")

    # -----------------------------------------------------------------------
    # Dataset
    # -----------------------------------------------------------------------
    if opt.a_feat_dir is None:
        train_dataset = StartEndDataset(
            dset_name=opt.dset_name,
            data_path=opt.train_path,
            v_feat_dirs=opt.v_feat_dirs,
            q_feat_dir=opt.t_feat_dir,
            q_feat_type="last_hidden_state",
            max_q_l=opt.max_q_l,
            max_v_l=opt.max_v_l,
            ctx_mode=opt.ctx_mode,
            data_ratio=opt.data_ratio,
            normalize_v=not opt.no_norm_vfeat,
            normalize_t=not opt.no_norm_tfeat,
            clip_len=opt.clip_length,
            max_windows=opt.max_windows,
            span_loss_type=opt.span_loss_type,
            txt_drop_ratio=opt.txt_drop_ratio,
            dset_domain=getattr(opt, "dset_domain", None),
        )
        eval_dataset = None
        if opt.eval_path is not None:
            eval_dataset = StartEndDataset(
                dset_name=opt.dset_name,
                data_path=opt.eval_path,
                v_feat_dirs=opt.v_feat_dirs,
                q_feat_dir=opt.t_feat_dir.replace("sub_features", "text_features"),
                q_feat_type="last_hidden_state",
                max_q_l=opt.max_q_l,
                max_v_l=opt.max_v_l,
                ctx_mode=opt.ctx_mode,
                data_ratio=opt.data_ratio,
                normalize_v=not opt.no_norm_vfeat,
                normalize_t=not opt.no_norm_tfeat,
                clip_len=opt.clip_length,
                max_windows=opt.max_windows,
                span_loss_type=opt.span_loss_type,
                txt_drop_ratio=0,
                dset_domain=getattr(opt, "dset_domain", None),
            )
    else:
        train_dataset = StartEndDataset_audio(
            dset_name=opt.dset_name, data_path=opt.train_path,
            v_feat_dirs=opt.v_feat_dirs, q_feat_dir=opt.t_feat_dir,
            a_feat_dir=opt.a_feat_dir,
            q_feat_type="last_hidden_state",
            max_q_l=opt.max_q_l, max_v_l=opt.max_v_l,
            ctx_mode=opt.ctx_mode, data_ratio=opt.data_ratio,
            normalize_v=not opt.no_norm_vfeat, normalize_t=not opt.no_norm_tfeat,
            clip_len=opt.clip_length, max_windows=opt.max_windows,
            span_loss_type=opt.span_loss_type, txt_drop_ratio=opt.txt_drop_ratio,
        )
        eval_dataset = None
        if opt.eval_path is not None:
            eval_dataset = StartEndDataset_audio(
                dset_name=opt.dset_name, data_path=opt.eval_path,
                v_feat_dirs=opt.v_feat_dirs,
                q_feat_dir=opt.t_feat_dir.replace("sub_features", "text_features"),
                a_feat_dir=opt.a_feat_dir,
                q_feat_type="last_hidden_state",
                max_q_l=opt.max_q_l, max_v_l=opt.max_v_l,
                ctx_mode=opt.ctx_mode, data_ratio=opt.data_ratio,
                normalize_v=not opt.no_norm_vfeat, normalize_t=not opt.no_norm_tfeat,
                clip_len=opt.clip_length, max_windows=opt.max_windows,
                span_loss_type=opt.span_loss_type, txt_drop_ratio=0,
                dset_domain=getattr(opt, "dset_domain", None),
            )

    model, criterion, optimizer, lr_scheduler, resume_metadata = setup_model_dq_cgp(opt)
    return train(
        model, criterion, optimizer, lr_scheduler,
        train_dataset, eval_dataset, opt, resume_metadata=resume_metadata,
    )


if __name__ == "__main__":
    start_training()
