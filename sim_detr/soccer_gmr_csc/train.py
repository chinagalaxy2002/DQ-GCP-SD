"""Train native/static/full Sim-DETR variants on Soccer-GMR Standard."""

from __future__ import annotations

import json
import logging
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from sim_detr.loss_fun.CTCLoss import CTC_Loss
from sim_detr.loss_fun.VTCLoss import VTCLoss
from utils.basic_utils import AverageMeter

from .config import parse_options, save_options
from .dataset import SoccerGMRDataset, prepare_soccer_gmr_batch, soccer_gmr_collate
from .evaluation import evaluate_model
from .model_builder import build_soccer_gmr_model, load_checkpoint_strict


logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO,
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_dataset(opt, path, data_ratio=1.0):
    return SoccerGMRDataset(
        path, opt.v_feat_dirs, opt.t_feat_dir,
        max_q_l=opt.max_q_l, max_v_l=opt.max_v_l,
        max_windows=opt.max_windows, clip_len=opt.clip_length,
        load_labels=True, data_ratio=data_ratio,
    )


def weighted_native_loss(losses, criterion):
    return sum(
        value * criterion.weight_dict[key]
        for key, value in losses.items() if key in criterion.weight_dict
    )


def alignment_losses(outputs, model_inputs, targets, opt):
    positive = targets["exist_label"].bool()
    ctc_loss = CTC_Loss()
    ctc_terms = []
    if positive.any():
        ctc_terms.append((
            ctc_loss(
                outputs["src_vid_ed"][positive], outputs["src_txt_ed"][positive],
                targets["src_pos_mask"][positive], model_inputs["src_vid_mask"][positive],
                model_inputs["src_txt_mask"][positive],
            ),
            positive.float().mean(),
        ))
    null = ~positive
    if null.any():
        ctc_terms.append((
            ctc_loss(
                outputs["src_vid_ed"][null], outputs["src_txt_ed"][null],
                targets["src_pos_mask"][null], model_inputs["src_vid_mask"][null],
                model_inputs["src_txt_mask"][null],
            ),
            null.float().mean() * opt.null_ctc_loss_weight,
        ))
    if ctc_terms:
        ctc = sum(value * weight for value, weight in ctc_terms)
    else:
        ctc = outputs["pred_logits"].sum() * 0.0
    if positive.any():
        text = outputs["src_txt_cls_ed"][positive]
        video = outputs["src_vid_cls_ed"][positive]
        # Keep the original VTC objective but cap the in-batch negative pool at
        # its released Soccer-GMR batch size.  bsz=64 otherwise changes the
        # contrastive task itself, rather than merely accelerating training.
        chunks = [
            VTCLoss()(text[start:start + opt.vtc_group_size], video[start:start + opt.vtc_group_size])
            for start in range(0, len(text), opt.vtc_group_size)
        ]
        vtc = torch.stack(chunks).mean()
    else:
        vtc = outputs["pred_logits"].sum() * 0.0
    return opt.CTC_loss_coef * ctc + opt.VTC_loss_coef * vtc, ctc, vtc


def train_epoch(model, criterion, loader, optimizer, opt, epoch):
    model.train()
    criterion.train()
    meters = defaultdict(AverageMeter)
    for batch_index, (_, batched) in enumerate(tqdm(loader, desc=f"epoch {epoch + 1}")):
        model_inputs, targets = prepare_soccer_gmr_batch(
            batched, opt.device, non_blocking=opt.pin_memory
        )
        outputs = model(**model_inputs)
        loss_dict = criterion(outputs, targets)
        native_loss = weighted_native_loss(loss_dict, criterion)
        align_loss, ctc, vtc = alignment_losses(outputs, model_inputs, targets, opt)
        loss = native_loss + align_loss
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at epoch={epoch + 1}, batch={batch_index}")
        optimizer.zero_grad()
        loss.backward()
        if opt.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
        optimizer.step()
        meters["loss_overall"].update(float(loss.detach()))
        meters["loss_ctc"].update(float(ctc.detach()))
        meters["loss_vtc"].update(float(vtc.detach()))
        for key, value in loss_dict.items():
            if torch.is_tensor(value) and value.numel() == 1:
                meters[key].update(float(value.detach()))
        if opt.debug and batch_index >= 3:
            break
    return {key: meter.avg for key, meter in meters.items()}


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_score, opt):
    torch.save({
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "lr_scheduler": scheduler.state_dict(), "epoch": epoch,
        "best_score": best_score, "opt": opt,
    }, path)


def main(argv=None):
    opt = parse_options(argv)
    if opt.device.type == "cuda":
        torch.cuda.set_device(opt.device)
    set_seed(opt.seed)
    output_dir = opt.results_root / opt.exp_id
    if output_dir.exists() and any(output_dir.iterdir()) and opt.resume is None:
        raise FileExistsError(f"Refusing to overwrite non-empty run directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    opt.output_dir = output_dir
    save_options(opt, output_dir)

    ratio = 0.01 if opt.debug else 1.0
    train_dataset = build_dataset(opt, opt.train_path, data_ratio=ratio)
    val_dataset = build_dataset(opt, opt.eval_path, data_ratio=ratio)
    loader = DataLoader(
        train_dataset, batch_size=opt.bsz, shuffle=True,
        num_workers=opt.num_workers, pin_memory=opt.pin_memory,
        collate_fn=soccer_gmr_collate,
    )
    model, criterion = build_soccer_gmr_model(opt)
    model.to(opt.device)
    criterion.to(opt.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=opt.wd)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, opt.lr_drop)
    start_epoch = 0
    best_score = float("-inf")
    if opt.resume:
        checkpoint = load_checkpoint_strict(model, opt.resume)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["lr_scheduler"])
        start_epoch = checkpoint["epoch"] + 1
        best_score = checkpoint.get("best_score", best_score)

    history_path = output_dir / "history.jsonl"
    patience = 0
    epochs = min(opt.n_epoch, 1) if opt.debug else opt.n_epoch
    try:
        for epoch in range(start_epoch, epochs):
            train_metrics = train_epoch(model, criterion, loader, optimizer, opt, epoch)
            scheduler.step()
            if (epoch + 1) % opt.eval_epoch_interval:
                continue
            metrics, _, _ = evaluate_model(model, val_dataset, opt, output_dir, "latest_val")
            score = float(metrics["brief"]["mAP"])
            record = {
                "time": time.strftime("%Y-%m-%d %H:%M:%S"), "epoch": epoch + 1,
                "train": train_metrics, "val": metrics["brief"],
                "selection_metric": "mAP", "selection_score": score,
            }
            with history_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            logger.info("epoch=%d train=%s val=%s", epoch + 1, train_metrics, metrics["brief"])
            save_checkpoint(
                output_dir / "model_latest.ckpt", model, optimizer, scheduler,
                epoch, max(best_score, score), opt,
            )
            if score > best_score:
                best_score = score
                patience = 0
                save_checkpoint(
                    output_dir / "model_best.ckpt", model, optimizer, scheduler,
                    epoch, best_score, opt,
                )
                (output_dir / "best_val_metrics.json").write_text(
                    json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
                )
            else:
                patience += 1
                if patience >= opt.max_es_cnt:
                    logger.info("Early stopping: patience=%d, best mAP=%.2f", patience, best_score)
                    break
    finally:
        model.close()


if __name__ == "__main__":
    main()
