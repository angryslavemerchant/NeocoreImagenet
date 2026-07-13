"""
train_asfnet_br.py — supervised training for the border-retention models.

  Single stage:  python train_asfnet_br.py
  Two stage:     python train_asfnet_br.py --two_stage --main_blocks 4

Mirrors train_asfnet.py (same optimiser, schedule, image-count logging,
checkpoint format). Differences:
  - builds ASFNetBR / ASFNetBR2 (flag --two_stage)
  - logs mean retained tokens (train/mean_kept) — the retention analogue
    of mean_groups; for --two_stage also logs stage-2 group count
  - no --weighted_merge flag for stage 1: the confidence residual IS the
    gradient path and is always on. Stage 2 pooling is border-weighted by
    default; --uniform_merge2 disables it (ablation only — this severs
    probs2 from the task loss).
"""

import os
import random
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.cuda.amp import GradScaler
from tqdm import tqdm

from dataset import get_dataloaders
from model_asfnet_br import ASFNetBR, ASFNetBR2
from utils import AverageMeter, accuracy


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(args) -> nn.Module:
    if args.two_stage:
        return ASFNetBR2(
            image_size          = args.image_size,
            patch_size          = args.patch_size,
            in_channels         = 3,
            d_model             = args.d_model,
            num_heads           = args.num_heads,
            encoder1_blocks     = args.encoder_blocks,
            encoder2_blocks     = args.encoder2_blocks,
            main_blocks         = args.main_blocks,
            mlp_ratio           = args.mlp_ratio,
            num_classes         = args.num_classes,
            target_group_size_1 = args.target_group_size,
            target_group_size_2 = args.target_group_size_2,
            router_proj_dim     = args.router_proj_dim,
            weighted_merge2     = not args.uniform_merge2,
        )
    return ASFNetBR(
        image_size        = args.image_size,
        patch_size        = args.patch_size,
        in_channels       = 3,
        d_model           = args.d_model,
        num_heads         = args.num_heads,
        encoder_blocks    = args.encoder_blocks,
        main_blocks       = args.main_blocks,
        mlp_ratio         = args.mlp_ratio,
        num_classes       = args.num_classes,
        target_group_size = args.target_group_size,
        router_proj_dim   = args.router_proj_dim,
    )


def forward_losses(model, images, labels, args):
    """
    One forward pass; returns (loss, task_loss, ratio_total, logits, diag)
    where diag = dict of diagnostics, uniform across BR / BR2.
    """
    if args.two_stage:
        logits, l1, l2, mean_kept, mean_g2 = model(images)
        w2 = args.ratio_loss_weight if args.ratio_loss_weight2 is None else args.ratio_loss_weight2
        ratio_total = args.ratio_loss_weight * l1 + w2 * l2
        diag = {"mean_kept": mean_kept, "mean_groups2": mean_g2,
                "ratio1": l1, "ratio2": l2}
    else:
        logits, l1, mean_kept, mean_groups = model(images)
        ratio_total = args.ratio_loss_weight * l1
        diag = {"mean_kept": mean_kept, "mean_groups": mean_groups, "ratio1": l1}

    task_loss = F.cross_entropy(logits, labels)
    return task_loss + ratio_total, task_loss, ratio_total, logits, diag


def save_checkpoint(state: dict, path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path, model, optimizer, scaler, device):
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    if "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
    start_epoch = ckpt["epoch"] + 1
    best_top1   = ckpt.get("best_top1", 0.0)
    images_seen = ckpt.get("images_seen", 0)
    print(f"Resumed from epoch {start_epoch}  (best top-1 so far: {best_top1:.2f}%)")
    return start_epoch, best_top1, images_seen


def train_one_epoch(model, loader, optimizer, scaler, args, epoch, device,
                    global_step, images_seen):
    model.train()

    losses      = AverageMeter()
    task_losses = AverageMeter()
    top1        = AverageMeter()
    top5        = AverageMeter()
    kept        = AverageMeter()
    groups2     = AverageMeter()

    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.num_epochs} [train]", leave=False)

    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        B = images.size(0)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss, task_loss, ratio_total, logits, diag = forward_losses(
                model, images, labels, args
            )

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        acc1, acc5 = accuracy(logits.float(), labels, topk=(1, 5))
        losses.update(loss.item(), B)
        task_losses.update(task_loss.item(), B)
        top1.update(acc1, B)
        top5.update(acc5, B)
        kept.update(diag["mean_kept"], B)
        if args.two_stage:
            groups2.update(diag["mean_groups2"], B)

        pbar.set_postfix(loss=f"{losses.avg:.3f}", top1=f"{top1.avg:.1f}%",
                         kept=f"{kept.avg:.1f}")

        # Log every log_interval IMAGES, not steps (see train_asfnet.py).
        prev_images  = images_seen
        images_seen += B
        if images_seen // args.log_interval != prev_images // args.log_interval:
            log = {
                "train/loss":        losses.avg,
                "train/task_loss":   task_losses.avg,
                "train/top1":        top1.avg,
                "train/top5":        top5.avg,
                "train/mean_kept":   kept.avg,
                "train/ratio1":      float(diag["ratio1"]),
                "train/images_seen": images_seen,
            }
            if args.two_stage:
                log["train/mean_groups2"] = groups2.avg
                log["train/ratio2"]       = float(diag["ratio2"])
            wandb.log(log, step=global_step)

        global_step += 1

    return losses.avg, top1.avg, top5.avg, global_step, images_seen


@torch.no_grad()
def validate(model, loader, args, epoch, device, global_step):
    model.eval()

    losses  = AverageMeter()
    top1    = AverageMeter()
    top5    = AverageMeter()
    kept    = AverageMeter()
    groups2 = AverageMeter()

    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.num_epochs} [val]  ", leave=False)

    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        B = images.size(0)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss, _, _, logits, diag = forward_losses(model, images, labels, args)

        acc1, acc5 = accuracy(logits.float(), labels, topk=(1, 5))
        losses.update(loss.item(), B)
        top1.update(acc1, B)
        top5.update(acc5, B)
        kept.update(diag["mean_kept"], B)
        if args.two_stage:
            groups2.update(diag["mean_groups2"], B)
        pbar.set_postfix(top1=f"{top1.avg:.1f}%", kept=f"{kept.avg:.1f}")

    log = {
        "val/loss":      losses.avg,
        "val/top1":      top1.avg,
        "val/top5":      top5.avg,
        "val/mean_kept": kept.avg,
    }
    if args.two_stage:
        log["val/mean_groups2"] = groups2.avg
    wandb.log(log, step=global_step)

    return losses.avg, top1.avg, top5.avg


def main():
    parser = argparse.ArgumentParser(description="Train border-retention ASFNet")

    # --- Model architecture ---
    parser.add_argument("--image_size",          type=int,   default=224)
    parser.add_argument("--patch_size",          type=int,   default=16)
    parser.add_argument("--d_model",             type=int,   default=256)
    parser.add_argument("--num_heads",           type=int,   default=8)
    parser.add_argument("--encoder_blocks",      type=int,   default=2)
    parser.add_argument("--encoder2_blocks",     type=int,   default=2,
                        help="Stage 2 encoder depth (two-stage only)")
    parser.add_argument("--main_blocks",         type=int,   default=6)
    parser.add_argument("--mlp_ratio",           type=float, default=3.0)
    parser.add_argument("--num_classes",         type=int,   default=100)
    parser.add_argument("--target_group_size",   type=float, default=3.0)
    parser.add_argument("--target_group_size_2", type=float, default=3.0,
                        help="Stage 2 compression target (two-stage only)")
    parser.add_argument("--router_proj_dim",     type=int,   default=64)

    parser.add_argument("--two_stage", action="store_true",
                        help="Train ASFNetBR2 instead of ASFNetBR")
    parser.add_argument("--uniform_merge2", action="store_true",
                        help="ABLATION ONLY: uniform stage-2 pool. Severs "
                             "probs2 from the task loss (the known gradient "
                             "disconnection). Default = border-weighted.")

    # --- Training ---
    parser.add_argument("--batch_size",         type=int,   default=1024)
    parser.add_argument("--num_epochs",         type=int,   default=90)
    parser.add_argument("--lr",                 type=float, default=3e-3)
    parser.add_argument("--weight_decay",       type=float, default=0.05)
    parser.add_argument("--grad_clip",          type=float, default=1.0)
    parser.add_argument("--warmup_epochs",      type=int,   default=10)
    parser.add_argument("--ratio_loss_weight",  type=float, default=0.03)
    parser.add_argument("--ratio_loss_weight2", type=float, default=None,
                        help="Stage 2 ratio weight; default = same as stage 1")

    # --- Data ---
    parser.add_argument("--dataset_name",      type=str, default="clane9/imagenet-100")
    parser.add_argument("--dataset_cache_dir", type=str, default="./data")
    parser.add_argument("--jpeg_cache_dir",    type=str, default="./jpeg_cache")
    parser.add_argument("--num_workers",       type=int, default=8)

    # --- Infrastructure ---
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints_asfnet_br")
    parser.add_argument("--resume",         type=str, default=None)
    parser.add_argument("--wandb_project",  type=str, default="asfnet")
    parser.add_argument("--wandb_entity",   type=str, default=None)
    parser.add_argument("--run_name",       type=str, default=None)
    parser.add_argument("--log_interval",   type=int, default=10000)
    parser.add_argument("--seed",           type=int, default=42)
    parser.add_argument("--device",         type=str, default="cuda")

    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if args.run_name is None:
        stage = "BR2" if args.two_stage else "BR"
        args.run_name = (
            f"{stage}_D{args.d_model}_enc{args.encoder_blocks}"
            f"_main{args.main_blocks}_N{args.target_group_size}"
        )
        if args.two_stage:
            args.run_name += f"-{args.target_group_size_2}_enc2-{args.encoder2_blocks}"
        if args.uniform_merge2:
            args.run_name += "_umerge2"

    wandb.init(project=args.wandb_project, entity=args.wandb_entity,
               name=args.run_name, config=vars(args))

    train_loader, val_loader = get_dataloaders(args)

    model = build_model(args).to(device)
    model = torch.compile(model)

    param_counts = model.count_parameters()
    print("\nParameter counts:")
    for name, count in param_counts.items():
        print(f"  {name:<16} {count:>10,}")
    wandb.config.update({"param_counts": param_counts})

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-6, end_factor=1.0, total_iters=args.warmup_epochs)
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_epochs - args.warmup_epochs, eta_min=1e-6)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_sched, cosine_sched],
        milestones=[args.warmup_epochs])

    scaler      = GradScaler()
    start_epoch = 0
    best_top1   = 0.0
    global_step = 0
    images_seen = 0

    if args.resume:
        start_epoch, best_top1, images_seen = load_checkpoint(
            args.resume, model, optimizer, scaler, device)

    print(f"\nTraining '{args.run_name}' on {device}")
    print(f"Epochs: {args.num_epochs}  |  Batch: {args.batch_size}  |  LR: {args.lr}\n")

    for epoch in range(start_epoch, args.num_epochs):
        current_lr = optimizer.param_groups[0]["lr"]
        wandb.log({"train/lr": current_lr}, step=global_step)

        train_loss, train_top1, train_top5, global_step, images_seen = train_one_epoch(
            model, train_loader, optimizer, scaler, args, epoch, device,
            global_step, images_seen)
        val_loss, val_top1, val_top5 = validate(
            model, val_loader, args, epoch, device, global_step)
        scheduler.step()

        print(f"[{epoch+1:03d}/{args.num_epochs}] "
              f"lr {current_lr:.2e} | "
              f"train loss {train_loss:.3f} top1 {train_top1:.1f}% | "
              f"val loss {val_loss:.3f} top1 {val_top1:.1f}% top5 {val_top5:.1f}%")

        ckpt = {
            "epoch":       epoch,
            "model":       model.state_dict(),
            "optimizer":   optimizer.state_dict(),
            "scaler":      scaler.state_dict(),
            "val_top1":    val_top1,
            "best_top1":   best_top1,
            "images_seen": images_seen,
            "args":        vars(args),
        }
        save_checkpoint(ckpt, os.path.join(args.checkpoint_dir, "latest.pt"))

        if val_top1 > best_top1:
            best_top1         = val_top1
            ckpt["best_top1"] = best_top1
            save_checkpoint(ckpt, os.path.join(args.checkpoint_dir, "best.pt"))
            print(f"  *** New best: {best_top1:.2f}%")

    wandb.finish()
    print(f"\nDone. Best val top-1: {best_top1:.2f}%")


if __name__ == "__main__":
    main()
