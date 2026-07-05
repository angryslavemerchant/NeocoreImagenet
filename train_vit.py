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
from model_vit import ViT
from utils import AverageMeter, accuracy


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def build_model(args) -> ViT:
    return ViT(
        image_size  = args.image_size,
        patch_size  = args.patch_size,
        in_channels = 3,
        d_model     = args.d_model,
        num_heads   = args.num_heads,
        depth       = args.depth,
        mlp_ratio   = args.mlp_ratio,
        num_classes = args.num_classes,
    )


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(state: dict, path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str, model: nn.Module, optimizer, scaler, device):
    ckpt        = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    if "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
    start_epoch = ckpt["epoch"] + 1
    best_top1   = ckpt.get("best_top1", 0.0)
    images_seen = ckpt.get("images_seen", 0)
    print(f"Resumed from epoch {start_epoch}  (best top-1 so far: {best_top1:.2f}%)")
    return start_epoch, best_top1, images_seen


# ---------------------------------------------------------------------------
# Train / validate one epoch
# ---------------------------------------------------------------------------

def train_one_epoch(
    model:       nn.Module,
    loader,
    optimizer,
    scaler:      GradScaler,
    args,
    epoch:       int,
    device:      torch.device,
    global_step: int,
    images_seen: int,
) -> tuple[float, float, float, int, int]:
    """
    Returns: (loss, top1, top5, global_step, images_seen)

    Logging is keyed on images_seen (cumulative images processed), not
    steps — identical convention to train_asfnet.py so wandb curves are
    directly comparable regardless of batch size.
    """
    model.train()

    losses = AverageMeter()
    top1   = AverageMeter()
    top5   = AverageMeter()

    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.num_epochs} [train]", leave=False)

    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        B = images.size(0)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(images)
            loss   = F.cross_entropy(logits, labels, label_smoothing=args.label_smoothing)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        acc1, acc5 = accuracy(logits.float(), labels, topk=(1, 5))
        losses.update(loss.item(), B)
        top1.update(acc1, B)
        top5.update(acc5, B)

        pbar.set_postfix(
            loss = f"{losses.avg:.3f}",
            top1 = f"{top1.avg:.1f}%",
        )

        prev_images  = images_seen
        images_seen += B
        if images_seen // args.log_interval != prev_images // args.log_interval:
            wandb.log({
                "train/loss":        losses.avg,
                "train/top1":        top1.avg,
                "train/top5":        top5.avg,
                "train/grad_scale":  scaler.get_scale(),
                "train/images_seen": images_seen,
            }, step=global_step)

        global_step += 1

    return losses.avg, top1.avg, top5.avg, global_step, images_seen


@torch.no_grad()
def validate(
    model:       nn.Module,
    loader,
    args,
    epoch:       int,
    device:      torch.device,
    global_step: int,
) -> tuple[float, float, float]:
    model.eval()

    losses = AverageMeter()
    top1   = AverageMeter()
    top5   = AverageMeter()

    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.num_epochs} [val]  ", leave=False)

    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        B = images.size(0)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(images)
            # Val loss uses label_smoothing=0 so it's always pure cross-entropy
            # and directly comparable across runs regardless of training smoothing.
            loss = F.cross_entropy(logits, labels)

        acc1, acc5 = accuracy(logits.float(), labels, topk=(1, 5))
        losses.update(loss.item(), B)
        top1.update(acc1, B)
        top5.update(acc5, B)
        pbar.set_postfix(top1=f"{top1.avg:.1f}%")

    wandb.log({
        "val/loss": losses.avg,
        "val/top1": top1.avg,
        "val/top5": top5.avg,
    }, step=global_step)

    return losses.avg, top1.avg, top5.avg


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train ViT baseline (ablation)")

    # --- Model architecture ---
    parser.add_argument("--image_size",  type=int,   default=224)
    parser.add_argument("--patch_size",  type=int,   default=16)
    parser.add_argument("--d_model",     type=int,   default=256,
                        help="Token dimension. Matches ASFNet's d_model for fair comparison.")
    parser.add_argument("--num_heads",   type=int,   default=8)
    parser.add_argument("--depth",       type=int,   default=7,
                        help="Number of transformer blocks. 7 blocks @ d_model=256 "
                             "gives ~5.8M params, matching the ~6M budget.")
    parser.add_argument("--mlp_ratio",   type=float, default=4.0)
    parser.add_argument("--num_classes", type=int,   default=100)

    # --- Training ---
    parser.add_argument("--batch_size",       type=int,   default=1024)
    parser.add_argument("--num_epochs",       type=int,   default=90)
    parser.add_argument("--lr",               type=float, default=3e-3)
    parser.add_argument("--weight_decay",     type=float, default=0.05)
    parser.add_argument("--grad_clip",        type=float, default=1.0)
    parser.add_argument("--warmup_epochs",    type=int,   default=10)
    parser.add_argument("--label_smoothing",  type=float, default=0.0,
                        help="Cross-entropy label smoothing. Default 0.0 (no smoothing) "
                             "keeps training identical to the other models for fair comparison. "
                             "Set to 0.1 to try the standard ViT training trick.")

    # --- Data ---
    parser.add_argument("--dataset_name",      type=str, default="clane9/imagenet-100")
    parser.add_argument("--dataset_cache_dir", type=str, default="./data")
    parser.add_argument("--jpeg_cache_dir",    type=str, default="./jpeg_cache")
    parser.add_argument("--num_workers",       type=int, default=8)

    # --- Infrastructure ---
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints_vit")
    parser.add_argument("--resume",         type=str, default=None)
    parser.add_argument("--wandb_project",  type=str, default="asfnet",
                        help="Log to the same project as ASFNet so curves sit side by side.")
    parser.add_argument("--wandb_entity",   type=str, default=None)
    parser.add_argument("--run_name",       type=str, default=None)
    parser.add_argument("--log_interval",   type=int, default=10000,
                        help="Log to wandb every N images processed (not steps). "
                             "Same convention as train_asfnet.py.")
    parser.add_argument("--seed",           type=int, default=42)
    parser.add_argument("--device",         type=str, default="cuda")

    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if args.run_name is None:
        args.run_name = (
            f"ViT_D{args.d_model}_depth{args.depth}"
            f"_mlp{args.mlp_ratio}_p{args.patch_size}"
        )

    wandb.init(
        project = args.wandb_project,
        entity  = args.wandb_entity,
        name    = args.run_name,
        config  = vars(args),
    )

    train_loader, val_loader = get_dataloaders(args)

    model = build_model(args).to(device)

    torch.compile(model)

    param_counts = model.count_parameters()
    print("\nParameter counts:")
    for name, count in param_counts.items():
        print(f"  {name:<18} {count:>10,}")
    wandb.config.update({"param_counts": param_counts})

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = args.lr,
        weight_decay = args.weight_decay,
    )
    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor = 1e-6,
        end_factor   = 1.0,
        total_iters  = args.warmup_epochs,
    )
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max   = args.num_epochs - args.warmup_epochs,
        eta_min = 1e-6,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers = [warmup_sched, cosine_sched],
        milestones = [args.warmup_epochs],
    )

    scaler      = GradScaler()
    start_epoch = 0
    best_top1   = 0.0
    global_step = 0
    images_seen = 0

    if args.resume:
        start_epoch, best_top1, images_seen = load_checkpoint(
            args.resume, model, optimizer, scaler, device
        )

    print(f"\nTraining '{args.run_name}' on {device}")
    print(f"Epochs: {args.num_epochs}  |  Batch: {args.batch_size}  |  LR: {args.lr}")
    if args.label_smoothing > 0:
        print(f"Label smoothing: {args.label_smoothing}")
    print(f"Logging every {args.log_interval:,} images\n")

    for epoch in range(start_epoch, args.num_epochs):
        current_lr = optimizer.param_groups[0]["lr"]
        wandb.log({"train/lr": current_lr}, step=global_step)

        train_loss, train_top1, train_top5, global_step, images_seen = train_one_epoch(
            model, train_loader, optimizer, scaler, args, epoch, device,
            global_step, images_seen,
        )
        val_loss, val_top1, val_top5 = validate(
            model, val_loader, args, epoch, device, global_step
        )
        scheduler.step()

        print(
            f"[{epoch+1:03d}/{args.num_epochs}] "
            f"lr {current_lr:.2e} | "
            f"train loss {train_loss:.3f} top1 {train_top1:.1f}% | "
            f"val loss {val_loss:.3f} top1 {val_top1:.1f}% top5 {val_top5:.1f}%"
        )

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
