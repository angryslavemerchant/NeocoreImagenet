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
from model_asfnet import ASFNet
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

def build_model(args) -> ASFNet:
    return ASFNet(
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


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(state: dict, path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str, model: nn.Module, optimizer, scaler, device):
    # args stored as plain dict (not a dataclass), so weights_only=True is safe
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    if "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
    start_epoch = ckpt["epoch"] + 1
    best_top1   = ckpt.get("best_top1", 0.0)
    print(f"Resumed from epoch {start_epoch}  (best top-1 so far: {best_top1:.2f}%)")
    return start_epoch, best_top1


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
) -> tuple[float, float, float, int]:
    model.train()

    losses       = AverageMeter()
    task_losses  = AverageMeter()
    ratio_losses = AverageMeter()
    top1         = AverageMeter()
    top5         = AverageMeter()
    group_counts = AverageMeter()

    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.num_epochs} [train]", leave=False)

    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        B = images.size(0)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, l_ratio, mean_groups = model(images)
            task_loss = F.cross_entropy(logits, labels)
            loss      = task_loss + args.ratio_loss_weight * l_ratio

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        acc1, acc5 = accuracy(logits.float(), labels, topk=(1, 5))
        losses.update(loss.item(), B)
        task_losses.update(task_loss.item(), B)
        ratio_losses.update(l_ratio.item(), B)
        top1.update(acc1, B)
        top5.update(acc5, B)
        group_counts.update(mean_groups, B)

        pbar.set_postfix(
            loss   = f"{losses.avg:.3f}",
            top1   = f"{top1.avg:.1f}%",
            groups = f"{group_counts.avg:.1f}",
        )

        if global_step % args.log_interval == 0:
            wandb.log({
                "train/loss":        losses.avg,
                "train/task_loss":   task_losses.avg,
                "train/ratio_loss":  ratio_losses.avg,
                "train/top1":        top1.avg,
                "train/top5":        top5.avg,
                "train/mean_groups": group_counts.avg,
                "train/grad_scale":  scaler.get_scale(),
            }, step=global_step)

        global_step += 1

    return losses.avg, top1.avg, top5.avg, global_step


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

    losses       = AverageMeter()
    top1         = AverageMeter()
    top5         = AverageMeter()
    group_counts = AverageMeter()

    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.num_epochs} [val]  ", leave=False)

    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        B = images.size(0)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, l_ratio, mean_groups = model(images)
            task_loss = F.cross_entropy(logits, labels)
            loss      = task_loss + args.ratio_loss_weight * l_ratio

        acc1, acc5 = accuracy(logits.float(), labels, topk=(1, 5))
        losses.update(loss.item(), B)
        top1.update(acc1, B)
        top5.update(acc5, B)
        group_counts.update(mean_groups, B)
        pbar.set_postfix(top1=f"{top1.avg:.1f}%", groups=f"{group_counts.avg:.1f}")

    wandb.log({
        "val/loss":        losses.avg,
        "val/top1":        top1.avg,
        "val/top5":        top5.avg,
        "val/mean_groups": group_counts.avg,
    }, step=global_step)

    return losses.avg, top1.avg, top5.avg


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train ASFNet")

    # --- Model architecture ---
    parser.add_argument("--image_size",        type=int,   default=224,
                        help="Input image size (assumed square)")
    parser.add_argument("--patch_size",        type=int,   default=16,
                        help="Patch token size in pixels")
    parser.add_argument("--d_model",           type=int,   default=256,
                        help="Transformer model dimension")
    parser.add_argument("--num_heads",         type=int,   default=8,
                        help="Attention heads (d_model must be divisible by this)")
    parser.add_argument("--encoder_blocks",    type=int,   default=2,
                        help="Transformer blocks before the router")
    parser.add_argument("--main_blocks",       type=int,   default=6,
                        help="Transformer blocks after group merging")
    parser.add_argument("--mlp_ratio",         type=float, default=3.0,
                        help="FFN hidden dim = d_model * mlp_ratio")
    parser.add_argument("--num_classes",       type=int,   default=100,
                        help="Number of output classes")
    parser.add_argument("--target_group_size", type=float, default=3.0,
                        help="Target average patches per group (N in ratio loss)")
    parser.add_argument("--router_proj_dim",   type=int,   default=64,
                        help="Projection dim for router W_q and W_k")

    # --- Training ---
    parser.add_argument("--batch_size",        type=int,   default=1024)
    parser.add_argument("--num_epochs",        type=int,   default=90)
    parser.add_argument("--lr",                type=float, default=3e-3)
    parser.add_argument("--weight_decay",      type=float, default=0.05)
    parser.add_argument("--grad_clip",         type=float, default=1.0)
    parser.add_argument("--warmup_epochs",     type=int,   default=10)
    parser.add_argument("--ratio_loss_weight", type=float, default=0.03,
                        help="Weight on L_ratio (0.03 from H-Net)")

    # --- Data (attribute names must match what dataset.py expects on a config object) ---
    parser.add_argument("--dataset_name",      type=str, default="clane9/imagenet-100")
    parser.add_argument("--dataset_cache_dir", type=str, default="./data")
    parser.add_argument("--jpeg_cache_dir",    type=str, default="./jpeg_cache")
    parser.add_argument("--num_workers",       type=int, default=8)

    # --- Infrastructure ---
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints_asfnet")
    parser.add_argument("--resume",         type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--wandb_project",  type=str, default="asfnet")
    parser.add_argument("--wandb_entity",   type=str, default=None)
    parser.add_argument("--run_name",       type=str, default=None,
                        help="wandb run name; auto-generated from key args if omitted")
    parser.add_argument("--log_interval",   type=int, default=50,
                        help="Log to wandb every N optimizer steps")
    parser.add_argument("--seed",           type=int, default=42)
    parser.add_argument("--device",         type=str, default="cuda")

    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Auto-generate a descriptive run name from key hyperparams
    if args.run_name is None:
        args.run_name = (
            f"D{args.d_model}_enc{args.encoder_blocks}_main{args.main_blocks}"
            f"_N{args.target_group_size}_mlp{args.mlp_ratio}"
        )

    wandb.init(
        project = args.wandb_project,
        entity  = args.wandb_entity,
        name    = args.run_name,
        config  = vars(args),
    )

    # get_dataloaders expects attribute access on a config-like object.
    # argparse Namespace provides the same interface as the old Config dataclass,
    # so no changes to dataset.py are needed.
    train_loader, val_loader = get_dataloaders(args)

    # Model
    model = build_model(args).to(device)
    param_counts = model.count_parameters()
    print("\nParameter counts:")
    for name, count in param_counts.items():
        print(f"  {name:<16} {count:>10,}")
    wandb.config.update({"param_counts": param_counts})

    # Optimizer: AdamW with cosine annealing and linear warmup
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

    # bfloat16 has float32-range exponents so loss scaling rarely triggers,
    # but keeping GradScaler handles edge cases gracefully
    scaler = GradScaler()

    # Resume
    start_epoch = 0
    best_top1   = 0.0
    global_step = 0

    if args.resume:
        start_epoch, best_top1 = load_checkpoint(
            args.resume, model, optimizer, scaler, device
        )

    print(f"\nTraining '{args.run_name}' on {device}")
    print(f"Epochs: {args.num_epochs}  |  Batch: {args.batch_size}  |  LR: {args.lr}\n")

    for epoch in range(start_epoch, args.num_epochs):
        current_lr = optimizer.param_groups[0]["lr"]
        wandb.log({"train/lr": current_lr}, step=global_step)

        train_loss, train_top1, train_top5, global_step = train_one_epoch(
            model, train_loader, optimizer, scaler, args, epoch, device, global_step
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

        # Store args as plain dict — safe to load with weights_only=True
        # (unlike the old Config dataclass which required weights_only=False)
        ckpt = {
            "epoch":     epoch,
            "model":     model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler":    scaler.state_dict(),
            "val_top1":  val_top1,
            "best_top1": best_top1,
            "args":      vars(args),
        }

        save_checkpoint(ckpt, os.path.join(args.checkpoint_dir, "latest.pt"))

        if val_top1 > best_top1:
            best_top1          = val_top1
            ckpt["best_top1"]  = best_top1
            save_checkpoint(ckpt, os.path.join(args.checkpoint_dir, "best.pt"))
            print(f"  *** New best: {best_top1:.2f}%")

    wandb.finish()
    print(f"\nDone. Best val top-1: {best_top1:.2f}%")


if __name__ == "__main__":
    main()
