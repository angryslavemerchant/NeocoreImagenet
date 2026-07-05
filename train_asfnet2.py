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
from model_asfnet2 import ASFNet2
from utils import AverageMeter, accuracy


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(args) -> ASFNet2:
    return ASFNet2(
        image_size           = args.image_size,
        patch_size           = args.patch_size,
        in_channels          = 3,
        d_model              = args.d_model,
        num_heads            = args.num_heads,
        encoder1_blocks      = args.encoder1_blocks,
        encoder2_blocks      = args.encoder2_blocks,
        main_blocks          = args.main_blocks,
        mlp_ratio            = args.mlp_ratio,
        num_classes          = args.num_classes,
        target_group_size_1  = args.target_group_size_1,
        target_group_size_2  = args.target_group_size_2,
        router_proj_dim      = args.router_proj_dim,
        knn_k                = args.knn_k,
        local_encoder1       = args.local_encoder1,
        local_radius         = args.local_radius,
        local_encoder2       = args.local_encoder2,
        local_encoder2_safe  = args.local_encoder2_safe,
    )


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
    print(f"Resumed from epoch {start_epoch}  (best top-1: {best_top1:.2f}%)")
    return start_epoch, best_top1, images_seen


def train_one_epoch(
    model, loader, optimizer, scaler, args,
    epoch, device, global_step, images_seen,
):
    model.train()

    losses        = AverageMeter()
    task_losses   = AverageMeter()
    ratio1_losses = AverageMeter()
    ratio2_losses = AverageMeter()
    top1          = AverageMeter()
    top5          = AverageMeter()
    groups1       = AverageMeter()
    groups2       = AverageMeter()
    grad_norms    = AverageMeter()
    nan_batches   = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.num_epochs} [train]", leave=False)

    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        B = images.size(0)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, l_ratio1, l_ratio2, mg1, mg2 = model(images)
            task_loss = F.cross_entropy(logits, labels)
            loss      = (task_loss
                         + args.ratio_loss_weight   * l_ratio1
                         + args.ratio_loss_weight_2 * l_ratio2)

        # Finite-loss guard: if the loss goes NaN/inf, skip the step rather than
        # backpropagating garbage, and count it so the collapse is visible in
        # wandb (a rising nan_batches localises numeric blow-ups in time).
        if not torch.isfinite(loss).item():
            nan_batches += 1
            optimizer.zero_grad(set_to_none=True)
            images_seen += B
            global_step += 1
            continue

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        # clip_grad_norm_ returns the total gradient norm BEFORE clipping — this
        # is the diagnostic signal. A spike here at the loss reversal = training
        # instability; inf/NaN here = numeric blow-up upstream.
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        grad_norms.update(float(grad_norm), B)
        scaler.step(optimizer)
        scaler.update()

        acc1, acc5 = accuracy(logits.float(), labels, topk=(1, 5))
        losses.update(loss.item(), B)
        task_losses.update(task_loss.item(), B)
        ratio1_losses.update(l_ratio1.item(), B)
        ratio2_losses.update(l_ratio2.item(), B)
        top1.update(acc1, B)
        top5.update(acc5, B)
        groups1.update(mg1, B)
        groups2.update(mg2, B)

        pbar.set_postfix(
            loss   = f"{losses.avg:.3f}",
            top1   = f"{top1.avg:.1f}%",
            g1     = f"{groups1.avg:.0f}",
            g2     = f"{groups2.avg:.0f}",
        )

        prev_images  = images_seen
        images_seen += B
        if images_seen // args.log_interval != prev_images // args.log_interval:
            wandb.log({
                "train/loss":          losses.avg,
                "train/task_loss":     task_losses.avg,
                "train/ratio_loss_1":  ratio1_losses.avg,
                "train/ratio_loss_2":  ratio2_losses.avg,
                "train/top1":          top1.avg,
                "train/top5":          top5.avg,
                "train/mean_groups_1": groups1.avg,
                "train/mean_groups_2": groups2.avg,
                "train/grad_scale":    scaler.get_scale(),
                "train/grad_norm":     float(grad_norm),   # instantaneous (latest batch)
                "train/grad_norm_avg": grad_norms.avg,     # running average this epoch
                "train/nan_batches":   nan_batches,        # cumulative this epoch
                "train/images_seen":   images_seen,
            }, step=global_step)

        global_step += 1

    return losses.avg, top1.avg, top5.avg, global_step, images_seen


@torch.no_grad()
def validate(model, loader, args, epoch, device, global_step):
    model.eval()

    losses  = AverageMeter()
    top1    = AverageMeter()
    top5    = AverageMeter()
    groups1 = AverageMeter()
    groups2 = AverageMeter()

    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.num_epochs} [val]  ", leave=False)

    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        B = images.size(0)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, l_ratio1, l_ratio2, mg1, mg2 = model(images)
            task_loss = F.cross_entropy(logits, labels)
            loss      = (task_loss
                         + args.ratio_loss_weight   * l_ratio1
                         + args.ratio_loss_weight_2 * l_ratio2)

        acc1, acc5 = accuracy(logits.float(), labels, topk=(1, 5))
        losses.update(loss.item(), B)
        top1.update(acc1, B)
        top5.update(acc5, B)
        groups1.update(mg1, B)
        groups2.update(mg2, B)
        pbar.set_postfix(top1=f"{top1.avg:.1f}%", g1=f"{groups1.avg:.0f}", g2=f"{groups2.avg:.0f}")

    wandb.log({
        "val/loss":          losses.avg,
        "val/top1":          top1.avg,
        "val/top5":          top5.avg,
        "val/mean_groups_1": groups1.avg,
        "val/mean_groups_2": groups2.avg,
    }, step=global_step)

    return losses.avg, top1.avg, top5.avg


def main():
    parser = argparse.ArgumentParser(description="Train two-stage hierarchical ASFNet")

    # --- Model ---
    parser.add_argument("--image_size",        type=int,   default=224)
    parser.add_argument("--patch_size",        type=int,   default=16)
    parser.add_argument("--d_model",           type=int,   default=256)
    parser.add_argument("--num_heads",         type=int,   default=8)
    parser.add_argument("--encoder1_blocks",   type=int,   default=2,
                        help="Transformer blocks before Stage 1 routing.")
    parser.add_argument("--encoder2_blocks",   type=int,   default=2,
                        help="Transformer blocks between Stage 1 merge and Stage 2 routing.")
    parser.add_argument("--main_blocks",       type=int,   default=4,
                        help="Transformer blocks after Stage 2 merge. "
                             "Default layout (2+2+4=8 blocks) gives ~5.7M params.")
    parser.add_argument("--mlp_ratio",         type=float, default=3.0)
    parser.add_argument("--num_classes",       type=int,   default=100)
    parser.add_argument("--target_group_size_1", type=float, default=3.0,
                        help="Stage 1 compression: N tokens → N/target_group_size_1 groups. "
                             "3.0 = compress by 1/3. Higher = more aggressive compression.")
    parser.add_argument("--target_group_size_2", type=float, default=3.0,
                        help="Stage 2 compression: G1 groups → G1/target_group_size_2 groups. "
                             "Independent of Stage 1. Combined effect: 1/(s1 * s2) of original. "
                             "e.g. s1=3, s2=3 → 1/9.  s1=4, s2=2 → 1/8.")
    parser.add_argument("--router_proj_dim",      type=int,   default=64)
    parser.add_argument("--knn_k",             type=int,   default=6,
                        help="Stage 2 k-NN neighbours per token. k=6 gives richer "
                             "connectivity than the Stage 1 grid's effective k=4.")

    parser.add_argument("--local_encoder1", action="store_true",
                        help="Use local-neighbourhood attention in encoder1")
    parser.add_argument("--local_radius", type=int, default=1,
                        help="Chebyshev radius of the encoder1 attention window")
    parser.add_argument("--local_encoder2", action="store_true",
                        help="Use k-NN local attention in encoder2 (mask = the same "
                             "per-image k-NN graph the Stage 2 router cuts)")
    parser.add_argument("--local_encoder2_safe", action="store_true",
                        help="Diagnostic: run encoder2 local attention in float32 with "
                             "an additive mask + math backend, to rule out low-precision "
                             "/ fused-kernel instability. Slower; disable torch.compile.")

    # --- Training ---
    parser.add_argument("--batch_size",          type=int,   default=1024)
    parser.add_argument("--num_epochs",          type=int,   default=90)
    parser.add_argument("--lr",                  type=float, default=3e-3)
    parser.add_argument("--weight_decay",        type=float, default=0.05)
    parser.add_argument("--grad_clip",           type=float, default=1.0)
    parser.add_argument("--warmup_epochs",       type=int,   default=10)
    parser.add_argument("--ratio_loss_weight",   type=float, default=0.03,
                        help="Weight on Stage 1 ratio loss. If using patch_size=8, "
                             "scale down proportionally (e.g. 0.007) — see train_asfnet.py.")
    parser.add_argument("--ratio_loss_weight_2", type=float, default=None,
                        help="Weight on Stage 2 ratio loss. Defaults to ratio_loss_weight "
                             "if not set. Stage 2 has fewer tokens and edges so may need "
                             "separate tuning if Stage 1's weight causes oscillation.")

    # --- Data ---
    parser.add_argument("--dataset_name",      type=str, default="clane9/imagenet-100")
    parser.add_argument("--dataset_cache_dir", type=str, default="./data")
    parser.add_argument("--jpeg_cache_dir",    type=str, default="./jpeg_cache")
    parser.add_argument("--num_workers",       type=int, default=8)

    # --- Infrastructure ---
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints_asfnet2")
    parser.add_argument("--resume",         type=str, default=None)
    parser.add_argument("--wandb_project",  type=str, default="asfnet")
    parser.add_argument("--wandb_entity",   type=str, default=None)
    parser.add_argument("--run_name",       type=str, default=None)
    parser.add_argument("--log_interval",   type=int, default=10000)
    parser.add_argument("--seed",           type=int, default=42)
    parser.add_argument("--device",         type=str, default="cuda")

    args = parser.parse_args()

    # Default Stage 2 ratio loss weight to Stage 1's if not explicitly set
    if args.ratio_loss_weight_2 is None:
        args.ratio_loss_weight_2 = args.ratio_loss_weight

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if args.run_name is None:
        args.run_name = (
            f"ASFNet2_D{args.d_model}"
            f"_loc{args.local_encoder1}-{args.local_radius}"
            f"_l2{args.local_encoder2}"
            f"_enc{args.encoder1_blocks}-{args.encoder2_blocks}"
            f"_main{args.main_blocks}"
            f"_N{args.target_group_size_1}-{args.target_group_size_2}"
            f"_k{args.knn_k}_p{args.patch_size}"
        )

    wandb.init(
        project = args.wandb_project,
        entity  = args.wandb_entity,
        name    = args.run_name,
        config  = vars(args),
    )

    train_loader, val_loader = get_dataloaders(args)

    model = build_model(args).to(device)
    #model = torch.compile(model)

    param_counts = model.count_parameters()
    print("\nParameter counts:")
    for name, count in param_counts.items():
        print(f"  {name:<16} {count:>10,}")
    wandb.config.update({"param_counts": param_counts})

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = args.lr,
        weight_decay = args.weight_decay,
    )
    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-6, end_factor=1.0, total_iters=args.warmup_epochs,
    )
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_epochs - args.warmup_epochs, eta_min=1e-6,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[args.warmup_epochs],
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
    print(f"ratio_loss_weight: stage1={args.ratio_loss_weight}  stage2={args.ratio_loss_weight_2}")
    overall = args.target_group_size_1 * args.target_group_size_2
    print(f"knn_k={args.knn_k}  "
          f"target_group_size: stage1={args.target_group_size_1}  stage2={args.target_group_size_2}  "
          f"(~1/{overall:.0f} overall)\n")

    for epoch in range(start_epoch, args.num_epochs):
        current_lr = optimizer.param_groups[0]["lr"]
        wandb.log({"train/lr": current_lr}, step=global_step)

        train_loss, train_top1, train_top5, global_step, images_seen = train_one_epoch(
            model, train_loader, optimizer, scaler, args,
            epoch, device, global_step, images_seen,
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