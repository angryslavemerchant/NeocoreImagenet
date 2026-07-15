"""
train_asfnet_ae.py — self-supervised pretraining for ASFNetAE.

No labels are used: the dataloader's labels are ignored. The objective is

    loss = reconstruction (dropped patches only) + w * ratio loss

The ratio loss remains as the guard rail against early-training percolation
collapse. NOTE FOR LATER: once reconstruction pressure is established, the
rate-distortion pair (retained-token count vs reconstruction error) can in
principle replace the fixed ratio target entirely — a --ratio decay
schedule is the natural first experiment. Kept fixed here per "keep
existing version".

Checkpoint selection: lowest validation reconstruction loss.
Defaults mirror train_asfnet.py; self-supervised methods typically want
more epochs than supervised (MAE pretrains 300-800 on IN-1k), so bump
--num_epochs when the pipeline is validated.

Linear probe afterwards: build a classifier ASFNetBR, load the backbone
weights from this checkpoint's "backbone." prefix, freeze, train only the
classifier head. (The keys line up by construction.)
"""

import os
import random
import argparse

import numpy as np
import torch
import torch.nn as nn
import wandb
from torch.cuda.amp import GradScaler
from tqdm import tqdm

from dataset import get_dataloaders
from model_asfnet_ae import ASFNetAE
from utils import AverageMeter


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(args) -> ASFNetAE:
    return ASFNetAE(
        image_size        = args.image_size,
        patch_size        = args.patch_size,
        in_channels       = 3,
        d_model           = args.d_model,
        num_heads         = args.num_heads,
        encoder_blocks    = args.encoder_blocks,
        main_blocks       = args.main_blocks,
        mlp_ratio         = args.mlp_ratio,
        target_group_size = args.target_group_size,
        router_proj_dim   = args.router_proj_dim,
        decoder_d_model   = args.decoder_d_model,
        decoder_blocks    = args.decoder_blocks,
        decoder_heads     = args.decoder_heads,
        norm_pix_loss     = not args.no_norm_pix,
    )


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
    best_val    = ckpt.get("best_val_rec", float("inf"))
    images_seen = ckpt.get("images_seen", 0)
    print(f"Resumed from epoch {start_epoch}  (best val rec so far: {best_val:.4f})")
    return start_epoch, best_val, images_seen


def run_epoch(model, loader, optimizer, scaler, args, epoch, device,
              global_step, images_seen, train: bool):
    model.train() if train else model.eval()

    rec_losses   = AverageMeter()
    ratio_losses = AverageMeter()
    kept         = AverageMeter()
    dropf        = AverageMeter()

    tag  = "train" if train else "val"
    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.num_epochs} [{tag}]", leave=False)

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for images, _labels in pbar:   # labels ignored — self-supervised
            images = images.to(device, non_blocking=True)
            B = images.size(0)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss_rec, l_ratio, mean_kept, _mean_groups, drop_frac = model(images)
                loss = loss_rec + args.ratio_loss_weight * l_ratio

            if train:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()

            rec_losses.update(loss_rec.item(), B)
            ratio_losses.update(l_ratio.item(), B)
            kept.update(mean_kept, B)
            dropf.update(drop_frac, B)

            pbar.set_postfix(rec=f"{rec_losses.avg:.4f}",
                             kept=f"{kept.avg:.1f}",
                             drop=f"{dropf.avg:.2f}")

            if train:
                prev_images  = images_seen
                images_seen += B
                if images_seen // args.log_interval != prev_images // args.log_interval:
                    wandb.log({
                        "train/rec_loss":    rec_losses.avg,
                        "train/ratio_loss":  ratio_losses.avg,
                        "train/mean_kept":   kept.avg,
                        "train/drop_frac":   dropf.avg,
                        "train/images_seen": images_seen,
                    }, step=global_step)
                global_step += 1

    if not train:
        wandb.log({
            "val/rec_loss":   rec_losses.avg,
            "val/ratio_loss": ratio_losses.avg,
            "val/mean_kept":  kept.avg,
            "val/drop_frac":  dropf.avg,
        }, step=global_step)

    return rec_losses.avg, global_step, images_seen


def main():
    parser = argparse.ArgumentParser(description="Self-supervised ASFNet autoencoder")

    # --- Backbone (matches train_asfnet_br.py single-stage) ---
    parser.add_argument("--image_size",        type=int,   default=224)
    parser.add_argument("--patch_size",        type=int,   default=16)
    parser.add_argument("--d_model",           type=int,   default=256)
    parser.add_argument("--num_heads",         type=int,   default=8)
    parser.add_argument("--encoder_blocks",    type=int,   default=2)
    parser.add_argument("--main_blocks",       type=int,   default=6)
    parser.add_argument("--mlp_ratio",         type=float, default=3.0)
    parser.add_argument("--target_group_size", type=float, default=3.0)
    parser.add_argument("--router_proj_dim",   type=int,   default=64)

    # --- Decoder (MAE-style: narrower + shallower than encoder) ---
    parser.add_argument("--decoder_d_model", type=int, default=128)
    parser.add_argument("--decoder_blocks",  type=int, default=4)
    parser.add_argument("--decoder_heads",   type=int, default=4)
    parser.add_argument("--no_norm_pix", action="store_true",
                        help="Regress raw pixels instead of per-patch "
                             "normalised targets (MAE found normalised better)")

    # --- Training ---
    parser.add_argument("--batch_size",        type=int,   default=1024)
    parser.add_argument("--num_epochs",        type=int,   default=180)
    parser.add_argument("--lr",                type=float, default=3e-3)
    parser.add_argument("--weight_decay",      type=float, default=0.05)
    parser.add_argument("--grad_clip",         type=float, default=1.0)
    parser.add_argument("--warmup_epochs",     type=int,   default=10)
    parser.add_argument("--ratio_loss_weight", type=float, default=0.03)

    # --- Data ---
    parser.add_argument("--dataset_name",      type=str, default="clane9/imagenet-100")
    parser.add_argument("--dataset_cache_dir", type=str, default="./data")
    parser.add_argument("--jpeg_cache_dir",    type=str, default="./jpeg_cache")
    parser.add_argument("--num_workers",       type=int, default=8)

    # --- Infrastructure ---
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints_asfnet_ae")
    parser.add_argument("--resume",         type=str, default=None)
    parser.add_argument("--wandb_project",  type=str, default="asfnet")
    parser.add_argument("--wandb_entity",   type=str, default=None)
    parser.add_argument("--run_name",       type=str, default=None)
    parser.add_argument("--log_interval",   type=int, default=10000)
    parser.add_argument("--artifact_every", type=int, default=0,
                        help="Upload latest.pt to wandb as an artifact every "
                             "N epochs (0 = off). Protects against losing "
                             "checkpoints when a cloud instance dies.")
    parser.add_argument("--seed",           type=int, default=42)
    parser.add_argument("--device",         type=str, default="cuda")

    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if args.run_name is None:
        args.run_name = (
            f"AE_D{args.d_model}_enc{args.encoder_blocks}_main{args.main_blocks}"
            f"_N{args.target_group_size}_dec{args.decoder_d_model}x{args.decoder_blocks}"
        )

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

    scaler       = GradScaler()
    start_epoch  = 0
    best_val_rec = float("inf")
    global_step  = 0
    images_seen  = 0

    if args.resume:
        start_epoch, best_val_rec, images_seen = load_checkpoint(
            args.resume, model, optimizer, scaler, device)

    print(f"\nTraining '{args.run_name}' on {device}  (self-supervised, labels unused)")
    print(f"Epochs: {args.num_epochs}  |  Batch: {args.batch_size}  |  LR: {args.lr}\n")

    for epoch in range(start_epoch, args.num_epochs):
        current_lr = optimizer.param_groups[0]["lr"]
        wandb.log({"train/lr": current_lr}, step=global_step)

        train_rec, global_step, images_seen = run_epoch(
            model, train_loader, optimizer, scaler, args, epoch, device,
            global_step, images_seen, train=True)
        val_rec, global_step, _ = run_epoch(
            model, val_loader, optimizer, scaler, args, epoch, device,
            global_step, images_seen, train=False)
        scheduler.step()

        print(f"[{epoch+1:03d}/{args.num_epochs}] lr {current_lr:.2e} | "
              f"train rec {train_rec:.4f} | val rec {val_rec:.4f}")

        ckpt = {
            "epoch":        epoch,
            "model":        model.state_dict(),
            "optimizer":    optimizer.state_dict(),
            "scaler":       scaler.state_dict(),
            "val_rec":      val_rec,
            "best_val_rec": best_val_rec,
            "images_seen":  images_seen,
            "args":         vars(args),
        }
        save_checkpoint(ckpt, os.path.join(args.checkpoint_dir, "latest.pt"))

        if val_rec < best_val_rec:
            best_val_rec         = val_rec
            ckpt["best_val_rec"] = best_val_rec
            save_checkpoint(ckpt, os.path.join(args.checkpoint_dir, "best.pt"))
            print(f"  *** New best val rec: {best_val_rec:.4f}")

        if args.artifact_every and (epoch + 1) % args.artifact_every == 0:
            art = wandb.Artifact(f"asfnet-ae-{wandb.run.id}", type="model",
                                 metadata={"epoch": epoch, "val_rec": val_rec,
                                           "best_val_rec": best_val_rec})
            art.add_file(os.path.join(args.checkpoint_dir, "latest.pt"))
            wandb.log_artifact(art)

    wandb.finish()
    print(f"\nDone. Best val reconstruction loss: {best_val_rec:.4f}")


if __name__ == "__main__":
    main()
