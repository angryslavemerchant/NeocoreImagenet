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
import time
import random
import argparse

import numpy as np
import torch
import torch.nn as nn
import wandb
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
        keep_budget       = args.keep_budget,
        keep_ratio_target = args.keep_ratio_target,
        xattn_slots       = args.xattn_slots,
    )


def save_checkpoint(state: dict, path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path, model, optimizer, device):
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    # older checkpoints carry a GradScaler state — obsolete (bf16 needs no
    # scaler), ignored on load
    start_epoch = ckpt["epoch"] + 1
    best_val    = ckpt.get("best_val_rec", float("inf"))
    images_seen = ckpt.get("images_seen", 0)
    print(f"Resumed from epoch {start_epoch}  (best val rec so far: {best_val:.4f})")
    return start_epoch, best_val, images_seen


def run_epoch(model, loader, optimizer, args, epoch, device,
              global_step, images_seen, train: bool):
    model.train() if train else model.eval()

    # Meters are fed 0-dim GPU tensors — averages stay on-device and are
    # only .item()'d at logging points. Anything per-step that touches the
    # value (f-strings included) forces a CPU/GPU sync and serialises the
    # pipeline, which dominates step time for a model this small.
    rec_losses   = AverageMeter()
    ratio_losses = AverageMeter()
    keep_losses  = AverageMeter()
    kept         = AverageMeter()
    dropf        = AverageMeter()

    tag  = "train" if train else "val"
    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.num_epochs} [{tag}]", leave=False)

    t_epoch   = time.perf_counter()
    data_wait = 0.0
    n_images  = 0
    step_in_epoch = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        t_iter = time.perf_counter()
        for images, _labels in pbar:   # labels ignored — self-supervised
            data_wait += time.perf_counter() - t_iter
            images = images.to(device, non_blocking=True)
            B = images.size(0)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss_rec, l_ratio, l_keep, mean_kept, _mean_groups, drop_frac = \
                    model(images)
                loss = (loss_rec
                        + args.ratio_loss_weight * l_ratio
                        + args.keep_loss_weight * l_keep)

            if train:
                optimizer.zero_grad()
                loss.backward()   # bf16 autocast — no GradScaler needed
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

            rec_losses.update(loss_rec.detach(), B)
            ratio_losses.update(l_ratio.detach(), B)
            keep_losses.update(l_keep.detach(), B)
            kept.update(mean_kept, B)
            dropf.update(drop_frac, B)
            n_images      += B
            step_in_epoch += 1

            if step_in_epoch % 25 == 0:   # sparse: each refresh syncs
                pbar.set_postfix(rec=f"{float(rec_losses.avg):.4f}",
                                 kept=f"{float(kept.avg):.1f}",
                                 drop=f"{float(dropf.avg):.2f}")

            if train:
                prev_images  = images_seen
                images_seen += B
                if images_seen // args.log_interval != prev_images // args.log_interval:
                    wandb.log({
                        "train/rec_loss":    float(rec_losses.avg),
                        "train/ratio_loss":  float(ratio_losses.avg),
                        "train/keep_loss":   float(keep_losses.avg),
                        "train/mean_kept":   float(kept.avg),
                        "train/drop_frac":   float(dropf.avg),
                        "train/images_seen": images_seen,
                    }, step=global_step)
                global_step += 1
            t_iter = time.perf_counter()

    epoch_sec = time.perf_counter() - t_epoch
    sys_stats = {
        f"sys/{tag}_epoch_sec":      round(epoch_sec, 1),
        f"sys/{tag}_imgs_per_sec":   round(n_images / max(epoch_sec, 1e-9), 1),
        f"sys/{tag}_data_wait_frac": round(data_wait / max(epoch_sec, 1e-9), 3),
    }

    if not train:
        wandb.log({
            "val/rec_loss":   float(rec_losses.avg),
            "val/ratio_loss": float(ratio_losses.avg),
            "val/keep_loss":  float(keep_losses.avg),
            "val/mean_kept":  float(kept.avg),
            "val/drop_frac":  float(dropf.avg),
        }, step=global_step)

    return float(rec_losses.avg), global_step, images_seen, sys_stats


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

    # --- Compression enforcement (ablations; both default OFF) ---
    parser.add_argument("--keep_budget", type=float, default=0.0,
                        help="Hard bottleneck: max fraction of patches that "
                             "may enter the decoder (top-k by boundary "
                             "evidence); the rest are masked + reconstructed. "
                             "e.g. 0.25. 0 = off.")
    parser.add_argument("--keep_ratio_target", type=float, default=0.0,
                        help="Token-level keep-rate target for the H-Net "
                             "loss on keep fraction (e.g. 0.25). 0 = off.")
    parser.add_argument("--keep_loss_weight", type=float, default=0.03,
                        help="Weight on the token-level keep-rate loss "
                             "(only active with --keep_ratio_target > 0).")
    parser.add_argument("--xattn_slots", type=int, default=0,
                        help="Slot bottleneck: S learned queries cross-attend "
                             "over the survivors and only the S x D code "
                             "reaches the decoder (loss on ALL patches). "
                             "e.g. 49 to rate-match --keep_budget 0.25. "
                             "0 = off; mutually exclusive with keep_budget.")

    # --- Data ---
    parser.add_argument("--dataset_name",      type=str, default="clane9/imagenet-100")
    parser.add_argument("--dataset_cache_dir", type=str, default="./data")
    parser.add_argument("--jpeg_cache_dir",    type=str, default="./jpeg_cache")
    parser.add_argument("--num_workers",       type=int, default=8)

    # --- Infrastructure ---
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints_asfnet_ae")
    parser.add_argument("--resume",         type=str, default=None)
    parser.add_argument("--resume_artifact", type=str, default=None,
                        help="wandb artifact ref to resume from (downloads "
                             "latest.pt), e.g. asfnetAE/asfnet-ae-<id>:latest. "
                             "Model args must match the current CLI args.")
    parser.add_argument("--wandb_project",  type=str,
                        default=os.environ.get("WANDB_PROJECT", "asfnetAE"))
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
    torch.set_float32_matmul_precision("high")   # TF32 for the fp32 paths
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if args.run_name is None:
        args.run_name = (
            f"AE_D{args.d_model}_enc{args.encoder_blocks}_main{args.main_blocks}"
            f"_N{args.target_group_size}_dec{args.decoder_d_model}x{args.decoder_blocks}"
        )

    wandb.init(project=args.wandb_project, entity=args.wandb_entity,
               name=args.run_name, config=vars(args))

    if args.resume_artifact:
        art = wandb.use_artifact(args.resume_artifact)
        for attempt in range(6):   # same 403 blips as the probe download
            try:
                args.resume = os.path.join(art.download(), "latest.pt")
                break
            except Exception as e:
                if attempt == 5:
                    raise
                wait = 60 * (attempt + 1)
                print(f"resume artifact download failed ({e!r}) — "
                      f"retry {attempt + 1}/5 in {wait}s")
                time.sleep(wait)

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

    start_epoch  = 0
    best_val_rec = float("inf")
    global_step  = 0
    images_seen  = 0

    if args.resume:
        start_epoch, best_val_rec, images_seen = load_checkpoint(
            args.resume, model, optimizer, device)
        # Scheduler state isn't checkpointed — fast-forward it so the LR
        # continues the cosine decay instead of restarting warmup mid-run.
        # (Triggers the "step before optimizer.step()" warning; harmless.)
        for _ in range(start_epoch):
            scheduler.step()

    print(f"\nTraining '{args.run_name}' on {device}  (self-supervised, labels unused)")
    print(f"Epochs: {args.num_epochs}  |  Batch: {args.batch_size}  |  LR: {args.lr}\n")

    for epoch in range(start_epoch, args.num_epochs):
        current_lr = optimizer.param_groups[0]["lr"]
        wandb.log({"train/lr": current_lr}, step=global_step)

        train_rec, global_step, images_seen, train_sys = run_epoch(
            model, train_loader, optimizer, args, epoch, device,
            global_step, images_seen, train=True)
        val_rec, global_step, _, val_sys = run_epoch(
            model, val_loader, optimizer, args, epoch, device,
            global_step, images_seen, train=False)
        scheduler.step()

        peak_vram_gb = round(torch.cuda.max_memory_allocated() / 2**30, 2) \
            if device.type == "cuda" else 0.0
        wandb.log({**train_sys, **val_sys, "sys/peak_vram_gb": peak_vram_gb},
                  step=global_step)

        print(f"[{epoch+1:03d}/{args.num_epochs}] lr {current_lr:.2e} | "
              f"train rec {train_rec:.4f} | val rec {val_rec:.4f} | "
              f"{train_sys['sys/train_epoch_sec']:.0f}s "
              f"({train_sys['sys/train_imgs_per_sec']:.0f} img/s, "
              f"data-wait {train_sys['sys/train_data_wait_frac']:.0%}) | "
              f"peak VRAM {peak_vram_gb:.1f} GB")

        if epoch == start_epoch:
            # One-time compile diagnostics: fragmentation here means
            # torch.compile is falling back to eager between graph breaks.
            try:
                from torch._dynamo.utils import counters
                stats  = dict(counters["stats"])
                breaks = sum(counters["graph_break"].values())
                print(f"[compile] stats={stats} graph_break_sites={breaks}")
                for reason, cnt in sorted(counters["graph_break"].items(),
                                          key=lambda kv: -kv[1])[:5]:
                    print(f"[compile]   x{cnt}  {str(reason)[:140]}")
            except Exception as e:
                print(f"[compile] diagnostics unavailable: {e}")

        ckpt = {
            "epoch":        epoch,
            "model":        model.state_dict(),
            "optimizer":    optimizer.state_dict(),
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
            # Checkpoint backup only — never let a wandb storage blip kill
            # the training run (their GCS 403'd project-wide 2026-07-15).
            try:
                art = wandb.Artifact(f"asfnet-ae-{wandb.run.id}", type="model",
                                     metadata={"epoch": epoch, "val_rec": val_rec,
                                               "best_val_rec": best_val_rec})
                art.add_file(os.path.join(args.checkpoint_dir, "latest.pt"))
                wandb.log_artifact(art)
            except Exception as e:
                print(f"[artifact] upload failed at epoch {epoch + 1}: {e!r} "
                      f"— continuing")

    wandb.finish()
    print(f"\nDone. Best val reconstruction loss: {best_val_rec:.4f}")


if __name__ == "__main__":
    main()
