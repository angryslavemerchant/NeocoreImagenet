"""
train_neocore.py — self-supervised pretraining for NeocoreAE (the
recursive-admission autoencoder; see model_neocore.py for the why).

The objective is reconstruction ONLY — no ratio loss, no keep loss, no
auxiliary anything: K and R are architectural constants, so there is
nothing left to regularise. What gets logged instead are the nullity
alarms (overlap_r1, admit_corr): if by ~epoch 20 overlap_r1 is pinned at
1.0 and admit_corr at -1.0, the loop has degenerated into a sorted
one-shot and the run should be killed rather than trained out.

Direct comparison target: AE_budget25 (val rec 0.101) — same patch size,
same 49/196 rate, same decoder, same dropped-only loss convention.
Caveat when comparing: Neocore's admitted tokens keep attending to the
full grid every round, where budget25's survivors only saw each other
post-selection — Neocore's encoder has strictly more information flow.

Checkpoint selection: lowest validation reconstruction loss.
wandb project: `neocore` (new era, new project).
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

from model_neocore import NeocoreAE
from utils import AverageMeter


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(args) -> NeocoreAE:
    return NeocoreAE(
        image_size       = args.image_size,
        patch_size       = args.patch_size,
        in_channels      = 3,
        d_model          = args.d_model,
        num_heads        = args.num_heads,
        core_blocks      = args.core_blocks,
        mlp_ratio        = args.mlp_ratio,
        rounds           = args.rounds,
        memory_tokens    = args.memory_tokens,
        decoder_d_model  = args.decoder_d_model,
        decoder_blocks   = args.decoder_blocks,
        decoder_heads    = args.decoder_heads,
        norm_pix_loss     = not args.no_norm_pix,
        checkpoint_rounds = args.checkpoint_rounds,
    )


def save_checkpoint(state: dict, path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path, model, optimizer, device):
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    start_epoch = ckpt["epoch"] + 1
    best_val    = ckpt.get("best_val_rec", float("inf"))
    images_seen = ckpt.get("images_seen", 0)
    print(f"Resumed from epoch {start_epoch}  (best val rec so far: {best_val:.4f})")
    return start_epoch, best_val, images_seen


def run_epoch(model, loader, optimizer, args, epoch, device,
              global_step, images_seen, train: bool):
    model.train() if train else model.eval()

    # Meters hold 0-dim GPU tensors; .item() only at logging points —
    # per-step syncs dominate step time for a model this small.
    rec_losses = AverageMeter()
    overlaps   = AverageMeter()
    corrs      = AverageMeter()

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
                loss_rec, overlap_r1, admit_corr = model(images)

            if train:
                optimizer.zero_grad()
                loss_rec.backward()   # bf16 autocast — no GradScaler needed
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

            rec_losses.update(loss_rec.detach(), B)
            overlaps.update(overlap_r1, B)
            corrs.update(admit_corr, B)
            n_images      += B
            step_in_epoch += 1

            if step_in_epoch % 25 == 0:   # sparse: each refresh syncs
                pbar.set_postfix(rec=f"{float(rec_losses.avg):.4f}",
                                 ovl=f"{float(overlaps.avg):.2f}")

            if train:
                prev_images  = images_seen
                images_seen += B
                if images_seen // args.log_interval != prev_images // args.log_interval:
                    wandb.log({
                        "train/rec_loss":    float(rec_losses.avg),
                        "train/overlap_r1":  float(overlaps.avg),
                        "train/admit_corr":  float(corrs.avg),
                        "train/images_seen": images_seen,
                    }, step=global_step)
                global_step += 1
            t_iter = time.perf_counter()

    epoch_sec = time.perf_counter() - t_epoch
    # data_wait_frac counts time blocked in the loader's __next__ — under
    # async CUDA the CPU races ahead and drains the GPU's queue THERE, so
    # a compute-bound run can read 85% "data wait" (measured 2026-07-17:
    # RAM loader + DALI identical epoch times, nvidia-smi 100% util).
    # Treat it as an upper bound on loader stall, not a measurement.
    sys_stats = {
        f"sys/{tag}_epoch_sec":      round(epoch_sec, 1),
        f"sys/{tag}_imgs_per_sec":   round(n_images / max(epoch_sec, 1e-9), 1),
        f"sys/{tag}_data_wait_frac": round(data_wait / max(epoch_sec, 1e-9), 3),
    }

    if not train:
        wandb.log({
            "val/rec_loss":   float(rec_losses.avg),
            "val/overlap_r1": float(overlaps.avg),
            "val/admit_corr": float(corrs.avg),
        }, step=global_step)

    return float(rec_losses.avg), global_step, images_seen, sys_stats


def main():
    parser = argparse.ArgumentParser(description="Neocore recursive-admission AE")

    # --- Core (defaults replicate AE_budget25: 2 enc + 6 main = 8 blocks,
    #     d=256, h=8; the split disappears because there is no selection
    #     boundary inside the pass anymore) ---
    parser.add_argument("--image_size",  type=int,   default=224)
    parser.add_argument("--patch_size",  type=int,   default=16)
    parser.add_argument("--d_model",     type=int,   default=256)
    parser.add_argument("--num_heads",   type=int,   default=8)
    parser.add_argument("--core_blocks", type=int,   default=8)
    parser.add_argument("--mlp_ratio",   type=float, default=3.0)

    # --- The loop (both architectural constants — the law) ---
    parser.add_argument("--rounds", type=int, default=7,
                        help="R recursive passes of the shared core; "
                             "R=1 == the one-shot budget model (control).")
    parser.add_argument("--memory_tokens", type=int, default=49,
                        help="K tokens in working memory at the end; "
                             "K/R admitted per round, exact.")

    # --- Decoder (MAE-style: narrower + shallower than the core) ---
    parser.add_argument("--decoder_d_model", type=int, default=128)
    parser.add_argument("--decoder_blocks",  type=int, default=4)
    parser.add_argument("--decoder_heads",   type=int, default=4)
    parser.add_argument("--no_norm_pix", action="store_true",
                        help="Regress raw pixels instead of per-patch "
                             "normalised targets (MAE found normalised better)")

    # --- Training (matches AE_budget25's recipe) ---
    parser.add_argument("--batch_size",    type=int,   default=1024)
    parser.add_argument("--num_epochs",    type=int,   default=180)
    parser.add_argument("--lr",            type=float, default=3e-3)
    parser.add_argument("--weight_decay",  type=float, default=0.05)
    parser.add_argument("--grad_clip",     type=float, default=1.0)
    parser.add_argument("--warmup_epochs", type=int,   default=10)
    parser.add_argument("--checkpoint_rounds", type=int, default=-1,
                        help="Gradient-checkpoint the first N rounds "
                             "(-1 = all, 0 = none). Measured at batch 1024 "
                             "on 96 GB: all ~21 GB (+33%% recompute), "
                             "0 OOMs (~90 GB), 3 ~57 GB (+14%% recompute) — "
                             "the speed/memory dial.")
    parser.add_argument("--compile_mode", type=str, default="max-autotune",
                        help="torch.compile mode. max-autotune costs a few "
                             "extra minutes at boot, worth it over a full "
                             "run; use 'default' if autotune misbehaves.")

    # --- Data ---
    parser.add_argument("--dataset_name",      type=str, default="clane9/imagenet-100")
    parser.add_argument("--dataset_cache_dir", type=str, default="./data")
    parser.add_argument("--jpeg_cache_dir",    type=str, default="./jpeg_cache")
    parser.add_argument("--num_workers",       type=int, default=8)
    parser.add_argument("--data", type=str, default="dali",
                        choices=["dali", "ram"],
                        help="dali: decode-per-epoch pipeline (dataset.py). "
                             "ram: raw uint8 blob resident in RAM/VRAM with "
                             "GPU-side augmentation (dataset_ram.py) — a "
                             "pipeline epoch; don't mix within a series.")
    parser.add_argument("--data_device", type=str, default="cpu",
                        choices=["cpu", "cuda"],
                        help="ram only: where the 25 GB blob lives. cuda = "
                             "fully VRAM-resident, zero PCIe per step "
                             "(needs headroom above the model's peak).")

    # --- Infrastructure ---
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                        help="default: runs/<run_name> — one folder per run "
                             "(checkpoints + eval viz), gitignored. wandb is "
                             "logging only; local disk is the system of record.")
    parser.add_argument("--resume",          type=str, default=None)
    parser.add_argument("--resume_artifact", type=str, default=None,
                        help="wandb artifact ref to resume from (downloads "
                             "latest.pt). Model args must match the CLI args.")
    parser.add_argument("--wandb_project", type=str,
                        default=os.environ.get("WANDB_PROJECT", "neocore"))
    parser.add_argument("--wandb_entity",  type=str, default=None)
    parser.add_argument("--run_name",      type=str, default=None)
    parser.add_argument("--log_interval",  type=int, default=10000)
    parser.add_argument("--artifact_every", type=int, default=0,
                        help="Upload latest.pt to wandb as an artifact every "
                             "N epochs (0 = off). Protects against losing "
                             "checkpoints when a cloud instance dies.")
    parser.add_argument("--seed",   type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    set_seed(args.seed)
    torch.set_float32_matmul_precision("high")   # TF32 for the fp32 paths
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if args.run_name is None:
        args.run_name = (f"NC_R{args.rounds}_K{args.memory_tokens}"
                         f"_D{args.d_model}x{args.core_blocks}")

    # Per-run local folder (system of record; wandb is logging only).
    # runs/LATEST points at it so vast/run_training.sh can find best.pt.
    if args.checkpoint_dir is None:
        args.checkpoint_dir = os.path.join("runs", args.run_name)
    os.makedirs("runs", exist_ok=True)
    with open(os.path.join("runs", "LATEST"), "w") as f:
        f.write(args.checkpoint_dir)

    wandb.init(project=args.wandb_project, entity=args.wandb_entity,
               name=args.run_name, config=vars(args))

    if args.resume_artifact:
        art = wandb.use_artifact(args.resume_artifact)
        for attempt in range(24):   # patient: outlast a wandb storage outage
            try:
                args.resume = os.path.join(art.download(), "latest.pt")
                break
            except Exception as e:
                if attempt == 23:
                    raise
                wait = min(300, 60 * (attempt + 1))
                print(f"resume artifact download failed ({e!r}) — "
                      f"retry {attempt + 1}/23 in {wait}s")
                time.sleep(wait)

    if args.data == "ram":
        from dataset_ram import get_ram_dataloaders
        train_loader, val_loader = get_ram_dataloaders(args)
    else:
        from dataset import get_dataloaders   # imports DALI — instance only
        train_loader, val_loader = get_dataloaders(args)

    model = build_model(args).to(device)
    model = torch.compile(model, mode=args.compile_mode)

    param_counts = model.count_parameters()
    print("\nParameter counts:")
    for name, count in param_counts.items():
        print(f"  {name:<16} {count:>10,}")
    print(f"Admission schedule: {model.admit_schedule} "
          f"(K={args.memory_tokens} over R={args.rounds} rounds)")
    wandb.config.update({"param_counts": param_counts})

    # fused AdamW: same math, one kernel — a few % on a model this small
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay,
                                  fused=(device.type == "cuda"))
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
        # Scheduler state isn't checkpointed — fast-forward the cosine.
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
                art = wandb.Artifact(f"neocore-{wandb.run.id}", type="model",
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
