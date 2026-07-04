import os
import random
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.cuda.amp import GradScaler
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from config import Config
from dataset import get_dataloaders, IMAGENET_MEAN, IMAGENET_STD
from model import SaccadeNet
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
# Loss
# ---------------------------------------------------------------------------

def compute_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    aux_preds: list,
    pos_history: list,
    pos_0: torch.Tensor,
    delta_history: list,
    cfg: Config,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Total loss = CrossEntropy(logits, labels)
               + loc_loss_weight    * mean_t( MSE(aux_pred_t, pos_t - pos_0) )
               + coverage_loss_weight * hinge_coverage

    Coverage loss — hinge on per-step movement magnitude (float32):
        coverage_loss = relu(min_step - ||delta_t||).mean()

    This replaces the broken variance formulation. Variance and its gradient
    are both exactly zero at the degenerate (constant delta) solution, so it
    cannot push the model out. The hinge has gradient -1 for every step below
    the threshold — nonzero and constant even when delta is exactly zero.

    Computed in float32 explicitly: bfloat16 rounds small deltas to zero near
    the collapse point, killing gradient signal even earlier than the math does.
    """
    task_loss = F.cross_entropy(logits, labels)

    aux_loss = torch.tensor(0.0, device=logits.device)
    for aux_pred, pos_t in zip(aux_preds, pos_history):
        true_disp = (pos_t - pos_0).detach()
        aux_loss  = aux_loss + F.mse_loss(aux_pred, true_disp)
    aux_loss = aux_loss / len(aux_preds)

    # Hinge coverage loss — always computed in float32
    deltas_f32    = torch.stack(delta_history, dim=1).float()  # (B, T, 2)
    step_sizes    = deltas_f32.norm(dim=-1)                    # (B, T)
    coverage_loss = F.relu(cfg.min_step - step_sizes).mean()   # zero when steps >= min_step

    total = (task_loss
             + cfg.loc_loss_weight      * aux_loss
             + cfg.coverage_loss_weight * coverage_loss)
    return total, task_loss, aux_loss, coverage_loss


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(state: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device,
):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    if "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
    start_epoch = ckpt["epoch"] + 1
    best_top1   = ckpt.get("best_top1", 0.0)
    print(f"Resumed from epoch {start_epoch} (best top-1 so far: {best_top1:.2f}%)")
    return start_epoch, best_top1


# ---------------------------------------------------------------------------
# Train / validate
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    cfg: Config,
    epoch: int,
    device: torch.device,
    global_step: int,
) -> tuple[float, float, float, int]:
    model.train()

    losses      = AverageMeter()
    task_losses = AverageMeter()
    aux_losses  = AverageMeter()
    cov_losses  = AverageMeter()
    top1        = AverageMeter()
    top5        = AverageMeter()

    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{cfg.num_epochs} [train]", leave=False)

    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        B = images.size(0)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, aux_preds, pos_history, pos_0, delta_history = model(images)
            loss, task_loss, aux_loss, coverage_loss = compute_loss(
                logits, labels, aux_preds, pos_history, pos_0, delta_history, cfg
            )

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        acc1, acc5 = accuracy(logits.float(), labels, topk=(1, 5))
        losses.update(loss.item(), B)
        task_losses.update(task_loss.item(), B)
        aux_losses.update(aux_loss.item(), B)
        cov_losses.update(coverage_loss.item(), B)
        top1.update(acc1, B)
        top5.update(acc5, B)

        pbar.set_postfix(loss=f"{losses.avg:.3f}", top1=f"{top1.avg:.1f}%",
                         cov=f"{cov_losses.avg:.4f}")

        if global_step % cfg.log_interval == 0:
            # Also log mean step size for easy diagnosis of movement collapse
            with torch.no_grad():
                deltas_f32 = torch.stack(delta_history, dim=1).float()
                mean_step  = deltas_f32.norm(dim=-1).mean().item()

            wandb.log({
                "train/loss":          losses.avg,
                "train/task_loss":     task_losses.avg,
                "train/aux_loss":      aux_losses.avg,
                "train/coverage_loss": cov_losses.avg,  # hinge: 0 = all steps ok
                "train/mean_step_size": mean_step,       # direct movement diagnostic
                "train/top1":          top1.avg,
                "train/top5":          top5.avg,
                "train/grad_scale":    scaler.get_scale(),
            }, step=global_step)

        global_step += 1

    return losses.avg, top1.avg, top5.avg, global_step


@torch.no_grad()
def validate(
    model: nn.Module,
    loader,
    cfg: Config,
    epoch: int,
    device: torch.device,
    global_step: int,
) -> tuple[float, float, float]:
    model.eval()

    losses = AverageMeter()
    top1   = AverageMeter()
    top5   = AverageMeter()

    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{cfg.num_epochs} [val]  ", leave=False)

    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        B = images.size(0)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, aux_preds, pos_history, pos_0, delta_history = model(images)
            loss, _, _, _ = compute_loss(
                logits, labels, aux_preds, pos_history, pos_0, delta_history, cfg
            )

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
# Per-epoch trajectory visualisation
# ---------------------------------------------------------------------------

def _denormalize(tensor: torch.Tensor) -> np.ndarray:
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std  = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img  = (tensor.cpu() * std + mean).clamp(0, 1)
    return img.permute(1, 2, 0).numpy()


@torch.no_grad()
def visualize_epoch_trajectories(
    model: nn.Module,
    loader,
    cfg: Config,
    device: torch.device,
    epoch: int,
    global_step: int,
    n_images: int = 8,
):
    model.eval()

    images, labels = next(iter(loader))
    images = images[:n_images].to(device)
    labels = labels[:n_images].to(device)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits, _, pos_history, pos_0, _ = model(images)

    preds  = logits.float().argmax(dim=1)
    H = W  = cfg.image_size
    half   = cfg.patch_size / 2
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, cfg.num_loops))

    fig, axes = plt.subplots(2, n_images // 2, figsize=(3 * (n_images // 2), 7))
    axes = axes.flatten()

    for idx in range(n_images):
        ax  = axes[idx]
        img = _denormalize(images[idx].cpu().float())
        ax.imshow(img)
        ax.axis("off")

        correct = preds[idx].item() == labels[idx].item()
        ax.set_title(
            f"{'✓' if correct else '✗'} p={preds[idx].item()} t={labels[idx].item()}",
            fontsize=7,
            color="green" if correct else "red",
        )

        prev_cx, prev_cy = None, None
        for t, (pos, color) in enumerate(zip(pos_history, colors)):
            cx = (pos[idx][0].item() + 1) / 2 * (W - 1)
            cy = (pos[idx][1].item() + 1) / 2 * (H - 1)

            rect = mpatches.Rectangle(
                (cx - half, cy - half), cfg.patch_size, cfg.patch_size,
                linewidth=1.2, edgecolor=color, facecolor="none", alpha=0.8,
            )
            ax.add_patch(rect)
            ax.text(cx, cy, str(t), fontsize=5, ha="center", va="center",
                    color=color, fontweight="bold")

            if prev_cx is not None:
                ax.annotate("", xy=(cx, cy), xytext=(prev_cx, prev_cy),
                            arrowprops=dict(arrowstyle="->", color=color, lw=0.8))
            prev_cx, prev_cy = cx, cy

    fig.suptitle(f"Epoch {epoch+1} — patch trajectories (dark=early, light=late)", fontsize=9)
    plt.tight_layout()
    wandb.log({"val/trajectories": wandb.Image(fig)}, step=global_step)
    plt.close(fig)

    model.train()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    cfg = Config()
    set_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    wandb.init(
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        config=cfg.__dict__,
    )

    train_loader, val_loader = get_dataloaders(cfg)

    model = SaccadeNet(cfg).to(device)
    param_counts = model.count_parameters()
    print("\nParameter counts:")
    for name, count in param_counts.items():
        print(f"  {name:<22} {count:>10,}")
    wandb.config.update({"param_counts": param_counts})
    # Only watch trainable params — backbone is frozen and enormous
    wandb.watch(model, log="gradients", log_freq=cfg.log_interval)

    # Optimizer only sees trainable parameters — backbone excluded automatically
    # since requires_grad=False params are skipped by AdamW
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1e-6,
        end_factor=1.0,
        total_iters=cfg.warmup_epochs,
    )
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.num_epochs - cfg.warmup_epochs,
        eta_min=1e-6,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_sched, cosine_sched],
        milestones=[cfg.warmup_epochs],
    )

    scaler      = GradScaler()
    start_epoch = 0
    best_top1   = 0.0
    global_step = 0

    if cfg.resume:
        start_epoch, best_top1 = load_checkpoint(
            cfg.resume, model, optimizer, scaler, device
        )

    print(f"\nTraining on {device} — frozen MobileNetV2 backbone")
    print(f"Epochs: {cfg.num_epochs}  |  Batch: {cfg.batch_size}  |  LR: {cfg.lr}\n")

    for epoch in range(start_epoch, cfg.num_epochs):
        current_lr = optimizer.param_groups[0]["lr"]
        wandb.log({"train/lr": current_lr}, step=global_step)

        train_loss, train_top1, train_top5, global_step = train_one_epoch(
            model, train_loader, optimizer, scaler, cfg, epoch, device, global_step
        )
        val_loss, val_top1, val_top5 = validate(
            model, val_loader, cfg, epoch, device, global_step
        )
        visualize_epoch_trajectories(
            model, val_loader, cfg, device, epoch, global_step
        )
        scheduler.step()

        print(
            f"[{epoch+1:03d}/{cfg.num_epochs}] "
            f"lr {current_lr:.2e} | "
            f"train loss {train_loss:.3f} top1 {train_top1:.1f}% | "
            f"val loss {val_loss:.3f} top1 {val_top1:.1f}% top5 {val_top5:.1f}%"
        )

        checkpoint_state = {
            "epoch":     epoch,
            "model":     model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler":    scaler.state_dict(),
            "val_top1":  val_top1,
            "best_top1": best_top1,
            "cfg":       cfg,
        }
        save_checkpoint(checkpoint_state, os.path.join(cfg.checkpoint_dir, "latest.pt"))

        if val_top1 > best_top1:
            best_top1 = val_top1
            checkpoint_state["best_top1"] = best_top1
            save_checkpoint(checkpoint_state, os.path.join(cfg.checkpoint_dir, "best.pt"))
            print(f"  *** New best: {best_top1:.2f}%")

    wandb.finish()
    print(f"\nDone. Best val top-1: {best_top1:.2f}%")


if __name__ == "__main__":
    main()