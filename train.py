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

from config import Config
from dataset import get_dataloaders
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
    cfg: Config,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Total loss = CrossEntropy(logits, labels)
               + loc_loss_weight * mean_t( MSE(aux_pred_t, pos_t - pos_0) )

    The auxiliary loss anchors a linear subspace of LocTracker's hidden state
    to real cumulative displacement, without constraining the rest of the state.

    pos_history[t] is detached before use as a regression target — we don't
    want gradients flowing back through the true positions into MoveNet via
    the loss target, only via the task loss path.
    """
    task_loss = F.cross_entropy(logits, labels)

    aux_loss = torch.tensor(0.0, device=logits.device)
    for aux_pred, pos_t in zip(aux_preds, pos_history):
        true_disp = (pos_t - pos_0).detach()  # (B, 2)
        aux_loss = aux_loss + F.mse_loss(aux_pred, true_disp)
    aux_loss = aux_loss / len(aux_preds)

    # Coverage loss: penalise low variance in patch positions across steps.
    # If the model camps in one spot, var -> 0 and this term is maximally penalised.
    # Detach positions so this only pushes MoveNet, not the CNN or OutputNet.
    positions = torch.stack([p.detach() for p in pos_history], dim=1)  # (B, T, 2)
    coverage_loss = -positions.var(dim=1).mean()

    total = (task_loss
             + cfg.loc_loss_weight * aux_loss
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

    losses          = AverageMeter()
    task_losses     = AverageMeter()
    aux_losses      = AverageMeter()
    top1            = AverageMeter()
    top5            = AverageMeter()

    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{cfg.num_epochs} [train]", leave=False)

    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        B = images.size(0)

        # Forward in bfloat16
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, aux_preds, pos_history, pos_0 = model(images)
            loss, task_loss, aux_loss, coverage_loss = compute_loss(
                logits, labels, aux_preds, pos_history, pos_0, cfg
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
        top1.update(acc1, B)
        top5.update(acc5, B)

        pbar.set_postfix(loss=f"{losses.avg:.3f}", top1=f"{top1.avg:.1f}%")

        if global_step % cfg.log_interval == 0:
            wandb.log({
                "train/loss":          losses.avg,
                "train/task_loss":     task_losses.avg,
                "train/aux_loss":      aux_losses.avg,
                "train/coverage_loss": coverage_loss.item(),
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
            logits, aux_preds, pos_history, pos_0 = model(images)
            loss, _, _, _ = compute_loss(logits, labels, aux_preds, pos_history, pos_0, cfg)

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
    cfg = Config()
    set_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # Logging
    wandb.init(
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        config=cfg.__dict__,
    )

    # Data
    train_loader, val_loader = get_dataloaders(cfg)

    # Model
    model = SaccadeNet(cfg).to(device)

    torch.compile(model)

    param_counts = model.count_parameters()
    print("\nParameter counts:")
    for name, count in param_counts.items():
        print(f"  {name:<16} {count:>10,}")
    wandb.config.update({"param_counts": param_counts})
    wandb.watch(model, log="gradients", log_freq=cfg.log_interval)

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
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

    # Mixed precision scaler
    # bfloat16 has float32-range exponents so loss scaling rarely triggers,
    # but keeping the scaler is harmless and handles edge cases gracefully.
    scaler = GradScaler()

    # Resume
    start_epoch = 0
    best_top1   = 0.0
    global_step = 0

    if cfg.resume:
        start_epoch, best_top1 = load_checkpoint(
            cfg.resume, model, optimizer, scaler, device
        )

    print(f"\nTraining on {device} with bfloat16 autocast")
    print(f"Epochs: {cfg.num_epochs}  |  Batch: {cfg.batch_size}  |  LR: {cfg.lr}\n")

    # Training loop
    for epoch in range(start_epoch, cfg.num_epochs):
        current_lr = optimizer.param_groups[0]["lr"]
        wandb.log({"train/lr": current_lr}, step=global_step)

        train_loss, train_top1, train_top5, global_step = train_one_epoch(
            model, train_loader, optimizer, scaler, cfg, epoch, device, global_step
        )
        val_loss, val_top1, val_top5 = validate(
            model, val_loader, cfg, epoch, device, global_step
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

        save_checkpoint(
            checkpoint_state,
            os.path.join(cfg.checkpoint_dir, "latest.pt"),
        )

        if val_top1 > best_top1:
            best_top1 = val_top1
            checkpoint_state["best_top1"] = best_top1
            save_checkpoint(
                checkpoint_state,
                os.path.join(cfg.checkpoint_dir, "best.pt"),
            )
            print(f"  *** New best: {best_top1:.2f}%")

    wandb.finish()
    print(f"\nDone. Best val top-1: {best_top1:.2f}%")


if __name__ == "__main__":
    main()