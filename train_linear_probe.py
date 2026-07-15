"""
train_linear_probe.py — linear probe of a pretrained ASFNetAE backbone on
ImageNet-100 classification.

Loads an AE checkpoint (local path or wandb artifact), copies the
`backbone.` weights into a classifier ASFNetBR (keys line up by
construction), FREEZES everything except the linear classifier head, and
trains that head only.

This is the cheap comparable metric across AE ablations, not the ceiling —
MAE-style representations famously score much lower under a linear probe
than after full fine-tuning. Logging is intentionally minimal: loss and
accuracy (project `asfnet` by default, separate from the AE runs).

Usage (cloud):
    python train_linear_probe.py --ae_artifact asfnetAE/asfnet-ae-<id>:final
Usage (local checkpoint):
    python train_linear_probe.py --ae_checkpoint checkpoints_asfnet_ae/best.pt
"""

import os
import time
import random
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from tqdm import tqdm

from dataset import get_dataloaders
from model_asfnet_br import ASFNetBR
from utils import AverageMeter


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_backbone(args, device) -> tuple[ASFNetBR, dict]:
    """Resolve the AE checkpoint, build a classifier ASFNetBR with matching
    architecture, and load the pretrained backbone weights into it."""
    if args.ae_artifact:
        art = wandb.use_artifact(args.ae_artifact)
        ckpt_path = os.path.join(art.download(), "best.pt")
    elif args.ae_checkpoint:
        ckpt_path = args.ae_checkpoint
    else:
        raise SystemExit("need --ae_artifact or --ae_checkpoint")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    a = ckpt["args"]

    model = ASFNetBR(
        image_size        = a["image_size"],
        patch_size        = a["patch_size"],
        d_model           = a["d_model"],
        num_heads         = a["num_heads"],
        encoder_blocks    = a["encoder_blocks"],
        main_blocks       = a["main_blocks"],
        mlp_ratio         = a["mlp_ratio"],
        num_classes       = args.num_classes,
        target_group_size = a["target_group_size"],
        router_proj_dim   = a["router_proj_dim"],
    )

    # AE state dict: strip compile wrapper, keep only backbone.* keys.
    sd = {k.replace("_orig_mod.", "", 1): v for k, v in ckpt["model"].items()}
    backbone_sd = {k[len("backbone."):]: v for k, v in sd.items()
                   if k.startswith("backbone.")}
    missing, unexpected = model.load_state_dict(backbone_sd, strict=False)
    # Only the fresh classifier head may be missing; nothing may be unexpected.
    non_head_missing = [k for k in missing if not k.startswith("classifier")]
    assert not non_head_missing, f"backbone keys missing: {non_head_missing}"
    assert not unexpected, f"unexpected keys: {unexpected}"
    print(f"Loaded backbone from {ckpt_path} "
          f"(AE epoch {ckpt['epoch'] + 1}, {len(backbone_sd)} tensors); "
          f"fresh head: {missing}")

    model.to(device)

    # Freeze everything except the classifier head.
    for p in model.parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = True

    return model, a


def evaluate(model, loader, device):
    model.eval()
    losses, correct1, correct5, n = AverageMeter(), 0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits, _, _, _ = model(images)
                loss = F.cross_entropy(logits, labels)
            losses.update(loss.detach(), images.size(0))
            top5 = logits.topk(5, dim=1).indices
            correct1 += (top5[:, 0] == labels).sum()
            correct5 += (top5 == labels.unsqueeze(1)).any(dim=1).sum()
            n += images.size(0)
    return float(losses.avg), float(correct1) / n * 100, float(correct5) / n * 100


def main():
    parser = argparse.ArgumentParser(description="Linear probe for ASFNetAE")
    parser.add_argument("--ae_artifact",   type=str, default=None,
                        help="wandb artifact ref, e.g. asfnetAE/asfnet-ae-<id>:final")
    parser.add_argument("--ae_checkpoint", type=str, default=None)
    parser.add_argument("--num_classes",   type=int, default=100)

    parser.add_argument("--batch_size",   type=int,   default=1024)
    parser.add_argument("--num_epochs",   type=int,   default=50)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)

    parser.add_argument("--dataset_name",      type=str, default="clane9/imagenet-100")
    parser.add_argument("--dataset_cache_dir", type=str, default="./data")
    parser.add_argument("--jpeg_cache_dir",    type=str, default="./jpeg_cache")
    parser.add_argument("--num_workers",       type=int, default=16)

    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints_probe")
    parser.add_argument("--wandb_project",  type=str, default="asfnet")
    parser.add_argument("--run_name",       type=str, default=None)
    parser.add_argument("--image_size",     type=int, default=224)  # dataloader
    parser.add_argument("--seed",           type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")

    if args.run_name is None:
        src = args.ae_artifact or os.path.basename(args.ae_checkpoint or "")
        args.run_name = f"probe_{src.split('/')[-1].replace(':', '_')}"

    wandb.init(project=args.wandb_project, name=args.run_name,
               config=vars(args))

    model, ae_args = load_backbone(args, device)
    wandb.config.update({"ae_args": ae_args})
    model = torch.compile(model)

    train_loader, val_loader = get_dataloaders(args)

    head_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable params: {sum(p.numel() for p in head_params):,} (head only)")
    optimizer = torch.optim.AdamW(head_params, lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_epochs, eta_min=1e-5)

    best_acc1 = 0.0
    for epoch in range(args.num_epochs):
        # Frozen backbone has no dropout/BN, so train() only affects the head.
        model.train()
        losses = AverageMeter()
        t0 = time.perf_counter()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}",
                    leave=False)
        for step, (images, labels) in enumerate(pbar):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits, _, _, _ = model(images)
                loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.update(loss.detach(), images.size(0))
            if (step + 1) % 25 == 0:
                pbar.set_postfix(loss=f"{float(losses.avg):.4f}")
        scheduler.step()

        val_loss, acc1, acc5 = evaluate(model, val_loader, device)
        best_acc1 = max(best_acc1, acc1)
        wandb.log({"train/loss": float(losses.avg), "val/loss": val_loss,
                   "val/acc1": acc1, "val/acc5": acc5, "epoch": epoch + 1})
        print(f"[{epoch+1:03d}/{args.num_epochs}] "
              f"train {float(losses.avg):.4f} | val {val_loss:.4f} | "
              f"acc1 {acc1:.2f}% | acc5 {acc5:.2f}% | "
              f"{time.perf_counter() - t0:.0f}s")

        os.makedirs(args.checkpoint_dir, exist_ok=True)
        torch.save({"epoch": epoch, "model": model.state_dict(),
                    "acc1": acc1, "args": vars(args)},
                   os.path.join(args.checkpoint_dir, "latest.pt"))

    wandb.summary["best_acc1"] = best_acc1
    wandb.finish()
    print(f"\nDone. Best top-1: {best_acc1:.2f}%")


if __name__ == "__main__":
    main()
