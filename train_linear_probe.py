"""
train_linear_probe.py — frozen-backbone probe of a pretrained ASFNetAE
backbone on ImageNet-100 classification.

Loads an AE checkpoint (local path or wandb artifact), copies the
`backbone.` weights into an ASFNetBR (keys line up by construction),
FREEZES the backbone, and trains only a probe head.

Two probe axes, both settable from the CLI:

  --pool attentive|mean   attentive (default) = single learned query
                          cross-attending over the probed tokens (MAP
                          head, the standard MAE/I-JEPA eval); mean =
                          masked GAP + linear (the original cheap probe).
  --topk N                probe only the top-N survivors ranked by the
                          router's boundary evidence — the SAME ranking
                          the AE's keep_budget bottleneck used, so the
                          probe sees exactly the tokens that received
                          reconstruction gradient. Default (unset) reads
                          keep_budget from the checkpoint and matches the
                          training bottleneck automatically (49 for
                          budget 0.25); 0 probes all survivors.

This is the cheap comparable metric across AE ablations, not the ceiling —
MAE-style representations famously score much lower under a probe than
after full fine-tuning. Logging is intentionally minimal: loss and
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
from model_asfnet_ae_ladder import ASFNetAELadder
from utils import AverageMeter


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class AttentivePool(nn.Module):
    """MAP head: one learned query cross-attends over the token set."""

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.query, std=0.02)
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, feats: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        q = self.query.expand(feats.shape[0], -1, -1)
        out, _ = self.attn(q, feats, feats,
                           key_padding_mask=pad_mask, need_weights=False)
        return self.norm(out[:, 0])


class ProbeModel(nn.Module):
    """Frozen ASFNetBR backbone + trainable probe head.

    topk > 0 reproduces the AE's `_apply_budget` selection: survivors are
    ranked by detached boundary evidence and only the top-K enter the pool,
    so the probe grades exactly the tokens the bottleneck trained. topk = 0
    pools all router-kept survivors (the original probe's behaviour, and a
    known mismatch when the AE trained with a budget).
    """

    def __init__(self, backbone: ASFNetBR, d_model: int, num_heads: int,
                 num_classes: int, pool: str, topk: int):
        super().__init__()
        self.backbone = backbone
        self.pool = pool
        self.topk = topk
        self.attn_pool = AttentivePool(d_model, num_heads) if pool == "attentive" else None
        self.head = nn.Linear(d_model, num_classes)

        for p in self.backbone.parameters():
            p.requires_grad = False

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if isinstance(self.backbone, ASFNetAELadder):
            # Ladder: pool the final (~<=49) survivors; the budget already
            # selected them, so no probe-side topk.
            feats, pad_mask, *_ = self.backbone.forward_features(images)
            if self.attn_pool is not None:
                pooled = self.attn_pool(feats, pad_mask)
            else:
                real_mask = (~pad_mask).float()
                pooled = (feats * real_mask.unsqueeze(-1)).sum(dim=1) \
                    / real_mask.sum(dim=1, keepdim=True).clamp(min=1)
            return self.head(pooled)

        feats, _, pad_mask, sel, _, _, _, _, s, _ = \
            self.backbone.forward_features(images)

        if self.topk > 0 and pad_mask.shape[1] > self.topk:
            # Same ranking key as ASFNetAE._apply_budget. Images with fewer
            # than topk survivors pull pad slots into the selection; the
            # gathered pad_mask keeps them out of the pool.
            s_slot = s.detach().gather(1, sel)
            s_slot = s_slot.masked_fill(pad_mask, float("-inf"))
            top = s_slot.topk(self.topk, dim=1).indices          # (B, topk)
            feats = feats.gather(
                1, top.unsqueeze(-1).expand(-1, -1, feats.shape[-1]))
            pad_mask = pad_mask.gather(1, top)

        if self.attn_pool is not None:
            pooled = self.attn_pool(feats, pad_mask)
        else:
            real_mask = (~pad_mask).float()
            pooled = (feats * real_mask.unsqueeze(-1)).sum(dim=1) \
                / real_mask.sum(dim=1, keepdim=True).clamp(min=1)

        return self.head(pooled)


def load_backbone(args, device) -> tuple[ASFNetBR, dict]:
    """Resolve the AE checkpoint, build a headless ASFNetBR with matching
    architecture, and load the pretrained backbone weights into it."""
    if args.ae_artifact:
        art = wandb.use_artifact(args.ae_artifact)
        # wandb's GCS storage 403s in incidents lasting minutes-to-hours
        # (observed 2026-07-15, took out two cloud runs at boot) — retry
        # patiently (~2 h) so a booted instance outlives the outage.
        for attempt in range(24):
            try:
                ckpt_path = os.path.join(art.download(), "best.pt")
                break
            except Exception as e:
                if attempt == 23:
                    raise
                wait = min(300, 60 * (attempt + 1))
                print(f"artifact download failed ({e!r}) — "
                      f"retry {attempt + 1}/23 in {wait}s")
                time.sleep(wait)
    elif args.ae_checkpoint:
        ckpt_path = args.ae_checkpoint
    else:
        raise SystemExit("need --ae_artifact or --ae_checkpoint")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    a = ckpt["args"]

    if a.get("ladder", False):
        # Ladder AE: the whole model (minus decoder usage) is the backbone;
        # keys load 1:1, decoder weights included but unused by the probe.
        model = ASFNetAELadder(
            image_size        = a["image_size"],
            patch_size        = a["patch_size"],
            mlp_ratio         = a["mlp_ratio"],
            target_group_size = a["target_group_size"],
            router_proj_dim   = a["router_proj_dim"],
            norm_pix_loss     = not a.get("no_norm_pix", False),
        )
        sd = {k.replace("_orig_mod.", "", 1): v for k, v in ckpt["model"].items()}
        model.load_state_dict(sd, strict=True)
        print(f"Loaded ladder model from {ckpt_path} "
              f"(AE epoch {ckpt['epoch'] + 1}, {len(sd)} tensors)")
        model.to(device)
        return model, a

    model = ASFNetBR(
        image_size        = a["image_size"],
        patch_size        = a["patch_size"],
        d_model           = a["d_model"],
        num_heads         = a["num_heads"],
        encoder_blocks    = a["encoder_blocks"],
        main_blocks       = a["main_blocks"],
        mlp_ratio         = a["mlp_ratio"],
        num_classes       = 0,
        target_group_size = a["target_group_size"],
        router_proj_dim   = a["router_proj_dim"],
    )

    # AE state dict: strip compile wrapper, keep only backbone.* keys.
    sd = {k.replace("_orig_mod.", "", 1): v for k, v in ckpt["model"].items()}
    backbone_sd = {k[len("backbone."):]: v for k, v in sd.items()
                   if k.startswith("backbone.")}
    missing, unexpected = model.load_state_dict(backbone_sd, strict=True)
    assert not missing and not unexpected
    print(f"Loaded backbone from {ckpt_path} "
          f"(AE epoch {ckpt['epoch'] + 1}, {len(backbone_sd)} tensors)")

    model.to(device)
    return model, a


def resolve_topk(args, ae_args: dict) -> int:
    """--topk unset → match the AE's keep_budget bottleneck; 0 → all."""
    if args.topk is not None:
        return args.topk
    keep_budget = ae_args.get("keep_budget", 0.0)
    if keep_budget <= 0:
        return 0
    n_patches = (ae_args["image_size"] // ae_args["patch_size"]) ** 2
    return max(1, round(n_patches * keep_budget))


def evaluate(model, loader, device):
    model.eval()
    losses, correct1, correct5, n = AverageMeter(), 0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(images)
                loss = F.cross_entropy(logits, labels)
            losses.update(loss.detach(), images.size(0))
            top5 = logits.topk(5, dim=1).indices
            correct1 += (top5[:, 0] == labels).sum()
            correct5 += (top5 == labels.unsqueeze(1)).any(dim=1).sum()
            n += images.size(0)
    return float(losses.avg), float(correct1) / n * 100, float(correct5) / n * 100


def main():
    parser = argparse.ArgumentParser(description="Frozen-backbone probe for ASFNetAE")
    parser.add_argument("--ae_artifact",   type=str, default=None,
                        help="wandb artifact ref, e.g. asfnetAE/asfnet-ae-<id>:final")
    parser.add_argument("--ae_checkpoint", type=str, default=None)
    parser.add_argument("--num_classes",   type=int, default=100)

    parser.add_argument("--pool", type=str, default="attentive",
                        choices=["attentive", "mean"])
    parser.add_argument("--topk", type=int, default=None,
                        help="probe only the top-K survivors by boundary "
                             "evidence; default matches the AE's keep_budget "
                             "bottleneck, 0 = all survivors")

    parser.add_argument("--batch_size",   type=int,   default=1024)
    parser.add_argument("--num_epochs",   type=int,   default=50)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)

    parser.add_argument("--dataset_name",      type=str, default="clane9/imagenet-100")
    parser.add_argument("--dataset_cache_dir", type=str, default="./data")
    parser.add_argument("--jpeg_cache_dir",    type=str, default="./jpeg_cache")
    parser.add_argument("--num_workers",       type=int, default=16)

    parser.add_argument("--checkpoint_dir", type=str, default=None,
                        help="default: runs/<run_name> (gitignored)")
    parser.add_argument("--wandb_project",  type=str, default="asfnet")
    parser.add_argument("--run_name",       type=str, default=None)
    parser.add_argument("--image_size",     type=int, default=224)  # dataloader
    parser.add_argument("--seed",           type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")

    auto_name = args.run_name is None
    if auto_name:
        src = args.ae_artifact or os.path.basename(args.ae_checkpoint or "")
        args.run_name = f"probe_{src.split('/')[-1].replace(':', '_')}"
    if args.checkpoint_dir is None:
        args.checkpoint_dir = os.path.join("runs", args.run_name)

    wandb.init(project=args.wandb_project, name=args.run_name,
               config=vars(args))

    backbone, ae_args = load_backbone(args, device)
    topk = resolve_topk(args, ae_args)
    if auto_name:
        wandb.run.name += f"_{args.pool}" + (f"_top{topk}" if topk else "_all")
    wandb.config.update({"ae_args": ae_args, "resolved_topk": topk})
    print(f"Probe: pool={args.pool}, topk={topk or 'all survivors'}")

    d_feat = backbone.norm.normalized_shape[0]   # final-stage width (works
    model = ProbeModel(backbone, d_feat, ae_args["num_heads"],  # for ladder too)
                       args.num_classes, args.pool, topk).to(device)
    model = torch.compile(model)

    train_loader, val_loader = get_dataloaders(args)

    head_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable params: {sum(p.numel() for p in head_params):,} (probe head only)")
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
                logits = model(images)
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
