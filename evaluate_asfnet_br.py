"""
evaluate_asfnet_br.py — evaluation + visualisation for the border-retention
models (ASFNetBR / ASFNetBR2) and the autoencoder (ASFNetAE).

The chunk-map instrument, adapted for retention:
  - patches are coloured by group ID (stage 1 for BR, stage 2 for BR2)
  - DROPPED patches are rendered dark grey — the new signal this
    architecture makes visible: exactly what the router chose to discard
  - thin white lines = stage-1 boundaries, thick = stage-2 (BR2 only)

Modes:
  (default)        full-val accuracy
  --grid           4×4 batch grid of retention chunk maps
  --visualize N    N single-image chunk maps
  --ae             checkpoint is an ASFNetAE: skip accuracy, save
                   original | reconstruction | retention panels instead

The forward pass runs under bfloat16 autocast for the same reason as
evaluate_asfnet.py: the 0.5 boundary threshold was calibrated under bf16
and float32 shifts borderline edges. Do not remove the autocast context.
"""

import os
import types
import argparse

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from dataset import get_dataloaders
from utils import AverageMeter, accuracy

from model_asfnet_br import (
    ASFNetBR, ASFNetBR2,
    token_boundary_evidence, border_keep_mask, compact_survivors,
    gpu_connected_components_masked, masked_edge_probs_to_token_weights,
)
from model_asfnet import gpu_connected_components
from model_asfnet_ae import ASFNetAE
from model_asfnet_ae2 import ASFNetAE2

# Reuse the existing viz primitives — evaluate_asfnet2 guards its main().
from evaluate_asfnet2 import (
    _denormalize, _colormap, _build_overlay,
    _draw_stage1_boundaries, _draw_stage2_boundaries,
)

_GRAY = np.array([0.35, 0.35, 0.35], dtype=np.float32)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _load_model(path: str, device: torch.device, ae: bool):
    ckpt = torch.load(path, map_location=device, weights_only=True)
    a = ckpt["args"]

    if ae and a.get("two_stage", False):
        model = ASFNetAE2(
            image_size          = a["image_size"],
            patch_size          = a["patch_size"],
            d_model             = a["d_model"],
            num_heads           = a["num_heads"],
            encoder1_blocks     = a["encoder_blocks"],
            encoder2_blocks     = a["encoder2_blocks"],
            main_blocks         = a["main_blocks"],
            mlp_ratio           = a["mlp_ratio"],
            target_group_size_1 = a["target_group_size"],
            target_group_size_2 = a["target_group_size_2"],
            router_proj_dim     = a["router_proj_dim"],
            decoder_d_model     = a["decoder_d_model"],
            decoder_blocks      = a["decoder_blocks"],
            decoder_heads       = a["decoder_heads"],
            norm_pix_loss       = not a.get("no_norm_pix", False),
        )
    elif ae:
        model = ASFNetAE(
            image_size        = a["image_size"],
            patch_size        = a["patch_size"],
            d_model           = a["d_model"],
            num_heads         = a["num_heads"],
            encoder_blocks    = a["encoder_blocks"],
            main_blocks       = a["main_blocks"],
            mlp_ratio         = a["mlp_ratio"],
            target_group_size = a["target_group_size"],
            router_proj_dim   = a["router_proj_dim"],
            decoder_d_model   = a["decoder_d_model"],
            decoder_blocks    = a["decoder_blocks"],
            decoder_heads     = a["decoder_heads"],
            norm_pix_loss     = not a.get("no_norm_pix", False),
            keep_budget       = a.get("keep_budget", 0.0),
            keep_ratio_target = a.get("keep_ratio_target", 0.0),
            xattn_slots       = a.get("xattn_slots", 0),
        )
    elif a.get("two_stage", False):
        model = ASFNetBR2(
            image_size          = a["image_size"],
            patch_size          = a["patch_size"],
            d_model             = a["d_model"],
            num_heads           = a["num_heads"],
            encoder1_blocks     = a["encoder_blocks"],
            encoder2_blocks     = a["encoder2_blocks"],
            main_blocks         = a["main_blocks"],
            mlp_ratio           = a["mlp_ratio"],
            num_classes         = a["num_classes"],
            target_group_size_1 = a["target_group_size"],
            target_group_size_2 = a["target_group_size_2"],
            router_proj_dim     = a["router_proj_dim"],
            weighted_merge2     = not a.get("uniform_merge2", False),
        )
    else:
        model = ASFNetBR(
            image_size        = a["image_size"],
            patch_size        = a["patch_size"],
            d_model           = a["d_model"],
            num_heads         = a["num_heads"],
            encoder_blocks    = a["encoder_blocks"],
            main_blocks       = a["main_blocks"],
            mlp_ratio         = a["mlp_ratio"],
            num_classes       = a["num_classes"],
            target_group_size = a["target_group_size"],
            router_proj_dim   = a["router_proj_dim"],
        )

    state_dict = {k.replace("_orig_mod.", "", 1): v for k, v in ckpt["model"].items()}
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print(f"Loaded checkpoint — epoch {ckpt['epoch'] + 1}")
    return model, a


# ---------------------------------------------------------------------------
# Intermediates (bf16 autocast — see module docstring)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _intermediates_br(model: ASFNetBR, images: torch.Tensor):
    """logits, group_ids1, keep, n_kept per image"""
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        tokens, coords = model.patch_embed(images)
        N = tokens.shape[1]
        for block in model.encoder:
            tokens = block(tokens, coords)

        hard, probs, _ = model.router(tokens, model.target_group_size)
        group_ids = gpu_connected_components(hard.detach(), model.router.edge_indices, N)

        s    = token_boundary_evidence(probs, model.router.edge_indices, N)
        keep = border_keep_mask(hard, model.router.edge_indices, N)
        keep = keep | (keep.sum(dim=1, keepdim=True) == 0)

        tokens = tokens + s.unsqueeze(-1) * tokens
        tok_c, coord_c, pad_mask, _, n_keep = compact_survivors(tokens, coords, keep)
        tok_c = model.stage_proj(tok_c)

        for block in model.main_net:
            tok_c = block(tok_c, coord_c, pad_mask)
        tok_c = model.norm(tok_c)

        real   = (~pad_mask).float()
        pooled = (tok_c * real.unsqueeze(-1)).sum(1) / real.sum(1, keepdim=True).clamp(min=1)
        logits = model.classifier(pooled)

    return logits, group_ids, keep, n_keep


@torch.no_grad()
def _intermediates_br2(model: ASFNetBR2, images: torch.Tensor):
    """logits, group_ids1 (viz), group_ids2, keep1, n_kept, n_groups2 per image"""
    B = images.shape[0]
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        tokens, coords = model.patch_embed(images)
        N = tokens.shape[1]
        for block in model.encoder1:
            tokens = block(tokens, coords)

        hard1, probs1, _ = model.router1(tokens, model.target_group_size_1)
        group_ids1 = gpu_connected_components(hard1.detach(), model.router1.edge_indices, N)

        s1    = token_boundary_evidence(probs1, model.router1.edge_indices, N)
        keep1 = border_keep_mask(hard1, model.router1.edge_indices, N)
        keep1 = keep1 | (keep1.sum(dim=1, keepdim=True) == 0)

        tokens = tokens + s1.unsqueeze(-1) * tokens
        tokens = model.stage1_proj(tokens)

        drop1 = ~keep1
        for block in model.encoder2:
            tokens = block(tokens, coords, drop1)

        hard2, probs2, valid2, _ = model.router2(tokens, keep1, model.target_group_size_2)
        group_ids2 = gpu_connected_components_masked(
            hard2.detach(), valid2, model.router2.edge_indices, N, keep1)

        token_weights2 = None
        if model.weighted_merge2:
            token_weights2 = masked_edge_probs_to_token_weights(
                probs2, hard2, valid2, model.router2.edge_indices, N, keep1)

        padded2, coords2, _, _ = model.merge2(tokens, coords, group_ids2,
                                              token_weights=token_weights2)

        max_G2   = padded2.shape[1]
        real_sum = torch.zeros(B, max_G2, device=images.device)
        real_sum.scatter_add_(1, group_ids2.clamp(0, max_G2 - 1), keep1.float())
        pad_mask2 = real_sum < 0.5

        for block in model.main_net:
            padded2 = block(padded2, coords2, pad_mask2)
        padded2 = model.norm(padded2)

        real   = (~pad_mask2).float()
        pooled = (padded2 * real.unsqueeze(-1)).sum(1) / real.sum(1, keepdim=True).clamp(min=1)
        logits = model.classifier(pooled)

        n_groups2 = (~pad_mask2).sum(dim=1)

    return logits, group_ids1, group_ids2, keep1, keep1.sum(dim=1), n_groups2


# ---------------------------------------------------------------------------
# Retention chunk map
# ---------------------------------------------------------------------------

def _retention_overlay(ids_np, keep_np, grid_size, patch_size):
    """Colour overlay by group ID; dropped patches dark grey."""
    n_ids   = int(ids_np.max()) + 1
    colors  = _colormap(n_ids)
    overlay = _build_overlay(ids_np, colors, grid_size, patch_size)
    for tok in range(grid_size * grid_size):
        if not keep_np[tok]:
            r, c = divmod(tok, grid_size)
            overlay[r*patch_size:(r+1)*patch_size,
                    c*patch_size:(c+1)*patch_size] = _GRAY
    return overlay


def _draw_panel(ax, img_np, ids_color_np, keep_np, ids1_np, ids2_np,
                grid_size, patch_size, alpha, title):
    overlay = _retention_overlay(ids_color_np, keep_np, grid_size, patch_size)
    ax.imshow(img_np)
    ax.imshow(overlay, alpha=alpha)
    _draw_stage1_boundaries(ax, ids1_np, grid_size, patch_size)
    if ids2_np is not None:
        _draw_stage2_boundaries(ax, ids2_np, grid_size, patch_size)
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def visualize_retention_grid(images, all_ids1, all_ids2, all_keep,
                             preds, labels, kept_counts, group_counts,
                             grid_size, patch_size, save_path, alpha=0.55,
                             two_stage=False):
    n = images.shape[0]
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = np.atleast_2d(axes)

    for i in range(nrows * ncols):
        ax = axes[i // ncols, i % ncols]
        if i >= n:
            ax.axis("off")
            continue
        img_np  = _denormalize(images[i])
        ids1_np = all_ids1[i].cpu().numpy()
        keep_np = all_keep[i].cpu().numpy()
        ids2_np = all_ids2[i].cpu().numpy() if two_stage else None
        ok      = preds[i].item() == labels[i].item()
        mark    = "\u2713" if ok else "\u2717"
        if two_stage:
            title = f"{mark}  kept={int(kept_counts[i])}  s2={int(group_counts[i])}"
            color_ids = ids2_np
        else:
            title = f"{mark}  kept={int(kept_counts[i])}  groups={int(group_counts[i])}"
            color_ids = ids1_np
        _draw_panel(ax, img_np, color_ids, keep_np, ids1_np, ids2_np,
                    grid_size, patch_size, alpha, title)

    plt.tight_layout()
    plt.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")


# ---------------------------------------------------------------------------
# AE reconstruction panels
# ---------------------------------------------------------------------------

@torch.no_grad()
def visualize_reconstructions(model: ASFNetAE, images, grid_size, patch_size,
                              save_path, n=8):
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        pred_imgs, keep = model.reconstruct(images[:n])
    pred_imgs = pred_imgs.float().cpu()

    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    axes = np.atleast_2d(axes)
    for i in range(n):
        img_np = _denormalize(images[i])
        axes[i, 0].imshow(img_np)
        axes[i, 0].set_title("original", fontsize=8)

        rec = pred_imgs[i].permute(1, 2, 0).numpy()
        rec = (rec - rec.min()) / (rec.max() - rec.min() + 1e-8)  # display scale
        axes[i, 1].imshow(rec)
        axes[i, 1].set_title("reconstruction (per-patch norm space)", fontsize=8)

        keep_np = keep[i].cpu().numpy()
        masked = img_np.copy()
        for tok in range(grid_size * grid_size):
            if not keep_np[tok]:
                r, c = divmod(tok, grid_size)
                masked[r*patch_size:(r+1)*patch_size,
                       c*patch_size:(c+1)*patch_size] = 0.35
        axes[i, 2].imshow(masked)
        axes[i, 2].set_title(f"retained ({int(keep_np.sum())} tokens)", fontsize=8)

        for j in range(3):
            axes[i, j].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {save_path}")


# ---------------------------------------------------------------------------
# Accuracy
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_accuracy(model, loader, device, two_stage):
    top1, top5, kept = AverageMeter(), AverageMeter(), AverageMeter()
    for images, labels in tqdm(loader, desc="Validation"):
        images = images.to(device)
        labels = labels.to(device)
        if two_stage:
            logits, _, _, _, n_kept, _ = _intermediates_br2(model, images)
        else:
            logits, _, _, n_kept = _intermediates_br(model, images)
        acc1, acc5 = accuracy(logits.float(), labels, topk=(1, 5))
        top1.update(acc1, images.size(0))
        top5.update(acc5, images.size(0))
        kept.update(n_kept.float().mean().item(), images.size(0))
    print(f"\nVal top-1: {top1.avg:.2f}%   top-5: {top5.avg:.2f}%   "
          f"mean kept tokens: {kept.avg:.1f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate border-retention ASFNet")
    parser.add_argument("--checkpoint",  type=str, required=True)
    parser.add_argument("--ae",          action="store_true",
                        help="Checkpoint is an ASFNetAE — visualise reconstructions")
    parser.add_argument("--grid",        action="store_true")
    parser.add_argument("--visualize",   type=int, default=0)
    parser.add_argument("--no_accuracy", action="store_true")
    parser.add_argument("--output_dir",  type=str, default="./viz_asfnet_br")
    parser.add_argument("--alpha",       type=float, default=0.55)
    parser.add_argument("--device",      type=str, default="cuda")

    parser.add_argument("--dataset_name",      type=str, default=None)
    parser.add_argument("--dataset_cache_dir", type=str, default=None)
    parser.add_argument("--jpeg_cache_dir",    type=str, default=None)
    parser.add_argument("--batch_size",        type=int, default=64)
    parser.add_argument("--num_workers",       type=int, default=4)
    parser.add_argument("--seed",              type=int, default=42)

    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model, ckpt_args = _load_model(args.checkpoint, device, ae=args.ae)
    two_stage = (not args.ae) and ckpt_args.get("two_stage", False)

    data_args = types.SimpleNamespace(
        image_size        = ckpt_args["image_size"],
        patch_size        = ckpt_args["patch_size"],
        dataset_name      = args.dataset_name      or ckpt_args["dataset_name"],
        dataset_cache_dir = args.dataset_cache_dir or ckpt_args["dataset_cache_dir"],
        jpeg_cache_dir    = args.jpeg_cache_dir    or ckpt_args["jpeg_cache_dir"],
        batch_size        = args.batch_size,
        num_workers       = args.num_workers,
        seed              = args.seed,
    )
    _, val_loader = get_dataloaders(data_args)

    grid_size  = ckpt_args["image_size"] // ckpt_args["patch_size"]
    patch_size = ckpt_args["patch_size"]

    os.makedirs(args.output_dir, exist_ok=True)
    images, labels = next(iter(val_loader))
    images, labels = images.to(device), labels.to(device)

    if args.ae:
        visualize_reconstructions(
            model, images, grid_size, patch_size,
            os.path.join(args.output_dir, "reconstructions.png"))
        return

    if not args.no_accuracy:
        run_accuracy(model, val_loader, device, two_stage)

    if args.grid or args.visualize > 0:
        if two_stage:
            logits, ids1, ids2, keep, n_kept, n_g2 = _intermediates_br2(model, images)
            group_counts = n_g2
        else:
            logits, ids1, keep, n_kept = _intermediates_br(model, images)
            ids2 = ids1  # unused
            group_counts = ids1.max(dim=1).values + 1
        preds = logits.argmax(dim=1)

        if args.grid:
            n_grid = min(16, images.size(0))
            visualize_retention_grid(
                images[:n_grid].cpu(), ids1[:n_grid], ids2[:n_grid], keep[:n_grid],
                preds[:n_grid], labels[:n_grid],
                n_kept[:n_grid].cpu(), group_counts[:n_grid].cpu(),
                grid_size, patch_size,
                os.path.join(args.output_dir, "retention_grid.png"),
                alpha=args.alpha, two_stage=two_stage)

        for idx in range(min(args.visualize, images.size(0))):
            fig, ax = plt.subplots(figsize=(6, 6))
            ok   = preds[idx].item() == labels[idx].item()
            mark = "\u2713" if ok else "\u2717"
            if two_stage:
                title = f"{mark}  kept={int(n_kept[idx])}  s2={int(group_counts[idx])}"
                color_ids = ids2[idx].cpu().numpy()
                ids2_np   = ids2[idx].cpu().numpy()
            else:
                title = f"{mark}  kept={int(n_kept[idx])}  groups={int(group_counts[idx])}"
                color_ids = ids1[idx].cpu().numpy()
                ids2_np   = None
            _draw_panel(ax, _denormalize(images[idx].cpu()), color_ids,
                        keep[idx].cpu().numpy(), ids1[idx].cpu().numpy(), ids2_np,
                        grid_size, patch_size, args.alpha, title)
            out = os.path.join(args.output_dir, f"retention_{idx:04d}.png")
            plt.savefig(out, dpi=140, bbox_inches="tight")
            plt.close(fig)
            print(f"  saved {out}")


if __name__ == "__main__":
    main()
