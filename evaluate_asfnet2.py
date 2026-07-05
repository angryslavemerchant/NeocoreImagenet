"""
evaluate_asfnet2.py — ASFNet2 evaluation and two-stage group visualisation.

Both stages are rendered at the original patch resolution so the hierarchy
is directly comparable in a single image. For each patch in the grid:
  - Stage 1 group: which Stage 1 group does this patch belong to?
  - Stage 2 group: which Stage 2 group does this patch belong to?

Stage 2 assignment is traced back through both sets of group IDs:
    stage2_at_patch[i] = group_ids2[ group_ids1[i] ]

Boundary lines use two visual weights to show the two levels:
  - Thin dim line  → Stage 1 boundary (adjacent patches in different S1 groups)
  - Thick white line → Stage 2 boundary (adjacent patches in different S2 groups)

Output per image: 3-panel figure (original | Stage 1 | Stage 2).
Output for batch: compact grid showing Stage 2 only (most compressed/interesting).
"""

import os
import argparse
import types

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from tqdm import tqdm

from dataset import get_dataloaders, IMAGENET_MEAN, IMAGENET_STD
from model_asfnet2 import (
    ASFNet2,
    build_knn_edges,
    knn_edges_to_mask,
    gpu_connected_components_dynamic,
)
from model_asfnet import gpu_connected_components
from utils import AverageMeter, accuracy


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_model(path: str, device: torch.device) -> tuple[ASFNet2, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=True)
    a    = ckpt["args"]

    model = ASFNet2(
        image_size          = a["image_size"],
        patch_size          = a["patch_size"],
        in_channels         = 3,
        d_model             = a["d_model"],
        num_heads           = a["num_heads"],
        encoder1_blocks     = a["encoder1_blocks"],
        encoder2_blocks     = a["encoder2_blocks"],
        main_blocks         = a["main_blocks"],
        mlp_ratio           = a["mlp_ratio"],
        num_classes         = a["num_classes"],
        target_group_size_1 = a["target_group_size_1"],
        target_group_size_2 = a["target_group_size_2"],
        router_proj_dim     = a["router_proj_dim"],
        knn_k               = a["knn_k"],
        local_encoder1      = a.get("local_encoder1", False),
        local_radius        = a.get("local_radius", 1),
        local_encoder2      = a.get("local_encoder2", False),
        local_encoder2_safe = a.get("local_encoder2_safe", False),
    )
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    print(f"Loaded  epoch {ckpt['epoch'] + 1}  "
          f"val top-1 {ckpt.get('val_top1', '?'):.2f}%  "
          f"(best {ckpt.get('best_top1', '?'):.2f}%)")
    print(f"  patch_size={a['patch_size']}  d_model={a['d_model']}  "
          f"target_group_size {a['target_group_size_1']} → {a['target_group_size_2']}  "
          f"knn_k={a['knn_k']}\n")
    return model, a


# ---------------------------------------------------------------------------
# Forward pass that captures both stages of intermediate outputs
# ---------------------------------------------------------------------------

def _get_intermediate(
    model:  ASFNet2,
    images: torch.Tensor,
) -> dict:
    """
    Run ASFNet2 manually submodule-by-submodule to capture every intermediate
    tensor needed for visualisation, without modifying model_asfnet2.py.

    Returns a dict with:
        logits          (B, num_classes)
        group_ids1      (B, N1)      — patch → Stage 1 group
        hard1           (B, E1)      — Stage 1 boundary decisions
        group_ids2      (B, max_G1)  — Stage 1 group → Stage 2 group
        stage2_at_patch (B, N1)      — patch → Stage 2 group  (traced back)
        pad_mask2       (B, max_G2)
        mean_groups1    float
        mean_groups2    float        — corrected (excludes leaked padding groups)
    """
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        B = images.shape[0]

        # ---- Stage 1 ----
        tokens, coords = model.patch_embed(images)
        for block in model.encoder1:
            tokens = block(tokens, coords)

        hard1, _, _ = model.router1(tokens, model.target_group_size_1)
        group_ids1  = gpu_connected_components(
            hard1.detach(), model.router1.edge_indices, tokens.shape[1]
        )
        padded_tokens1, padded_coords1, pad_mask1, mean_groups1 = model.merge1(
            tokens, coords, group_ids1
        )

        # ---- Stage 2 ----
        # k-NN adjacency depends only on Stage 1 centroids + padding, so build it
        # before encoder2 and reuse it for both the local-attention mask and the
        # router (mirrors ASFNet2.forward). Reordering is behaviour-identical in
        # the global-encoder2 case.
        n_real1       = (~pad_mask1).sum(dim=1)
        src2, dst2, valid2 = build_knn_edges(padded_coords1, pad_mask1, model.knn_k)
        max_G1        = padded_tokens1.shape[1]

        if model.local_encoder2:
            adj2 = knn_edges_to_mask(src2, dst2, valid2, max_G1)
            for block in model.encoder2:
                padded_tokens1 = block(padded_tokens1, padded_coords1, adj2)
        else:
            for block in model.encoder2:
                padded_tokens1 = block(padded_tokens1, padded_coords1, pad_mask1)

        hard2, _, _   = model.router2(
            padded_tokens1, src2, dst2, valid2, model.target_group_size_2, n_real1
        )
        group_ids2    = gpu_connected_components_dynamic(
            hard2.detach(), valid2, src2, dst2, max_G1
        )
        padded_tokens2, padded_coords2, _, _ = model.merge2(
            padded_tokens1, padded_coords1, group_ids2
        )

        # Corrected Stage 2 padding mask — same logic as ASFNet2.forward()
        max_G2        = padded_tokens2.shape[1]
        real1         = (~pad_mask1).float()
        real2_sum     = torch.zeros(B, max_G2, device=images.device)
        real2_sum.scatter_add_(1, group_ids2.clamp(0, max_G2 - 1), real1)
        pad_mask2     = real2_sum < 0.5
        mean_groups2  = float((~pad_mask2).sum(dim=1).float().mean().item())

        # Trace Stage 2 group back to patch level.
        # group_ids1:   (B, N1)      patch i → Stage 1 group id
        # group_ids2:   (B, max_G1)  Stage 1 group id → Stage 2 group id
        # Clamping is a safety measure; group_ids1 values are in [0, max_G1)
        # by construction so no clamping should actually trigger.
        stage2_at_patch = group_ids2.gather(1, group_ids1.clamp(0, max_G1 - 1))  # (B, N1)

        # ---- Main network ----
        for block in model.main_net:
            padded_tokens2 = block(padded_tokens2, padded_coords2, pad_mask2)
        padded_tokens2 = model.norm(padded_tokens2)
        real_mask   = (~pad_mask2).float()
        token_sum   = (padded_tokens2 * real_mask.unsqueeze(-1)).sum(dim=1)
        token_count = real_mask.sum(dim=1, keepdim=True).clamp(min=1)
        logits      = model.classifier(token_sum / token_count)

    return dict(
        logits          = logits,
        group_ids1      = group_ids1,
        hard1           = hard1,
        group_ids2      = group_ids2,
        stage2_at_patch = stage2_at_patch,
        pad_mask2       = pad_mask2,
        mean_groups1    = mean_groups1,
        mean_groups2    = mean_groups2,
    )


# ---------------------------------------------------------------------------
# Shared rendering helpers
# ---------------------------------------------------------------------------

def _denormalize(t: torch.Tensor) -> np.ndarray:
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std  = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (t.cpu().float() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()


def _colormap(n: int, seed: int = 0) -> np.ndarray:
    """
    (n, 3) RGB array — unique colour per group.
    Hues are shuffled so spatially adjacent groups (which tend to get
    sequential IDs from the connected-components labelling) don't end up
    with similar colours.
    """
    rng  = np.random.RandomState(seed)
    hues = np.linspace(0, 1, n, endpoint=False)
    rng.shuffle(hues)
    hsv  = np.stack([hues, np.full(n, 0.65), np.full(n, 0.90)], axis=1)
    return mcolors.hsv_to_rgb(hsv)


def _build_overlay(ids_np: np.ndarray, colors: np.ndarray,
                   grid_size: int, patch_size: int) -> np.ndarray:
    """Fill a (H, W, 3) colour array one patch cell at a time."""
    H = W = grid_size * patch_size
    overlay = np.zeros((H, W, 3), dtype=np.float32)
    for tok in range(grid_size * grid_size):
        r, c = divmod(tok, grid_size)
        overlay[r*patch_size:(r+1)*patch_size, c*patch_size:(c+1)*patch_size] = colors[ids_np[tok]]
    return overlay


def _draw_stage1_boundaries(ax, ids_np: np.ndarray, grid_size: int, patch_size: int):
    """
    Thin, semi-transparent lines where adjacent patches belong to
    different Stage 1 groups — the fine-grained level of the hierarchy.
    """
    for tok in range(grid_size * grid_size):
        r, c = divmod(tok, grid_size)
        # Right neighbour
        if c + 1 < grid_size:
            nbr = tok + 1
            if ids_np[tok] != ids_np[nbr]:
                x_px = (c + 1) * patch_size
                ax.plot([x_px, x_px], [r * patch_size, (r+1) * patch_size],
                        color="white", lw=0.5, alpha=0.5)
        # Down neighbour
        if r + 1 < grid_size:
            nbr = tok + grid_size
            if ids_np[tok] != ids_np[nbr]:
                y_px = (r + 1) * patch_size
                ax.plot([c * patch_size, (c+1) * patch_size], [y_px, y_px],
                        color="white", lw=0.5, alpha=0.5)


def _draw_stage2_boundaries(ax, ids_np: np.ndarray, grid_size: int, patch_size: int):
    """
    Thick white lines where adjacent patches belong to different Stage 2
    groups — the coarse level of the hierarchy. Drawn on top of Stage 1
    lines so the two levels are visually distinguishable.
    """
    for tok in range(grid_size * grid_size):
        r, c = divmod(tok, grid_size)
        if c + 1 < grid_size:
            nbr = tok + 1
            if ids_np[tok] != ids_np[nbr]:
                x_px = (c + 1) * patch_size
                ax.plot([x_px, x_px], [r * patch_size, (r+1) * patch_size],
                        color="white", lw=1.5, alpha=0.95)
        if r + 1 < grid_size:
            nbr = tok + grid_size
            if ids_np[tok] != ids_np[nbr]:
                y_px = (r + 1) * patch_size
                ax.plot([c * patch_size, (c+1) * patch_size], [y_px, y_px],
                        color="white", lw=1.5, alpha=0.95)


# ---------------------------------------------------------------------------
# Single-image three-panel visualisation
# ---------------------------------------------------------------------------

def visualize_two_stages(
    image_tensor:    torch.Tensor,   # (C, H, W) normalised
    group_ids1_1d:   torch.Tensor,   # (N,)  patch → Stage 1 group
    stage2_patch_1d: torch.Tensor,   # (N,)  patch → Stage 2 group
    grid_size:       int,
    patch_size:      int,
    n_groups1:       int,
    n_groups2:       int,
    title:           str  = "",
    save_path:       str  = None,
    alpha:           float = 0.55,
):
    """
    Three-panel figure: original | Stage 1 grouping | Stage 2 grouping.

    Both grouping panels are rendered at the original patch resolution.
    Stage 1 shows the fine-grained cuts; Stage 2 shows the coarser result
    after the second merge. Boundary lines on the Stage 2 panel use two
    weights to show both hierarchy levels simultaneously:
      thin dim line   → Stage 1 boundary within a Stage 2 group
      thick white line → Stage 2 boundary
    """
    img       = _denormalize(image_tensor)
    ids1_np   = group_ids1_1d.cpu().numpy()
    ids2_np   = stage2_patch_1d.cpu().numpy()

    colors1   = _colormap(n_groups1, seed=1)
    colors2   = _colormap(n_groups2, seed=2)

    ov1 = _build_overlay(ids1_np, colors1, grid_size, patch_size)
    ov2 = _build_overlay(ids2_np, colors2, grid_size, patch_size)

    comp1 = ((1 - alpha) * img + alpha * ov1).clip(0, 1)
    comp2 = ((1 - alpha) * img + alpha * ov2).clip(0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    axes[0].imshow(img)
    axes[0].set_title("Original", fontsize=9)
    axes[0].axis("off")

    axes[1].imshow(comp1)
    axes[1].set_title(f"Stage 1  (n={n_groups1})", fontsize=9)
    _draw_stage1_boundaries(axes[1], ids1_np, grid_size, patch_size)
    axes[1].axis("off")

    axes[2].imshow(comp2)
    axes[2].set_title(f"Stage 2  (n={n_groups2})", fontsize=9)
    # Draw Stage 1 boundaries first (thin), then Stage 2 on top (thick)
    _draw_stage1_boundaries(axes[2], ids1_np, grid_size, patch_size)
    _draw_stage2_boundaries(axes[2], ids2_np, grid_size, patch_size)
    axes[2].axis("off")

    if title:
        fig.suptitle(title, fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close()
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Compact batch grid (Stage 2 view only — most informative at a glance)
# ---------------------------------------------------------------------------

def visualize_grid(
    images:          torch.Tensor,   # (B, C, H, W)
    group_ids1:      torch.Tensor,   # (B, N)
    stage2_at_patch: torch.Tensor,   # (B, N)
    preds:           torch.Tensor,   # (B,)
    labels:          torch.Tensor,   # (B,)
    grid_size:       int,
    patch_size:      int,
    save_path:       str,
    n_cols:          int   = 4,
    alpha:           float = 0.55,
):
    """
    Compact grid — each cell shows the Stage 2 grouping overlay with both
    levels of boundary lines. Correct predictions are titled green, wrong red.
    """
    B      = images.size(0)
    n_rows = (B + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 3.5 * n_rows))
    axes = np.array(axes).flatten()

    for idx in range(B):
        ax       = axes[idx]
        img      = _denormalize(images[idx])
        ids1_np  = group_ids1[idx].cpu().numpy()
        ids2_np  = stage2_at_patch[idx].cpu().numpy()

        n_g2    = int(ids2_np.max()) + 1
        colors2 = _colormap(n_g2, seed=2)
        ov2     = _build_overlay(ids2_np, colors2, grid_size, patch_size)
        comp    = ((1 - alpha) * img + alpha * ov2).clip(0, 1)

        ax.imshow(comp)
        _draw_stage1_boundaries(ax, ids1_np, grid_size, patch_size)
        _draw_stage2_boundaries(ax, ids2_np, grid_size, patch_size)

        correct = preds[idx].item() == labels[idx].item()
        n_g1    = int(ids1_np.max()) + 1
        ax.set_title(
            f"{'✓' if correct else '✗'}  s1={n_g1}  s2={n_g2}",
            fontsize=7,
            color="green" if correct else "red",
        )
        ax.axis("off")

    for idx in range(B, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved grid → {save_path}")


# ---------------------------------------------------------------------------
# Full validation accuracy
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_accuracy(model: ASFNet2, loader, device: torch.device):
    top1 = AverageMeter()
    top5 = AverageMeter()

    for images, labels in tqdm(loader, desc="Evaluating"):
        images = images.to(device)
        labels = labels.to(device)
        out    = _get_intermediate(model, images)
        acc1, acc5 = accuracy(out["logits"], labels, topk=(1, 5))
        top1.update(acc1, images.size(0))
        top5.update(acc5, images.size(0))

    print(f"\nTop-1: {top1.avg:.2f}%  |  Top-5: {top5.avg:.2f}%")
    return top1.avg, top5.avg


# ---------------------------------------------------------------------------
# Accuracy binned by Stage 2 group count
# ---------------------------------------------------------------------------

@torch.no_grad()
def accuracy_by_group_count(
    model:    ASFNet2,
    loader,
    device:   torch.device,
    n_bins:   int = 6,
):
    """
    Tests the hypothesis that images the model compresses well are also the
    images it classifies well.

    For every validation image, records two numbers:
      - its Stage 2 group count (how many groups survived compression —
        fewer means the router merged more aggressively, which happens when
        the image has large uniform regions)
      - whether the top-1 prediction was correct

    Then splits the images into equal-width bins by group count and reports
    accuracy within each bin. If accuracy falls as the group count rises,
    that confirms busy / hard-to-compress images are also the hard-to-classify
    ones, meaning the model is limited by its ability to represent hard images
    rather than by the grouping mechanism itself.

    Args:
        n_bins: how many group-count buckets to split the range into.
    """
    counts    = []   # per-image Stage 2 group count
    correct   = []   # per-image 1 if top-1 correct else 0

    for images, labels in tqdm(loader, desc="Binning by group count"):
        images = images.to(device)
        labels = labels.to(device)

        out = _get_intermediate(model, images)

        # Per-image real (non-padding) Stage 2 group count
        per_image_counts = (~out["pad_mask2"]).sum(dim=1)          # (B,)
        preds            = out["logits"].argmax(dim=1)             # (B,)
        is_correct       = (preds == labels).long()               # (B,)

        counts.append(per_image_counts.cpu())
        correct.append(is_correct.cpu())

    counts  = torch.cat(counts).numpy()
    correct = torch.cat(correct).numpy()

    overall = 100.0 * correct.mean()
    lo, hi  = int(counts.min()), int(counts.max())

    print(f"\nOverall top-1: {overall:.2f}%  "
          f"({len(counts)} images, group count range {lo}–{hi})")
    print(f"\n{'group count':>14} | {'images':>7} | {'accuracy':>9} | {'mean groups':>11}")
    print("-" * 52)

    # Equal-width bins across the observed group-count range
    edges = np.linspace(lo, hi + 1e-6, n_bins + 1)
    for b in range(n_bins):
        in_bin = (counts >= edges[b]) & (counts < edges[b + 1])
        n      = int(in_bin.sum())
        if n == 0:
            print(f"{int(edges[b]):>6}–{int(edges[b+1]):<6} |"
                  f" {0:>7} | {'—':>9} | {'—':>11}")
            continue
        acc      = 100.0 * correct[in_bin].mean()
        mean_grp = counts[in_bin].mean()
        print(f"{int(edges[b]):>6}–{int(edges[b+1]):<6} |"
              f" {n:>7} | {acc:>8.2f}% | {mean_grp:>11.1f}")

    # Simple correlation summary: split at the median and compare halves.
    # A clear gap here is the headline result.
    median = np.median(counts)
    low_half  = counts <= median
    high_half = counts >  median
    if low_half.sum() > 0 and high_half.sum() > 0:
        acc_low  = 100.0 * correct[low_half].mean()
        acc_high = 100.0 * correct[high_half].mean()
        print("-" * 52)
        print(f"Fewer groups than median ({median:.0f}): {acc_low:.2f}%  "
              f"({int(low_half.sum())} imgs)")
        print(f"More groups than median  ({median:.0f}): {acc_high:.2f}%  "
              f"({int(high_half.sum())} imgs)")
        print(f"Gap: {acc_low - acc_high:+.2f} percentage points "
              f"(positive = cleanly-compressed images classify better)")

    return counts, correct


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate and visualise ASFNet2 (two-stage)")

    parser.add_argument("--checkpoint",  type=str, required=True)
    parser.add_argument("--visualize",   type=int, default=0,
                        help="Number of individual 3-panel images to save.")
    parser.add_argument("--grid",        action="store_true",
                        help="Save a compact batch grid (Stage 2 view).")
    parser.add_argument("--no_accuracy", action="store_true")
    parser.add_argument("--bin_by_groups", action="store_true",
                        help="Run the full validation set and report accuracy binned by "
                             "Stage 2 group count. Tests whether cleanly-compressed images "
                             "also classify better.")
    parser.add_argument("--n_bins",      type=int, default=6,
                        help="Number of group-count buckets for --bin_by_groups.")
    parser.add_argument("--output_dir",  type=str, default="./viz_asfnet2")
    parser.add_argument("--alpha",       type=float, default=0.55)
    parser.add_argument("--device",      type=str, default="cuda")

    parser.add_argument("--dataset_name",      type=str, default=None)
    parser.add_argument("--dataset_cache_dir", type=str, default=None)
    parser.add_argument("--jpeg_cache_dir",    type=str, default=None)
    parser.add_argument("--batch_size",        type=int, default=32)
    parser.add_argument("--num_workers",       type=int, default=4)
    parser.add_argument("--seed",              type=int, default=42)

    args   = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model, ckpt_args = _load_model(args.checkpoint, device)

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

    if not args.no_accuracy:
        run_accuracy(model, val_loader, device)

    if args.bin_by_groups:
        accuracy_by_group_count(model, val_loader, device, n_bins=args.n_bins)

    if args.visualize > 0 or args.grid:
        os.makedirs(args.output_dir, exist_ok=True)

        images, labels = next(iter(val_loader))
        images = images.to(device)
        labels = labels.to(device)

        out   = _get_intermediate(model, images)
        preds = out["logits"].argmax(dim=1)

        print(f"\nFirst batch — mean Stage 1 groups: {out['mean_groups1']:.1f}  "
              f"Stage 2 groups: {out['mean_groups2']:.1f}")

        if args.grid:
            visualize_grid(
                images          = images[:16].cpu(),
                group_ids1      = out["group_ids1"][:16],
                stage2_at_patch = out["stage2_at_patch"][:16],
                preds           = preds[:16],
                labels          = labels[:16],
                grid_size       = grid_size,
                patch_size      = patch_size,
                save_path       = os.path.join(args.output_dir, "group_grid.png"),
                alpha           = args.alpha,
            )

        n_vis = min(args.visualize, images.size(0))
        for idx in range(n_vis):
            ids1     = out["group_ids1"][idx]
            ids2_pat = out["stage2_at_patch"][idx]
            n_g1     = int(ids1.max().item()) + 1
            n_g2     = int(ids2_pat.max().item()) + 1
            correct  = preds[idx].item() == labels[idx].item()
            title    = (f"{'✓' if correct else '✗'}  "
                        f"pred={preds[idx].item()}  true={labels[idx].item()}  "
                        f"s1={n_g1}  s2={n_g2}")
            visualize_two_stages(
                image_tensor    = images[idx].cpu(),
                group_ids1_1d   = ids1,
                stage2_patch_1d = ids2_pat,
                grid_size       = grid_size,
                patch_size      = patch_size,
                n_groups1       = n_g1,
                n_groups2       = n_g2,
                title           = title,
                save_path       = os.path.join(args.output_dir, f"groups_{idx:04d}.png"),
                alpha           = args.alpha,
            )
            print(f"  [{idx+1}/{n_vis}] {title}")

        print(f"\nSaved to {args.output_dir}/")


if __name__ == "__main__":
    main()