"""
evaluate_asfnet.py — ASFNet evaluation and group visualisation

Two modes:
  --visualize N   Save N side-by-side images showing how the router
                  chunked the image into groups. Each patch cell is
                  coloured by its group ID; boundary edges are drawn
                  as white lines. Quick way to sanity-check that the
                  router is doing something sensible.

  (no flag)       Just run accuracy over the full validation set and
                  print top-1 / top-5.

Both modes can be combined.
"""

import os
import argparse

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import numpy as np
from tqdm import tqdm

from dataset import get_dataloaders, IMAGENET_MEAN, IMAGENET_STD
from model_asfnet import ASFNet, gpu_connected_components
from utils import AverageMeter, accuracy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_model_from_checkpoint(path: str, device: torch.device) -> tuple[ASFNet, dict]:
    """
    Load an ASFNet checkpoint. The checkpoint stores the training args as a
    plain dict (not a Config dataclass), so weights_only=True is safe.
    """
    ckpt = torch.load(path, map_location=device, weights_only=True)
    args = ckpt["args"]

    model = ASFNet(
        image_size        = args["image_size"],
        patch_size        = args["patch_size"],
        in_channels       = 3,
        d_model           = args["d_model"],
        num_heads         = args["num_heads"],
        encoder_blocks    = args["encoder_blocks"],
        main_blocks       = args["main_blocks"],
        mlp_ratio         = args["mlp_ratio"],
        num_classes       = args["num_classes"],
        target_group_size = args["target_group_size"],
        router_proj_dim   = args["router_proj_dim"],
    )
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    print(f"Loaded checkpoint — epoch {ckpt['epoch'] + 1}, "
          f"val top-1: {ckpt.get('val_top1', '?'):.2f}%  "
          f"(best: {ckpt.get('best_top1', '?'):.2f}%)")
    print(f"  patch_size={args['patch_size']}  "
          f"d_model={args['d_model']}  "
          f"target_group_size={args['target_group_size']}\n")
    return model, args


def _get_intermediate(
    model: ASFNet,
    images: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Run the model up through the grouping step and return everything needed
    for visualisation, without touching model_asfnet.py.

    Calls each submodule manually in the same order as ASFNet.forward(), so
    this is guaranteed to stay in sync as long as the architecture doesn't change.

    Returns:
        logits:     (B, num_classes)
        group_ids:  (B, N) int64   — contiguous group IDs, same as inside forward()
        hard:       (B, E) float   — 1 = boundary edge, 0 = connected
        tokens:     (B, N, D)      — encoder output (pre-merge), for diagnostics
        mean_groups: float
    """
    with torch.no_grad():
        tokens, coords = model.patch_embed(images)

        for block in model.encoder:
            tokens = block(tokens, coords)

        hard, probs, l_ratio = model.router(tokens, model.target_group_size)

        group_ids = gpu_connected_components(
            hard.detach(),
            model.router.edge_indices,
            tokens.shape[1],
        )

        padded_tokens, padded_coords, pad_mask, mean_groups = model.group_merge(
            tokens, coords, group_ids
        )

        for block in model.main_net:
            padded_tokens = block(padded_tokens, padded_coords, pad_mask)

        padded_tokens = model.norm(padded_tokens)
        real_mask   = (~pad_mask).float()
        token_sum   = (padded_tokens * real_mask.unsqueeze(-1)).sum(dim=1)
        token_count = real_mask.sum(dim=1, keepdim=True).clamp(min=1)
        pooled      = token_sum / token_count
        logits      = model.classifier(pooled)

    return logits, group_ids, hard, tokens, mean_groups


def _denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Invert ImageNet normalisation → HxWxC float32 in [0, 1]."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std  = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img  = (tensor.cpu().float() * std + mean).clamp(0, 1)
    return img.permute(1, 2, 0).numpy()


def _group_colormap(n_groups: int) -> np.ndarray:
    """
    Return an (n_groups, 3) RGB array of visually distinct colours.

    Samples from HSV (Hue-Saturation-Value — a cylindrical colour space
    where hue encodes the base colour, saturation its purity, and value
    its brightness) with shuffled hue so adjacent group IDs (which tend
    to be spatially nearby because connected components are labelled by
    their minimum token index) don't end up with similar colours.
    """
    rng = np.random.RandomState(0)
    hues = np.linspace(0, 1, n_groups, endpoint=False)
    rng.shuffle(hues)
    hsv = np.stack([
        hues,
        np.full(n_groups, 0.65),   # moderate saturation — not too garish
        np.full(n_groups, 0.90),   # bright but not white
    ], axis=1)                     # (n_groups, 3)
    return mcolors.hsv_to_rgb(hsv) # (n_groups, 3)


# ---------------------------------------------------------------------------
# Single-image group visualisation
# ---------------------------------------------------------------------------

def visualize_groups(
    image_tensor: torch.Tensor,      # (C, H, W) normalised
    group_ids_1d: torch.Tensor,      # (N,) for one image
    hard_1d: torch.Tensor,           # (E,) for one image
    edge_indices: torch.Tensor,      # (E, 2) from router buffer
    grid_size: int,
    patch_size: int,
    n_groups: int,
    title: str = "",
    save_path: str = None,
    alpha: float = 0.55,
):
    """
    Draw the original image with a semi-transparent group colouring overlay.

    Each patch cell is filled with a colour unique to its group. White lines
    are drawn along boundary edges (where the router decided hard=1). The
    overlay sits on top of the actual image so you can still see content.

    Args:
        image_tensor:  (C, H, W) normalised tensor for a single image
        group_ids_1d:  (N,) group IDs in row-major patch order
        hard_1d:       (E,) hard boundary decisions for this image
        edge_indices:  (E, 2) from model.router.edge_indices
        grid_size:     number of patches along each side (e.g. 14 for p16)
        patch_size:    patch side length in pixels
        n_groups:      number of distinct groups in this image
        title:         plot title
        save_path:     if provided, save here; otherwise display
        alpha:         opacity of the colour overlay (0=invisible, 1=solid)
    """
    img = _denormalize(image_tensor)   # (H, W, 3)
    H = W = grid_size * patch_size     # should equal image_size

    # Build a per-pixel colour overlay by filling each patch cell
    colors  = _group_colormap(n_groups)
    overlay = np.zeros((H, W, 3), dtype=np.float32)

    ids = group_ids_1d.cpu().numpy()   # (N,) — row-major
    for token_idx in range(grid_size * grid_size):
        row = token_idx // grid_size
        col = token_idx  % grid_size
        r0, r1 = row * patch_size, (row + 1) * patch_size
        c0, c1 = col * patch_size, (col + 1) * patch_size
        gid = int(ids[token_idx])
        overlay[r0:r1, c0:c1] = colors[gid]

    # Composite: original image + semi-transparent colour layer
    composite = (1 - alpha) * img + alpha * overlay
    composite = composite.clip(0, 1)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(img)
    axes[0].set_title("Original", fontsize=9)
    axes[0].axis("off")

    axes[1].imshow(composite)
    axes[1].set_title(f"Groups  (n={n_groups})", fontsize=9)
    axes[1].axis("off")

    # Draw boundary edges as white lines on the group panel
    hard_np = hard_1d.cpu().numpy()
    ei      = edge_indices.cpu().numpy()   # (E, 2)
    for e_idx, (i, j) in enumerate(ei):
        if hard_np[e_idx] < 0.5:
            continue   # connected — no line
        ri, ci = divmod(int(i), grid_size)
        rj, cj = divmod(int(j), grid_size)

        # Determine which shared edge to draw
        if ci == cj:   # vertical edge — i and j are in the same column, different rows
            y_px = max(ri, rj) * patch_size
            x0   = ci * patch_size
            axes[1].plot([x0, x0 + patch_size], [y_px, y_px],
                         color="white", lw=0.6, alpha=0.9)
        else:          # horizontal edge — same row, different columns
            x_px = max(ci, cj) * patch_size
            y0   = ri * patch_size
            axes[1].plot([x_px, x_px], [y0, y0 + patch_size],
                         color="white", lw=0.6, alpha=0.9)

    if title:
        fig.suptitle(title, fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close()
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Grid of many images (compact summary view)
# ---------------------------------------------------------------------------

def visualize_group_grid(
    images: torch.Tensor,            # (B, C, H, W) for a batch
    all_group_ids: torch.Tensor,     # (B, N)
    all_hard: torch.Tensor,          # (B, E)
    edge_indices: torch.Tensor,      # (E, 2)
    grid_size: int,
    patch_size: int,
    preds: torch.Tensor,             # (B,)
    labels: torch.Tensor,            # (B,)
    save_path: str,
    n_cols: int = 4,
    alpha: float = 0.55,
):
    """
    Compact grid showing group overlays for a whole batch.
    Correct predictions are titled in green, wrong ones in red.
    """
    B       = images.size(0)
    n_rows  = (B + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 3.5 * n_rows))
    axes = np.array(axes).flatten()

    for idx in range(B):
        ax      = axes[idx]
        img     = _denormalize(images[idx])
        ids     = all_group_ids[idx]
        n_grps  = int(ids.max().item()) + 1
        colors  = _group_colormap(n_grps)

        H = W = grid_size * patch_size
        overlay = np.zeros((H, W, 3), dtype=np.float32)
        ids_np  = ids.cpu().numpy()

        for tok in range(grid_size * grid_size):
            r, c = divmod(tok, grid_size)
            r0, r1 = r * patch_size, (r + 1) * patch_size
            c0, c1 = c * patch_size, (c + 1) * patch_size
            overlay[r0:r1, c0:c1] = colors[int(ids_np[tok])]

        composite = ((1 - alpha) * img + alpha * overlay).clip(0, 1)
        ax.imshow(composite)

        # Boundary lines
        hard_np = all_hard[idx].cpu().numpy()
        ei      = edge_indices.cpu().numpy()
        for e_idx, (i, j) in enumerate(ei):
            if hard_np[e_idx] < 0.5:
                continue
            ri, ci = divmod(int(i), grid_size)
            rj, cj = divmod(int(j), grid_size)
            if ci == cj:
                y_px = max(ri, rj) * patch_size
                x0   = ci * patch_size
                ax.plot([x0, x0 + patch_size], [y_px, y_px],
                        color="white", lw=0.5, alpha=0.85)
            else:
                x_px = max(ci, cj) * patch_size
                y0   = ri * patch_size
                ax.plot([x_px, x_px], [y0, y0 + patch_size],
                        color="white", lw=0.5, alpha=0.85)

        correct = preds[idx].item() == labels[idx].item()
        ax.set_title(
            f"{'✓' if correct else '✗'}  groups={n_grps}",
            fontsize=7,
            color="green" if correct else "red",
        )
        ax.axis("off")

    # Hide any unused subplot cells
    for idx in range(B, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved group grid → {save_path}")


# ---------------------------------------------------------------------------
# Full validation accuracy loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_accuracy(model: ASFNet, loader, device: torch.device, args_dict: dict):
    top1 = AverageMeter()
    top5 = AverageMeter()

    for images, labels in tqdm(loader, desc="Evaluating accuracy"):
        images = images.to(device)
        labels = labels.to(device)
        logits, _, _, _, _ = _get_intermediate(model, images)
        acc1, acc5 = accuracy(logits, labels, topk=(1, 5))
        top1.update(acc1, images.size(0))
        top5.update(acc5, images.size(0))

    print(f"\nTop-1: {top1.avg:.2f}%  |  Top-5: {top5.avg:.2f}%")
    return top1.avg, top5.avg


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate and visualise ASFNet grouping")
    parser.add_argument("--checkpoint",   type=str, required=True,
                        help="Path to ASFNet .pt checkpoint file")
    parser.add_argument("--visualize",    type=int, default=0,
                        help="Number of individual images to save group visualisations for. "
                             "0 = skip visualisation, just print accuracy.")
    parser.add_argument("--grid",         action="store_true",
                        help="Also save a compact grid image showing one batch at a glance.")
    parser.add_argument("--no_accuracy",  action="store_true",
                        help="Skip the full accuracy evaluation loop — useful when you just "
                             "want quick visualisations without waiting for the full val pass.")
    parser.add_argument("--output_dir",   type=str, default="./viz_asfnet",
                        help="Directory to save visualisation images.")
    parser.add_argument("--alpha",        type=float, default=0.55,
                        help="Opacity of the group colour overlay on top of the image. "
                             "0 = original only, 1 = solid colour only. Default 0.55.")
    parser.add_argument("--device",       type=str, default="cuda")

    # Data args — only needed if you want to override the checkpoint's saved values
    parser.add_argument("--dataset_name",      type=str, default=None)
    parser.add_argument("--dataset_cache_dir", type=str, default=None)
    parser.add_argument("--jpeg_cache_dir",    type=str, default=None)
    parser.add_argument("--batch_size",        type=int, default=64)
    parser.add_argument("--num_workers",       type=int, default=4)
    parser.add_argument("--seed",              type=int, default=42)

    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model, ckpt_args = _load_model_from_checkpoint(args.checkpoint, device)

    # Build a namespace for get_dataloaders using checkpoint args as defaults,
    # with CLI overrides applied on top
    import types
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
        run_accuracy(model, val_loader, device, ckpt_args)

    if args.visualize > 0 or args.grid:
        os.makedirs(args.output_dir, exist_ok=True)

        images, labels = next(iter(val_loader))
        images = images.to(device)
        labels = labels.to(device)

        logits, group_ids, hard, _, mean_groups = _get_intermediate(model, images)
        preds = logits.argmax(dim=1)
        print(f"\nFirst batch — mean groups: {mean_groups:.1f}")

        if args.grid:
            n_grid = min(16, images.size(0))
            visualize_group_grid(
                images    = images[:n_grid].cpu(),
                all_group_ids = group_ids[:n_grid],
                all_hard  = hard[:n_grid],
                edge_indices  = model.router.edge_indices,
                grid_size = grid_size,
                patch_size = patch_size,
                preds     = preds[:n_grid],
                labels    = labels[:n_grid],
                save_path = os.path.join(args.output_dir, "group_grid.png"),
                alpha     = args.alpha,
            )

        n_vis = min(args.visualize, images.size(0))
        for idx in range(n_vis):
            n_grps  = int(group_ids[idx].max().item()) + 1
            correct = preds[idx].item() == labels[idx].item()
            title   = (
                f"{'✓' if correct else '✗'}  "
                f"pred={preds[idx].item()}  true={labels[idx].item()}  "
                f"groups={n_grps}"
            )
            visualize_groups(
                image_tensor = images[idx].cpu(),
                group_ids_1d = group_ids[idx],
                hard_1d      = hard[idx],
                edge_indices = model.router.edge_indices,
                grid_size    = grid_size,
                patch_size   = patch_size,
                n_groups     = n_grps,
                title        = title,
                save_path    = os.path.join(args.output_dir, f"groups_{idx:04d}.png"),
                alpha        = args.alpha,
            )
            print(f"  [{idx+1}/{n_vis}] {title}")

        print(f"\nVisualisations saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
