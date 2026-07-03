import os
import argparse

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from tqdm import tqdm

from config import Config
from dataset import get_dataloaders, IMAGENET_MEAN, IMAGENET_STD
from model import SaccadeNet
from utils import AverageMeter, accuracy


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """
    Invert ImageNet normalization and return an HxWxC numpy array in [0, 1].
    """
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std  = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = (tensor.cpu() * std + mean).clamp(0, 1)
    return img.permute(1, 2, 0).numpy()


def visualize_trajectory(
    image_tensor: torch.Tensor,
    pos_history: list,
    cfg: Config,
    title: str = "",
    save_path: str = None,
):
    """
    Render the sequence of patch positions over the image.

    Each rectangle shows where the model looked at step t.
    Rectangles are colour-coded dark->light (early->late steps).
    Arrows show the direction of each movement.

    Args:
        image_tensor: (C, H, W) normalized tensor for a single image
        pos_history:  list of (2,) tensors in normalized [-1, 1] coords
        cfg:          Config
        title:        plot title (e.g. predicted vs true class)
        save_path:    save to file if provided, otherwise display
    """
    img = denormalize(image_tensor)
    H = W = cfg.image_size
    half = cfg.patch_size / 2

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img)
    if title:
        ax.set_title(title, fontsize=10)
    ax.axis("off")

    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(pos_history)))

    def norm_to_px(pos):
        """Convert normalized [-1, 1] to pixel coords."""
        px = (pos[0] + 1) / 2 * (W - 1)
        py = (pos[1] + 1) / 2 * (H - 1)
        return float(px), float(py)

    prev_cx, prev_cy = None, None
    for t, (pos, color) in enumerate(zip(pos_history, colors)):
        cx, cy = norm_to_px(pos)

        rect = mpatches.Rectangle(
            (cx - half, cy - half), cfg.patch_size, cfg.patch_size,
            linewidth=1.5, edgecolor=color, facecolor="none", alpha=0.85,
        )
        ax.add_patch(rect)
        ax.text(cx, cy, str(t), fontsize=6, ha="center", va="center",
                color=color, fontweight="bold")

        if prev_cx is not None:
            ax.annotate(
                "", xy=(cx, cy), xytext=(prev_cx, prev_cy),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.0),
            )
        prev_cx, prev_cy = cx, cy

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close()
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    cfg: Config,
    checkpoint_path: str,
    visualize_n: int = 0,
    traj_dir: str = "./trajectories",
):
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    model = SaccadeNet(cfg).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded checkpoint — epoch {ckpt['epoch']+1}, "
          f"saved val top-1: {ckpt.get('val_top1', '?'):.2f}%\n")

    _, val_loader = get_dataloaders(cfg)

    top1 = AverageMeter()
    top5 = AverageMeter()
    per_class_correct = torch.zeros(cfg.num_classes)
    per_class_total   = torch.zeros(cfg.num_classes)

    visualized = 0
    if visualize_n > 0:
        os.makedirs(traj_dir, exist_ok=True)

    # Print one batch of raw trajectory coordinates before the main loop.
    # Lets us distinguish "model is genuinely stuck" from "visualization bug".
    print("\n--- Trajectory debug (first batch, first 3 images) ---")
    _debug_images, _ = next(iter(val_loader))
    _debug_images = _debug_images.to(device)
    with torch.no_grad():
        _, _, _pos_history, _pos_0 = model(_debug_images)
    for img_idx in range(min(3, _debug_images.size(0))):
        print(f"  image {img_idx}  start: [{_pos_0[img_idx][0].item():.3f}, {_pos_0[img_idx][1].item():.3f}]")
        for t, pos in enumerate(_pos_history):
            x, y = pos[img_idx][0].item(), pos[img_idx][1].item()
            print(f"    step {t:02d}: [{x:.3f}, {y:.3f}]")
    print("--- end debug ---\n")

    for images, labels in tqdm(val_loader, desc="Evaluating"):
        images = images.to(device)
        labels = labels.to(device)

        logits, _, pos_history, _ = model(images)

        acc1, acc5 = accuracy(logits, labels, topk=(1, 5))
        B = images.size(0)
        top1.update(acc1, B)
        top5.update(acc5, B)

        preds = logits.argmax(dim=1)
        for pred, label in zip(preds, labels):
            c = label.item()
            per_class_total[c] += 1
            if pred.item() == c:
                per_class_correct[c] += 1

        # Visualize the first N images
        if visualized < visualize_n:
            # pos_history is list of (B, 2); take the first image in batch
            pos_list = [p[0].cpu() for p in pos_history]
            pred_cls  = preds[0].item()
            true_cls  = labels[0].item()
            title = f"pred={pred_cls}  true={true_cls}  {'✓' if pred_cls == true_cls else '✗'}"
            visualize_trajectory(
                images[0].cpu(), pos_list, cfg,
                title=title,
                save_path=os.path.join(traj_dir, f"traj_{visualized:04d}.png"),
            )
            visualized += 1

    # Summary
    per_class_acc = (per_class_correct / per_class_total.clamp(min=1)) * 100
    best_idx  = per_class_acc.argsort(descending=True)[:5].tolist()
    worst_idx = per_class_acc.argsort()[:5].tolist()

    print(f"Top-1: {top1.avg:.2f}%  |  Top-5: {top5.avg:.2f}%")
    print(f"\nBest 5 classes  (class idx: acc%): "
          + ", ".join(f"{i}: {per_class_acc[i]:.1f}%" for i in best_idx))
    print(f"Worst 5 classes (class idx: acc%): "
          + ", ".join(f"{i}: {per_class_acc[i]:.1f}%" for i in worst_idx))

    if visualize_n > 0:
        print(f"\nTrajectories saved to {traj_dir}/")

    return top1.avg, top5.avg


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SaccadeNet checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to .pt checkpoint file")
    parser.add_argument("--visualize", type=int, default=0,
                        help="Number of validation images to render trajectories for")
    parser.add_argument("--traj_dir", type=str, default="./trajectories",
                        help="Directory to save trajectory images")
    args = parser.parse_args()

    cfg = Config()
    evaluate(cfg, args.checkpoint, visualize_n=args.visualize, traj_dir=args.traj_dir)