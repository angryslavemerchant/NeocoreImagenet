"""
evaluate_neocore.py — evaluation + instruments for NeocoreAE.

Three outputs into --output_dir:

  neocore_panels.png      original | admission-order overlay | reconstruction
                          — the admission-order map is THE instrument of the
                          loop era: patches coloured by the round they entered
                          working memory (dark = never admitted).
  neocore_rounds.png      per-round partial reconstructions for a few images
                          — watch memory accumulate. (Decoder is only trained
                          on full-K memories; early rounds are off-
                          distribution, read structure not fidelity.)
  admission_stats.txt/.png the anti-correlation test: for each round t, at
                          what percentile of the not-yet-admitted error
                          distribution (decoded from memory as of round t)
                          do round t+1's admissions land?
                          > 0.5 = error-seeking (the loop uses its memory);
                          ~ 0.5 = error-blind (sorted one-shot — nullity).

Runs on the instance after training (vast/run_training.sh) and locally.
"""

import os
import json
import types
import argparse

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dataset import get_dataloaders, IMAGENET_MEAN, IMAGENET_STD
from model_neocore import NeocoreAE


def load_model(path: str, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=True)
    a = ckpt["args"]
    model = NeocoreAE(
        image_size       = a["image_size"],
        patch_size       = a["patch_size"],
        d_model          = a["d_model"],
        num_heads        = a["num_heads"],
        core_blocks      = a["core_blocks"],
        mlp_ratio        = a["mlp_ratio"],
        rounds           = a["rounds"],
        memory_tokens    = a["memory_tokens"],
        decoder_d_model  = a["decoder_d_model"],
        decoder_blocks   = a["decoder_blocks"],
        decoder_heads    = a["decoder_heads"],
        norm_pix_loss    = not a.get("no_norm_pix", False),
        round_checkpoint = False,   # eval: no grads, no checkpointing
    )
    state_dict = {k.replace("_orig_mod.", "", 1): v
                  for k, v in ckpt["model"].items()}
    model.load_state_dict(state_dict)
    model.to(device).eval()
    print(f"Loaded {path} — epoch {ckpt['epoch'] + 1}, "
          f"val rec {ckpt.get('val_rec', float('nan')):.4f}, "
          f"R={a['rounds']} K={a['memory_tokens']}")
    return model, a


def denorm(img: torch.Tensor) -> np.ndarray:
    """(3, H, W) normalised tensor -> (H, W, 3) float image in [0, 1]."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std  = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    x = img.detach().float().cpu() * std + mean
    return x.clamp(0, 1).permute(1, 2, 0).numpy()


def norm_pix_view(pred_img: torch.Tensor) -> np.ndarray:
    """Normalised-pixel-space decoder output -> displayable [0, 1] image."""
    x = pred_img.detach().float().cpu()
    x = (x - x.min()) / (x.max() - x.min() + 1e-6)
    return x.permute(1, 2, 0).numpy()


def admission_overlay(img: np.ndarray, admit_round: np.ndarray,
                      grid: int, patch: int, rounds: int) -> np.ndarray:
    """
    Original dimmed; admitted patches tinted by admission round
    (viridis: dark blue = round 1, yellow = round R). Never-admitted
    patches stay dark grey.
    """
    cmap = plt.get_cmap("viridis", max(rounds, 2))
    out = img * 0.25
    rmap = admit_round.reshape(grid, grid)
    for r in range(grid):
        for c in range(grid):
            rd = rmap[r, c]
            ys, xs = r * patch, c * patch
            if rd >= 0:
                tint = np.array(cmap(rd)[:3])
                out[ys:ys + patch, xs:xs + patch] = (
                    0.45 * img[ys:ys + patch, xs:xs + patch] + 0.55 * tint)
    return out


@torch.no_grad()
def render_panels(model, images, out_dir, n_images):
    device = images.device
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                        enabled=device.type == "cuda"):
        pred_imgs, admitted, admit_round = model.reconstruct(images[:n_images])

    g, p, R = model.grid_size, model.patch_size, model.rounds
    fig, axes = plt.subplots(n_images, 3, figsize=(9, 3 * n_images))
    axes = np.atleast_2d(axes)
    for i in range(n_images):
        orig = denorm(images[i])
        axes[i, 0].imshow(orig)
        axes[i, 0].set_title("original" if i == 0 else "")
        axes[i, 1].imshow(admission_overlay(
            orig, admit_round[i].cpu().numpy(), g, p, R))
        axes[i, 1].set_title(f"admission order (R={R})" if i == 0 else "")
        axes[i, 2].imshow(norm_pix_view(pred_imgs[i]))
        axes[i, 2].set_title(f"reconstruction from K={model.memory_tokens}"
                             if i == 0 else "")
        for ax in axes[i]:
            ax.axis("off")
    fig.tight_layout()
    path = os.path.join(out_dir, "neocore_panels.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


@torch.no_grad()
def render_rounds(model, images, out_dir, n_images):
    device = images.device
    imgs = images[:n_images]
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                        enabled=device.type == "cuda"):
        preds, admits, _ = model.round_trace(imgs)

    R = model.rounds
    fig, axes = plt.subplots(n_images, R + 1,
                             figsize=(2.2 * (R + 1), 2.2 * n_images))
    axes = np.atleast_2d(axes)
    for i in range(n_images):
        axes[i, 0].imshow(denorm(imgs[i]))
        axes[i, 0].set_title("original" if i == 0 else "")
        for r in range(R):
            rec = model.unpatchify(preds[r].float())[i]
            k = int(admits[r][i].sum())
            axes[i, r + 1].imshow(norm_pix_view(rec))
            axes[i, r + 1].set_title(f"round {r+1} ({k} tok)"
                                     if i == 0 else "")
        for ax in axes[i]:
            ax.axis("off")
    fig.tight_layout()
    path = os.path.join(out_dir, "neocore_rounds.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


@torch.no_grad()
def admission_stats(model, loader, device, out_dir, max_batches):
    """
    The anti-correlation test, plus the nullity alarms and val rec over
    the sampled batches. Percentile semantics: for each token admitted at
    round t+1, the fraction of then-unadmitted tokens whose round-t
    reconstruction error is LOWER — 0.5 = error-blind, 1.0 = it always
    picks the currently worst-explained patches.
    """
    R = model.rounds
    pct_sums   = torch.zeros(max(R - 1, 1))
    pct_counts = torch.zeros(max(R - 1, 1))
    rec_sum, ovl_sum, corr_sum, n_b = 0.0, 0.0, 0.0, 0

    for b, (images, _labels) in enumerate(loader):
        if b >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            loss_rec, overlap, corr = model(images)
            preds, admits, admit_round = model.round_trace(images)

        target = model.patchify(images)
        if model.norm_pix_loss:
            mu  = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mu) / (var + 1e-6) ** 0.5

        for t in range(R - 1):
            err = ((preds[t].float() - target.float()) ** 2).mean(-1)  # (B,N)
            pool  = ~admits[t]                      # not yet admitted
            picks = admit_round == (t + 1)          # admitted next round
            # percentile of each pick's error within its image's pool:
            # lower[b, i, j] = err[b, j] < err[b, i]
            lower = (err.unsqueeze(1) < err.unsqueeze(2))       # (B, N, N)
            pool_f = pool.unsqueeze(1).float()
            frac_lower = (lower.float() * pool_f).sum(-1) \
                / pool_f.sum(-1).clamp(min=1.0)                 # (B, N)
            pct_sums[t]   += frac_lower[picks].sum().cpu()
            pct_counts[t] += picks.sum().cpu()

        rec_sum  += float(loss_rec)
        ovl_sum  += float(overlap)
        corr_sum += float(corr)
        n_b += 1

    stats = {
        "val_rec_sampled": rec_sum / max(n_b, 1),
        "overlap_r1":      ovl_sum / max(n_b, 1),
        "admit_corr":      corr_sum / max(n_b, 1),
        "error_percentile_by_round":
            [round(float(pct_sums[t] / pct_counts[t].clamp(min=1)), 4)
             for t in range(R - 1)],
    }
    print("\nAdmission statistics "
          f"(over {n_b} val batches):")
    print(f"  val rec (sampled)  {stats['val_rec_sampled']:.4f}")
    print(f"  overlap_r1         {stats['overlap_r1']:.3f}   "
          "(1.0 = memory == one-shot top-K)")
    print(f"  admit_corr         {stats['admit_corr']:.3f}   "
          "(-1.0 = admission order == round-1 ranking)")
    for t, p in enumerate(stats["error_percentile_by_round"]):
        print(f"  round {t+2} picks land at error percentile {p:.3f} "
              "(0.5 = error-blind, >0.5 = error-seeking)")

    with open(os.path.join(out_dir, "admission_stats.txt"), "w") as f:
        json.dump(stats, f, indent=2)

    if R > 1:
        fig, ax = plt.subplots(figsize=(5, 3))
        xs = np.arange(2, R + 1)
        ax.bar(xs, stats["error_percentile_by_round"], color="#3b7dd8")
        ax.axhline(0.5, color="gray", ls="--", lw=1, label="error-blind")
        ax.set_xlabel("admission round")
        ax.set_ylabel("residual-error percentile of picks")
        ax.set_ylim(0, 1)
        ax.legend()
        fig.tight_layout()
        path = os.path.join(out_dir, "admission_percentile.png")
        fig.savefig(path, dpi=120)
        plt.close(fig)
        print(f"saved {path}")
    return stats


def main():
    parser = argparse.ArgumentParser(description="NeocoreAE evaluation")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="viz_neocore")
    parser.add_argument("--num_images", type=int, default=6)
    parser.add_argument("--stat_batches", type=int, default=8,
                        help="val batches for the admission statistics")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    model, a = load_model(args.checkpoint, device)

    cfg = types.SimpleNamespace(**{**a, "batch_size": args.batch_size,
                                   "num_workers": min(a.get("num_workers", 8), 8)})
    _train_loader, val_loader = get_dataloaders(cfg)

    images, _ = next(iter(val_loader))
    images = images.to(device, non_blocking=True)

    render_panels(model, images, args.output_dir, args.num_images)
    render_rounds(model, images, args.output_dir, min(args.num_images, 4))
    admission_stats(model, val_loader, device, args.output_dir,
                    args.stat_batches)


if __name__ == "__main__":
    main()
