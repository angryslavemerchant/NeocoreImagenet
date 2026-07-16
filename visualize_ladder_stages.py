"""
visualize_ladder_stages.py — show the ladder AE's chunking process stage by
stage: original | stage-1 keep (784) | stage-2 keep (196) | stage-3 keep
(~30) | reconstruction, with retention drawn at 4x4-patch granularity.

Runs locally on CPU from a pulled checkpoint (the model is ~5M params).
Sample images stream from the HuggingFace dataset — no local 13 GB cache
needed. Also prints per-stage survivor-subgraph statistics (valid edges,
islands, cut edges) — the numbers behind the "ratio target collapses on
sparse subgraphs" hypothesis from the 2026-07-16 checkpoint notes.

Usage:
    python visualize_ladder_stages.py --checkpoint runs/AEL_4px_3stage/best.pt
"""

import argparse

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from model_asfnet_ae_ladder import ASFNetAELadder

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD  = np.array([0.229, 0.224, 0.225])


def load_samples(n: int, seed: int = 0) -> torch.Tensor:
    """Fetch n validation images via the HF datasets-server rows API (no
    local dataset cache, no `datasets` import); val preprocessing = resize
    shorter side to 256, center-crop 224, ImageNet-normalise (matches the
    DALI val pipeline)."""
    import io
    import requests

    offset = seed * 100  # crude variety knob; val split is class-ordered
    rows = requests.get(
        "https://datasets-server.huggingface.co/rows",
        params={"dataset": "clane9/imagenet-100", "config": "default",
                "split": "validation", "offset": offset,
                "length": min(100, max(n * 17, n))},
        timeout=60,
    ).json()["rows"]

    imgs = []
    for row in rows[::17][:n] if len(rows) >= n * 17 else rows[:n]:
        url = row["row"]["image"]["src"]
        img = Image.open(io.BytesIO(requests.get(url, timeout=60).content)).convert("RGB")
        w, h = img.size
        scale = 256 / min(w, h)
        img = img.resize((round(w * scale), round(h * scale)), Image.BILINEAR)
        w, h = img.size
        left, top = (w - 224) // 2, (h - 224) // 2
        img = img.crop((left, top, left + 224, top + 224))
        arr = (np.asarray(img, dtype=np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        imgs.append(torch.from_numpy(arr.transpose(2, 0, 1)).float())
    return torch.stack(imgs)


def denorm(img_chw: torch.Tensor) -> np.ndarray:
    x = img_chw.numpy().transpose(1, 2, 0)
    return np.clip(x * IMAGENET_STD + IMAGENET_MEAN, 0, 1)


def keep_overlay(img_np: np.ndarray, keep_row: torch.Tensor,
                 grid: int, patch: int) -> np.ndarray:
    """Dim dropped patches to 25% brightness."""
    out = img_np.copy()
    k = keep_row.reshape(grid, grid).numpy()
    for r in range(grid):
        for c in range(grid):
            if not k[r, c]:
                out[r * patch:(r + 1) * patch, c * patch:(c + 1) * patch] *= 0.25
    return out


def components_np(keep_row: np.ndarray, hard_row: np.ndarray,
                  valid_row: np.ndarray, edges: np.ndarray, n: int) -> np.ndarray:
    """Connected components of the survivor subgraph (uncut valid edges);
    dropped tokens = -1. Plain BFS on CPU."""
    adj = [[] for _ in range(n)]
    for (i, j), h, v in zip(edges, hard_row, valid_row):
        if v and h < 0.5:
            adj[i].append(j)
            adj[j].append(i)
    labels = np.full(n, -1, dtype=np.int64)
    nxt = 0
    for start in range(n):
        if not keep_row[start] or labels[start] >= 0:
            continue
        stack = [start]
        labels[start] = nxt
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if keep_row[v] and labels[v] < 0:
                    labels[v] = nxt
                    stack.append(v)
        nxt += 1
    return labels


def chunk_map(img_np: np.ndarray, labels: np.ndarray, grid: int, patch: int,
              seed: int = 0) -> np.ndarray:
    """Random-color chunk overlay blended onto the (dimmed) image; dropped
    tokens stay dark."""
    rng = np.random.RandomState(seed)
    n_lab = labels.max() + 1
    palette = rng.rand(max(n_lab, 1), 3) * 0.9 + 0.1
    out = img_np * 0.35
    lab2d = labels.reshape(grid, grid)
    for r in range(grid):
        for c in range(grid):
            if lab2d[r, c] >= 0:
                col = palette[lab2d[r, c]]
                blk = out[r * patch:(r + 1) * patch, c * patch:(c + 1) * patch]
                out[r * patch:(r + 1) * patch, c * patch:(c + 1) * patch] = \
                    0.45 * blk + 0.55 * col
    return np.clip(out, 0, 1)


def subgraph_stats(model, keep_prev: torch.Tensor, hard: torch.Tensor,
                   valid: torch.Tensor) -> str:
    n_tok   = keep_prev.sum(dim=1).float()
    n_edge  = valid.sum(dim=1).float()
    cuts    = (hard * valid.float()).sum(dim=1)
    # islands: kept tokens with zero valid incident edges
    ei = model.stages[0].router.edge_indices
    vf = valid.float()
    per_tok = vf.new_zeros(vf.shape[0], model.n_patches)
    per_tok = per_tok.index_add(1, ei[:, 0], vf).index_add(1, ei[:, 1], vf)
    islands = (keep_prev & (per_tok < 0.5)).sum(dim=1).float()
    return (f"tok={n_tok.mean():6.1f} edges={n_edge.mean():7.1f} "
            f"edges/tok={(n_edge / n_tok.clamp(min=1)).mean():.2f} "
            f"cuts={cuts.mean():6.1f} islands={islands.mean():5.1f}")


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str,
                    default="runs/AEL_4px_3stage/best.pt")
    ap.add_argument("--n_images", type=int, default=6)
    ap.add_argument("--out", type=str,
                    default="runs/AEL_4px_3stage/viz/stages.png")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    a = ckpt["args"]
    assert a.get("ladder"), "not a ladder checkpoint"
    model = ASFNetAELadder(
        image_size        = a["image_size"],
        patch_size        = a["patch_size"],
        mlp_ratio         = a["mlp_ratio"],
        target_group_size = a["target_group_size"],
        router_proj_dim   = a["router_proj_dim"],
        norm_pix_loss     = not a.get("no_norm_pix", False),
        router_kind       = a.get("router_kind", "edge"),
        budget_floor      = a.get("budget_floor", False),
    )
    sd = {k.replace("_orig_mod.", "", 1): v for k, v in ckpt["model"].items()}
    model.load_state_dict(sd, strict=True)
    model.eval()
    print(f"loaded {args.checkpoint} (epoch {ckpt['epoch'] + 1})")

    imgs = load_samples(args.n_images, args.seed)
    print(f"streamed {len(imgs)} val images")

    feats, pad_mask, sel, keep, _, kept_means, stage_keeps = \
        model.forward_features(imgs)
    recon, _ = model.reconstruct(imgs)

    # per-stage subgraph stats (rerun routers on the recorded keeps),
    # collecting (keep_prev, hard, valid) for the chunk maps
    print("\nper-stage survivor-subgraph statistics:")
    stage_cutdata = []
    keep_prev = torch.ones_like(stage_keeps[0])
    tokens, coords = model.patch_embed(imgs)
    tok_c, coord_c, pm, sl = tokens, coords, None, None
    from model_asfnet_br import compact_survivors
    for i, stage in enumerate(model.stages):
        tok_c = stage.proj(tok_c)
        for blk in stage.blocks:
            tok_c = blk(tok_c, coord_c, pm)
        d = tok_c.shape[-1]
        if sl is None:
            full = tok_c
        else:
            full = tok_c.new_zeros(imgs.shape[0], model.n_patches, d)
            full = full.scatter(1, sl.unsqueeze(-1).expand(-1, -1, d), tok_c)
        hard, probs, valid, _ = stage.router(full, keep_prev,
                                             model.target_group_size)
        print(f"  stage {i + 1}: {subgraph_stats(model, keep_prev, hard, valid)}"
              f"  -> kept {stage_keeps[i].sum(dim=1).float().mean():.1f}")
        stage_cutdata.append((keep_prev.clone(), hard.clone(), valid.clone()))
        # replay the residual+compact so stage i+1 sees the right features
        ei = stage.router.edge_indices
        pw = probs * valid.to(probs.dtype)
        s_k = pw.new_zeros(pw.shape[0], model.n_patches)
        s_k = s_k.index_add(1, ei[:, 0], pw).index_add(1, ei[:, 1], pw)
        s_slot = s_k if sl is None else s_k.gather(1, sl)
        tok_c = tok_c + s_slot.unsqueeze(-1) * tok_c
        if sl is None:
            full_post = tok_c
        else:
            full_post = tok_c.new_zeros(imgs.shape[0], model.n_patches, d)
            full_post = full_post.scatter(
                1, sl.unsqueeze(-1).expand(-1, -1, d), tok_c)
        keep_prev = stage_keeps[i]
        tok_c, coord_c, pm, sl, _ = compact_survivors(full_post, coords,
                                                      keep_prev)

    # ---- panel: original | per stage (chunk map, keep) | reconstruction ----
    g, p = model.grid_size, model.patch_size
    n = len(imgs)
    edges_np = model.stages[0].router.edge_indices.numpy()
    cols = 2 + 2 * len(stage_keeps)
    fig, axes = plt.subplots(n, cols, figsize=(2.2 * cols, 2.2 * n))
    for i in range(n):
        img_np = denorm(imgs[i])
        axes[i, 0].imshow(img_np)
        axes[i, 0].set_title("original" if i == 0 else "", fontsize=8)
        for j in range(len(stage_keeps)):
            kp, hard, valid = stage_cutdata[j]
            labels = components_np(kp[i].numpy(), hard[i].numpy(),
                                   valid[i].numpy(), edges_np, model.n_patches)
            n_chunks = labels.max() + 1
            axes[i, 1 + 2 * j].imshow(chunk_map(img_np, labels, g, p))
            axes[i, 1 + 2 * j].set_title(
                f"stage {j + 1} chunks ({n_chunks})", fontsize=8 if i == 0 else 7)
            sk = stage_keeps[j]
            axes[i, 2 + 2 * j].imshow(keep_overlay(img_np, sk[i], g, p))
            axes[i, 2 + 2 * j].set_title(
                f"keep ({int(sk[i].sum())})", fontsize=8 if i == 0 else 7)
        rec_np = recon[i].numpy().transpose(1, 2, 0)
        rec_np = (rec_np - rec_np.min()) / (rec_np.max() - rec_np.min() + 1e-8)
        axes[i, -1].imshow(rec_np)
        axes[i, -1].set_title("reconstruction (norm space)" if i == 0 else "",
                              fontsize=8)
    for ax in axes.flat:
        ax.axis("off")
    fig.tight_layout()
    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
