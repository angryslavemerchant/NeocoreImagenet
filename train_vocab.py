"""
Stage 0-1 of the vocabulary program (2026-07-18).

Stage 0 — the token lake: run frozen DINOv2-S/14 over ImageNet-100 (RAM
blobs, deterministic 224 center-crop) and save all patch tokens (N, 256,
384 fp16) + CLS tokens per split.

Stage 1 — the vocabulary: EMA k-means codebooks over the train lake for a
sweep of K, plus the instruments:
  - perplexity curves + final usage histograms (Zipf plot)
  - quantization error (MSE + cosine) per K
  - code maps (val images colored by code id) and code exemplar sheets
    (top patches per frequent code) for the headline K
  - attentive probes over raw vs quantized val tokens (rate-distortion
    of the vocabulary) — trained directly on the lake, no backbone
    forward, so epochs are seconds
  - kNN on CLS tokens as a training-free reference

The codebook adds no information (it is strictly lossy); its job is to
reify DINO's latent cluster structure into named, countable symbols.
Probe numbers here are NOT comparable to the train_linear_probe.py table
(different backbone, no augmentation); only raw-vs-quantized within this
script is a fair comparison.

Everything logs to wandb; codebooks + figures also upload as one small
verified artifact (the lake itself stays on-instance — rebuildable in
~15 min).
"""

import argparse
import math
import os
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Stage 0 — token lake
# ---------------------------------------------------------------------------

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
CROP = 224
GRID = 16                      # 224 / 14
D_FEAT = 384                   # DINOv2-S


def load_dinov2(device):
    torch.hub.set_dir(os.environ.get("TORCH_HOME",
                                     str(Path.home() / ".cache" / "torch")))
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    return model.to(device).eval()


def _center_crop_norm(u8: torch.Tensor, device) -> torch.Tensor:
    """(B,3,256,256) uint8 cpu -> (B,3,224,224) normalized float on device."""
    off = (u8.shape[-1] - CROP) // 2
    x = u8[:, :, off:off + CROP, off:off + CROP].to(device).float()
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1) * 255
    std  = torch.tensor(IMAGENET_STD,  device=device).view(1, 3, 1, 1) * 255
    return (x - mean) / std


@torch.no_grad()
def extract_split(model, blob_path: Path, out_path: Path, device,
                  batch: int = 256):
    if out_path.exists():
        print(f"[lake] {out_path.name} exists — skipping")
        return
    blob = torch.load(blob_path, map_location="cpu", mmap=True)
    images, labels = blob["images"], blob["labels"]
    n = images.shape[0]
    tokens = torch.empty(n, GRID * GRID, D_FEAT, dtype=torch.float16)
    cls    = torch.empty(n, D_FEAT, dtype=torch.float16)
    t0 = time.time()
    for i in range(0, n, batch):
        x = _center_crop_norm(images[i:i + batch], device)
        with torch.autocast("cuda", dtype=torch.float16):
            out = model.forward_features(x)
        tokens[i:i + batch] = out["x_norm_patchtokens"].half().cpu()
        cls[i:i + batch]    = out["x_norm_clstoken"].half().cpu()
        if (i // batch) % 50 == 0:
            done = i + x.shape[0]
            print(f"[lake] {blob_path.stem}: {done}/{n} "
                  f"({done / max(time.time() - t0, 1e-9):.0f} img/s)")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp")
    torch.save({"tokens": tokens, "cls": cls, "labels": labels.clone()}, tmp)
    tmp.rename(out_path)
    print(f"[lake] wrote {out_path.name}: {n} imgs, "
          f"{tokens.numel() * 2 / 1e9:.1f} GB")


def stage0(args, device):
    from dataset_ram import ensure_ram_cache
    cfg = SimpleNamespace(jpeg_cache_dir=args.jpeg_cache_dir,
                          dataset_name=args.dataset_name,
                          dataset_cache_dir=args.dataset_cache_dir)
    train_blob, val_blob = ensure_ram_cache(cfg)
    model = load_dinov2(device)
    lake = Path(args.lake_dir)
    extract_split(model, train_blob, lake / "train_tokens.pt", device)
    extract_split(model, val_blob,   lake / "val_tokens.pt",   device)
    del model
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Stage 1a — EMA k-means codebook
# ---------------------------------------------------------------------------

class EMACodebook:
    """Online (minibatch EMA) k-means over a stationary token stream.

    The classic VQ instability is a two-body problem (encoder chasing
    codebook); with frozen features there is one moving part and this is
    plain online k-means. gamma=0.99, Laplace-smoothed counts, dead codes
    reset to random batch tokens after each pass."""

    def __init__(self, k: int, d: int, device, gamma: float = 0.99):
        self.k, self.gamma = k, gamma
        self.codes = torch.empty(k, d, device=device)
        self.counts = torch.zeros(k, device=device)
        self.sums = torch.zeros(k, d, device=device)
        self.usage = torch.zeros(k, device=device)   # per-pass usage
        self.inited = False

    def init_from(self, tokens: torch.Tensor):
        idx = torch.randperm(tokens.shape[0], device=tokens.device)[:self.k]
        self.codes.copy_(tokens[idx])
        # counts/sums start at ZERO: the m/N ratio is then exactly the
        # (EMA-weighted) mean of assigned tokens from the first batch on —
        # no init anchor, unbiased at any update count.
        self.inited = True

    @torch.no_grad()
    def assign(self, t: torch.Tensor) -> torch.Tensor:
        # ||t-c||^2 argmin via the expanded form; t (B,d) fp32
        d2 = (t.pow(2).sum(1, keepdim=True)
              - 2 * t @ self.codes.T
              + self.codes.pow(2).sum(1)[None, :])
        return d2.argmin(dim=1)

    @torch.no_grad()
    def update(self, t: torch.Tensor) -> float:
        a = self.assign(t)
        one = torch.ones_like(a, dtype=torch.float)
        n_k = torch.zeros(self.k, device=t.device).index_add_(0, a, one)
        s_k = torch.zeros_like(self.sums).index_add_(0, a, t)
        self.counts.mul_(self.gamma).add_(n_k, alpha=1 - self.gamma)
        self.sums.mul_(self.gamma).add_(s_k, alpha=1 - self.gamma)
        live = self.counts > 1e-6      # untouched codes keep their init
        self.codes[live] = self.sums[live] / self.counts[live].unsqueeze(1)
        self.usage.index_add_(0, a, one)
        p = n_k / n_k.sum()
        return float(torch.exp(-(p * (p + 1e-12).log()).sum()))  # batch pplx

    @torch.no_grad()
    def reset_dead(self, t: torch.Tensor) -> int:
        """Re-seed under-used codes at tokens sampled prop. to squared
        distance from their nearest code (k-means++ logic at reset time —
        collisions resolve toward unclaimed density instead of randomly)."""
        share = self.usage / self.usage.sum().clamp(min=1)
        dead = share < (1.0 / (10 * self.k))
        n_dead = int(dead.sum())
        if n_dead:
            d2 = (t.pow(2).sum(1, keepdim=True)
                  - 2 * t @ self.codes.T
                  + self.codes.pow(2).sum(1)[None, :]).min(dim=1).values
            idx = torch.multinomial(d2.clamp(min=0) + 1e-9, n_dead,
                                    replacement=False)
            self.codes[dead] = t[idx]
            self.counts[dead] = 0.0    # fresh EMA — first assignment lands
            self.sums[dead] = 0.0      # exactly on its batch mean

        self.usage.zero_()
        return n_dead


def token_batches(tokens: torch.Tensor, device, batch_tokens: int,
                  seed: int):
    """Yield shuffled (batch_tokens, d) fp32 GPU batches for one pass."""
    n_img = tokens.shape[0]
    per = tokens.shape[1]
    g = torch.Generator().manual_seed(seed)
    order = torch.randperm(n_img, generator=g)
    imgs_per_batch = max(batch_tokens // per, 1)
    for i in range(0, n_img, imgs_per_batch):
        sel = order[i:i + imgs_per_batch]
        yield tokens[sel].reshape(-1, tokens.shape[-1]).to(device).float()


def train_codebook(tokens, k, device, passes, batch_tokens, wandb_run,
                   tag, reset_every: int = 50):
    cb = EMACodebook(k, tokens.shape[-1], device)
    step = 0
    for p in range(passes):
        n_dead = 0
        for t in token_batches(tokens, device, batch_tokens, seed=1000 + p):
            if not cb.inited:
                cb.init_from(t)
            pplx = cb.update(t)
            if step % 50 == 0:
                wandb_run.log({f"{tag}/batch_pplx": pplx,
                               f"{tag}/step": step})
            step += 1
            if step % reset_every == 0 and p < passes - 1:
                n_dead += cb.reset_dead(t)   # no resets in the final pass
        print(f"[vq K={k}] pass {p + 1}/{passes}: batch_pplx={pplx:.0f}, "
              f"reset {n_dead} dead codes")
    return cb


@torch.no_grad()
def codebook_stats(cb, tokens, device, batch_tokens=2 ** 16):
    """Full-lake usage histogram + quantization error."""
    usage = torch.zeros(cb.k, device=device)
    mse_sum, cos_sum, n_tok = 0.0, 0.0, 0
    for t in token_batches(tokens, device, batch_tokens, seed=7):
        a = cb.assign(t)
        usage.index_add_(0, a, torch.ones_like(a, dtype=torch.float))
        q = cb.codes[a]
        mse_sum += float(F.mse_loss(q, t, reduction="sum"))
        cos_sum += float(F.cosine_similarity(q, t, dim=1).sum())
        n_tok += t.shape[0]
    p = usage / usage.sum()
    pplx = float(torch.exp(-(p * (p + 1e-12).log()).sum()))
    return {"usage": usage.cpu(), "perplexity": pplx,
            "mse": mse_sum / (n_tok * tokens.shape[-1]),
            "cosine": cos_sum / n_tok}


# ---------------------------------------------------------------------------
# Stage 1b — instruments
# ---------------------------------------------------------------------------

def zipf_figure(stats_by_k, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for k, s in sorted(stats_by_k.items()):
        u = s["usage"].sort(descending=True).values.numpy()
        ax[0].loglog(np.arange(1, len(u) + 1), u / u.sum(),
                     label=f"K={k} (pplx {s['perplexity']:.0f})")
    ax[0].set_xlabel("code rank"); ax[0].set_ylabel("usage share")
    ax[0].set_title("Vocabulary usage (Zipf plot)"); ax[0].legend()
    ks = sorted(stats_by_k)
    ax[1].semilogx(ks, [stats_by_k[k]["cosine"] for k in ks], "o-")
    ax[1].set_xlabel("K"); ax[1].set_ylabel("mean cosine(token, code)")
    ax[1].set_title("Quantization fidelity vs vocabulary size")
    fig.tight_layout(); fig.savefig(out_png, dpi=140); plt.close(fig)


@torch.no_grad()
def code_map_figure(cb, val_tokens, val_blob_path, out_png, device,
                    n_imgs=16):
    """Val images beside their code-id maps (headline K)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    blob = torch.load(val_blob_path, map_location="cpu", mmap=True)
    g = torch.Generator().manual_seed(3)
    sel = torch.randperm(val_tokens.shape[0], generator=g)[:n_imgs]
    rows = int(math.ceil(n_imgs / 4))
    fig, axes = plt.subplots(rows, 8, figsize=(20, 2.6 * rows))
    for j, i in enumerate(sel.tolist()):
        img = blob["images"][i].permute(1, 2, 0).numpy()
        a = cb.assign(val_tokens[i].to(device).float()) \
            .reshape(GRID, GRID).cpu().numpy()
        r, c = divmod(j, 4)
        axes[r, 2 * c].imshow(img); axes[r, 2 * c].axis("off")
        axes[r, 2 * c + 1].imshow(a % 20, cmap="tab20",
                                  interpolation="nearest")
        axes[r, 2 * c + 1].axis("off")
    fig.suptitle(f"Code maps (K={cb.k}; colors = code id mod 20)")
    fig.tight_layout(); fig.savefig(out_png, dpi=140); plt.close(fig)


@torch.no_grad()
def code_exemplar_figure(cb, val_tokens, val_blob_path, out_png, device,
                         n_codes=24, n_ex=8):
    """For the most frequent codes: the val patches that best match them.
    'What does code i look like' — the part-vs-texture verdict at a glance."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    blob = torch.load(val_blob_path, map_location="cpu", mmap=True)
    n_img = val_tokens.shape[0]
    usage = torch.zeros(cb.k, device=device)
    best = {}          # code -> list of (sim, img, patch)
    for i0 in range(0, n_img, 512):
        t = val_tokens[i0:i0 + 512].to(device).float()
        B, P, D = t.shape
        a = cb.assign(t.reshape(-1, D))
        usage.index_add_(0, a, torch.ones_like(a, dtype=torch.float))
    top_codes = usage.topk(n_codes).indices
    codes_n = F.normalize(cb.codes[top_codes], dim=1)
    sims_store = [[] for _ in range(n_codes)]
    for i0 in range(0, n_img, 512):
        t = val_tokens[i0:i0 + 512].to(device).float()
        B, P, D = t.shape
        sims = F.normalize(t.reshape(-1, D), dim=1) @ codes_n.T  # (B*P,nc)
        v, idx = sims.topk(2, dim=0).values, sims.topk(2, dim=0).indices
        for c in range(n_codes):
            for r in range(2):
                flat = int(idx[r, c]); s = float(v[r, c])
                sims_store[c].append((s, i0 + flat // P, flat % P))
    patch_px = 256 // GRID
    fig, axes = plt.subplots(n_codes, n_ex,
                             figsize=(1.1 * n_ex, 1.1 * n_codes))
    for c in range(n_codes):
        ex = sorted(sims_store[c], reverse=True)[:n_ex]
        for e, (s, img_i, p_i) in enumerate(ex):
            py, px = divmod(p_i, GRID)
            y0, x0 = py * patch_px, px * patch_px
            pad = patch_px  # show 3x3 patch neighborhood for context
            img = blob["images"][img_i].permute(1, 2, 0).numpy()
            y1, y2 = max(y0 - pad, 0), min(y0 + 2 * patch_px, 256)
            x1, x2 = max(x0 - pad, 0), min(x0 + 2 * patch_px, 256)
            axes[c, e].imshow(img[y1:y2, x1:x2])
        for e in range(n_ex):
            axes[c, e].axis("off")
        axes[c, 0].set_ylabel(f"c{int(top_codes[c])}", rotation=0,
                              labelpad=18, fontsize=7)
    fig.suptitle(f"Top-{n_codes} codes by usage — nearest val patches "
                 f"(3x3 patch context)")
    fig.tight_layout(); fig.savefig(out_png, dpi=140); plt.close(fig)


# ---------------------------------------------------------------------------
# Stage 1c — probes on the lake (raw vs quantized)
# ---------------------------------------------------------------------------

class AttentiveProbe(nn.Module):
    def __init__(self, d=D_FEAT, heads=6, n_classes=100):
        super().__init__()
        self.q = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, n_classes)

    def forward(self, tok):                      # (B,P,d)
        q = self.q.expand(tok.shape[0], -1, -1)
        pooled, _ = self.attn(q, tok, tok)
        return self.head(self.norm(pooled.squeeze(1)))


def probe_on_lake(train_tokens, train_labels, val_tokens, val_labels,
                  device, cb=None, epochs=40, batch=1024, lr=1e-3,
                  tag="raw", wandb_run=None):
    """Attentive probe trained directly on saved tokens (optionally
    quantized through codebook cb). No backbone forward — epochs are
    seconds. Only raw-vs-quantized comparisons within this script are
    fair (no augmentation here)."""
    model = AttentiveProbe().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n = train_tokens.shape[0]
    y_train = train_labels.long()
    y_val = val_labels.long().to(device)

    def get(tokens_cpu, sel):
        t = tokens_cpu[sel].to(device).float()
        if cb is not None:
            B, P, D = t.shape
            t = cb.codes[cb.assign(t.reshape(-1, D))].reshape(B, P, D)
        return t

    best = 0.0
    for ep in range(epochs):
        model.train()
        order = torch.randperm(n)
        for i in range(0, n, batch):
            sel = order[i:i + batch]
            logits = model(get(train_tokens, sel))
            loss = F.cross_entropy(logits, y_train[sel].to(device))
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        model.eval()
        correct = tot = 0
        with torch.no_grad():
            for i in range(0, val_tokens.shape[0], batch):
                sel = torch.arange(i, min(i + batch, val_tokens.shape[0]))
                pred = model(get(val_tokens, sel)).argmax(1)
                correct += int((pred == y_val[sel]).sum()); tot += len(sel)
        acc = 100 * correct / tot
        best = max(best, acc)
        if wandb_run:
            wandb_run.log({f"probe_{tag}/val_top1": acc,
                           f"probe_{tag}/epoch": ep})
        if ep % 10 == 0 or ep == epochs - 1:
            print(f"[probe {tag}] ep {ep + 1}/{epochs}: top1 {acc:.2f} "
                  f"(best {best:.2f})")
    return best


@torch.no_grad()
def knn_cls(train_cls, train_labels, val_cls, val_labels, device, k=20):
    """Training-free kNN on CLS tokens (DINO-standard eval, cosine)."""
    tr = F.normalize(train_cls.to(device).float(), dim=1)
    y = train_labels.long().to(device)
    correct = tot = 0
    for i in range(0, val_cls.shape[0], 1024):
        v = F.normalize(val_cls[i:i + 1024].to(device).float(), dim=1)
        sims, idx = (v @ tr.T).topk(k, dim=1)
        votes = F.one_hot(y[idx], 100).float() * sims.softmax(1)[..., None]
        pred = votes.sum(1).argmax(1)
        yv = val_labels[i:i + 1024].long().to(device)
        correct += int((pred == yv).sum()); tot += len(yv)
    return 100 * correct / tot


# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", type=str, default="all",
                   choices=["0", "1", "all"])
    p.add_argument("--lake_dir", type=str, default="./data/dino_lake")
    p.add_argument("--jpeg_cache_dir",    type=str, default="./jpeg_cache")
    p.add_argument("--dataset_name",      type=str,
                   default="clane9/imagenet-100")
    p.add_argument("--dataset_cache_dir", type=str, default="./data")
    p.add_argument("--ks", type=int, nargs="+",
                   default=[512, 1024, 2048, 4096, 8192])
    p.add_argument("--headline_k", type=int, default=2048)
    p.add_argument("--probe_ks", type=int, nargs="+", default=[1024, 8192])
    p.add_argument("--passes", type=int, default=3)
    p.add_argument("--batch_tokens", type=int, default=2 ** 16)
    p.add_argument("--probe_epochs", type=int, default=40)
    p.add_argument("--wandb_project", type=str, default="neocore")
    p.add_argument("--run_name", type=str, default="VOCAB_stage01")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda"

    import wandb
    run = wandb.init(project=args.wandb_project, name=args.run_name,
                     config=vars(args))

    run_dir = Path("runs") / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (Path("runs") / "LATEST").write_text(str(run_dir))

    if args.stage in ("0", "all"):
        stage0(args, device)
    if args.stage == "0":
        run.finish(); return

    lake = Path(args.lake_dir)
    train = torch.load(lake / "train_tokens.pt", map_location="cpu",
                       mmap=True)
    val = torch.load(lake / "val_tokens.pt", map_location="cpu", mmap=True)
    print(f"[lake] train {train['tokens'].shape}, val {val['tokens'].shape}")

    # --- codebooks + census ---
    stats_by_k, books = {}, {}
    for k in args.ks:
        cb = train_codebook(train["tokens"], k, device, args.passes,
                            args.batch_tokens, run, tag=f"vq_k{k}")
        s = codebook_stats(cb, train["tokens"], device)
        stats_by_k[k], books[k] = s, cb
        print(f"[vq K={k}] lake pplx {s['perplexity']:.0f}  "
              f"mse {s['mse']:.4f}  cos {s['cosine']:.4f}")
        run.log({f"final/pplx_k{k}": s["perplexity"],
                 f"final/cos_k{k}": s["cosine"]})
        torch.save(cb.codes.cpu(), run_dir / f"codebook_k{k}.pt")

    zipf_figure(stats_by_k, run_dir / "zipf.png")
    val_blob = Path(args.jpeg_cache_dir) / "ram256" / "validation.pt"
    hk = books[args.headline_k]
    code_map_figure(hk, val["tokens"], val_blob,
                    run_dir / f"code_maps_k{args.headline_k}.png", device)
    code_exemplar_figure(hk, val["tokens"], val_blob,
                         run_dir / f"code_exemplars_k{args.headline_k}.png",
                         device)

    # --- probes: ceiling, quantized sweep, kNN reference ---
    knn = knn_cls(train["cls"], train["labels"], val["cls"], val["labels"],
                  device)
    print(f"[knn] CLS 20-nn top1: {knn:.2f}")
    run.log({"final/knn_cls_top1": knn})

    raw = probe_on_lake(train["tokens"], train["labels"], val["tokens"],
                        val["labels"], device, cb=None,
                        epochs=args.probe_epochs, tag="raw", wandb_run=run)
    run.log({"final/probe_raw_top1": raw})
    for k in args.probe_ks:
        acc = probe_on_lake(train["tokens"], train["labels"], val["tokens"],
                            val["labels"], device, cb=books[k],
                            epochs=args.probe_epochs, tag=f"q{k}",
                            wandb_run=run)
        run.log({f"final/probe_q{k}_top1": acc})
        print(f"[rate-distortion] raw {raw:.2f} vs K={k}: {acc:.2f}")

    # --- small verified artifact: codebooks + figures ---
    art = wandb.Artifact(f"vocab-{run.id}", type="vocabulary")
    art.add_dir(str(run_dir))
    run.log_artifact(art)
    art.wait()
    print("ARTIFACT_VERIFIED")
    run.finish()


if __name__ == "__main__":
    main()
