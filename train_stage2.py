"""
Stage 2 of the vocabulary program: admission machinery over the frozen
codebook (2026-07-18; design in CLAUDE.md checkpoint).

The image is a 16x16 grid of code indices (frozen K=2048 codebook over
frozen DINOv2-S/14 tokens — both immovable; gradient descent can only
touch selection and arrangement). Memory admits either POSITIONS
(patches) or TYPES (codes; admitting a code covers its full spatial
support). After admission, a masked prediction pass must classify the
code index at every uncovered position — cross-entropy over vocabulary,
no continuous decoder to leak through.

Arms (--arm):
  T-learned   recursive scored admission of codes, exact-K/round
  T-random    K random codes among those present         (selection control)
  T-freq      K codes with largest within-image support  (counting control)
  P-learned   recursive scored admission of positions
  P-random    K random positions
  P-uniform   evenly spaced grid positions
Each run sweeps --budgets (fresh model per budget) so cross-arm
comparisons are rec-vs-coverage CURVES, not single points.

Leak audit: prediction-pass input at uncovered positions is a mask
embedding + position only; verified by the invariance smoke test
(changing an uncovered position's code cannot change any logit).
"""

import argparse
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

N_POS = 256          # 16x16 grid
GRID = 16

ARMS = ["T-learned", "T-random", "T-freq",
        "P-learned", "P-random", "P-uniform"]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class Block(nn.Module):
    def __init__(self, d, heads, mlp_ratio):
        super().__init__()
        self.n1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.n2 = nn.LayerNorm(d)
        h = int(d * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Linear(h, d))

    def forward(self, x):
        a, _ = self.attn(self.n1(x), self.n1(x), self.n1(x), need_weights=False)
        x = x + a
        return x + self.mlp(self.n2(x))


class VocabNeocore(nn.Module):
    """Admission + masked code prediction over a frozen symbol grid."""

    def __init__(self, k_codes: int, arm: str, budget: int, rounds: int = 4,
                 d: int = 256, blocks: int = 8, heads: int = 8,
                 mlp_ratio: float = 3.0, codebook: torch.Tensor = None):
        super().__init__()
        assert arm in ARMS
        self.k_codes, self.arm, self.budget = k_codes, arm, budget
        self.rounds = rounds if arm.endswith("learned") else 1
        self.emb = nn.Embedding(k_codes, d)
        self.pos = nn.Parameter(torch.randn(1, N_POS, d) * 0.02)
        self.marker = nn.Parameter(torch.zeros(1, 1, d))
        self.mask_emb = nn.Parameter(torch.zeros(1, 1, d))
        self.core = nn.ModuleList(Block(d, heads, mlp_ratio)
                                  for _ in range(blocks))
        self.norm = nn.LayerNorm(d)
        self.score_head = nn.Linear(d, 1)
        self.pred_head = nn.Linear(d, k_codes)
        if arm == "P-uniform":
            self.register_buffer("uniform_idx", self._uniform_idx(budget))
        if codebook is not None:      # frozen; for soft-accuracy metric only
            self.register_buffer("codebook", codebook.float())

    @staticmethod
    def _uniform_idx(b: int) -> torch.Tensor:
        """b maximally-spread positions on the 16x16 grid."""
        idx = torch.linspace(0, N_POS - 1, b).round().long()
        idx = torch.unique(idx)
        fill = 0
        while idx.numel() < b:       # dedupe fallback: densify from start
            if fill not in idx:
                idx = torch.cat([idx, torch.tensor([fill])])
            fill += 1
        return idx.sort().values[:b]

    def _run_core(self, x):
        for blk in self.core:
            x = blk(x)
        return self.norm(x)

    # --- admission policies -------------------------------------------------
    @torch.no_grad()
    def _covered_control(self, codes, generator=None):
        """Admitted/covered masks for the non-learned arms."""
        B = codes.shape[0]
        dev = codes.device
        if self.arm == "P-uniform":
            covered = torch.zeros(B, N_POS, dtype=torch.bool, device=dev)
            covered[:, self.uniform_idx] = True
            return covered, None
        if self.arm == "P-random":
            r = torch.rand(B, N_POS, device=dev, generator=generator)
            idx = r.topk(self.budget, dim=1).indices
            covered = torch.zeros(B, N_POS, dtype=torch.bool, device=dev)
            covered.scatter_(1, idx, True)
            return covered, None
        # type arms: counts (B, K)
        one = torch.ones_like(codes, dtype=torch.float)
        counts = torch.zeros(B, self.k_codes, device=dev) \
            .scatter_add_(1, codes, one)
        present = counts > 0
        if self.arm == "T-freq":
            key = counts
        else:                          # T-random among present codes
            key = torch.rand(B, self.k_codes, device=dev,
                             generator=generator) * present.float()
        key = key.masked_fill(~present, -1.0)
        idx = key.topk(self.budget, dim=1).indices          # (B, budget)
        admitted = torch.zeros(B, self.k_codes, dtype=torch.bool, device=dev)
        admitted.scatter_(1, idx, True)
        admitted &= present            # images with < budget codes: all in
        covered = admitted.gather(1, codes)
        return covered, admitted

    def _admit_learned(self, codes):
        """R rounds: score -> exact-K/round detached top-k -> mark -> re-run.
        Returns covered mask, per-position final scores (for the gate) and
        per-round admission map (for instruments)."""
        B = codes.shape[0]
        dev = codes.device
        tok0 = self.emb(codes) + self.pos
        covered = torch.zeros(B, N_POS, dtype=torch.bool, device=dev)
        admitted_codes = torch.zeros(B, self.k_codes, dtype=torch.bool,
                                     device=dev)
        admit_round = torch.full((B, N_POS), -1, dtype=torch.long,
                                 device=dev)
        per_round = [self.budget // self.rounds] * self.rounds
        per_round[-1] += self.budget - sum(per_round)
        scores = None
        for r, k_r in enumerate(per_round):
            x = tok0 + covered.unsqueeze(-1) * self.marker
            h = self._run_core(x)
            scores = self.score_head(h).squeeze(-1)          # (B, N_POS)
            with torch.no_grad():
                if self.arm.startswith("T"):
                    one = torch.ones_like(codes, dtype=torch.float)
                    counts = torch.zeros(B, self.k_codes, device=dev) \
                        .scatter_add_(1, codes, one)
                    present = counts > 0
                    cs = torch.zeros(B, self.k_codes, device=dev) \
                        .scatter_add_(1, codes, scores.detach().float())
                    cs = cs.masked_fill(~present | admitted_codes,
                                        float("-inf"))
                    k_eff = min(k_r, int((~admitted_codes & present)
                                         .sum(1).min()))
                    if k_eff > 0:
                        idx = cs.topk(k_eff, dim=1).indices
                        admitted_codes.scatter_(1, idx, True)
                    admitted_codes &= present
                    new_cov = admitted_codes.gather(1, codes) & ~covered
                else:
                    s = scores.detach().masked_fill(covered, float("-inf"))
                    idx = s.topk(k_r, dim=1).indices
                    new_cov = torch.zeros_like(covered)
                    new_cov.scatter_(1, idx, True)
                    new_cov = new_cov & ~covered
                # out-of-place: `covered` is saved by the marker multiply's
                # backward; in-place |= breaks autograd
                admit_round = torch.where(new_cov,
                                          torch.full_like(admit_round, r),
                                          admit_round)
                covered = covered | new_cov
        return covered, scores, admit_round

    # --- forward ------------------------------------------------------------
    def forward(self, codes: torch.Tensor, generator=None):
        """codes: (B, 256) long. Returns (loss, metrics dict)."""
        if self.arm.endswith("learned"):
            covered, scores, admit_round = self._admit_learned(codes)
            gate = torch.sigmoid(scores)                     # scorer gradient
        else:
            covered, _ = self._covered_control(codes, generator)
            gate, admit_round = None, None

        tok = self.emb(codes) + self.pos
        if gate is not None:
            tok = tok * (1 + gate.unsqueeze(-1) * covered.unsqueeze(-1))
        inp = torch.where(covered.unsqueeze(-1),
                          tok + self.marker,
                          self.mask_emb + self.pos.expand_as(tok))
        h = self._run_core(inp)
        logits = self.pred_head(h)                           # (B, P, K)

        uncov = ~covered
        n_unc = uncov.sum()
        if n_unc == 0:
            loss = logits.sum() * 0.0
        else:
            loss = F.cross_entropy(logits[uncov], codes[uncov])
        with torch.no_grad():
            m = {"coverage": covered.float().mean().item()}
            if n_unc > 0:
                pred = logits[uncov].argmax(-1)
                true = codes[uncov]
                m["top1"] = (pred == true).float().mean().item()
                if hasattr(self, "codebook"):
                    m["soft_cos"] = F.cosine_similarity(
                        self.codebook[pred], self.codebook[true], dim=1
                    ).mean().item()
        return loss, m, covered, admit_round


# ---------------------------------------------------------------------------
# Data: the lake as a code grid
# ---------------------------------------------------------------------------

def assign_lake(tokens_path: Path, codebook: torch.Tensor, device,
                out_path: Path):
    """Quantize a token lake split to (N, 256) int16 code grids (cached)."""
    if out_path.exists():
        d = torch.load(out_path, map_location="cpu")
        return d["codes"], d["labels"]
    lake = torch.load(tokens_path, map_location="cpu", mmap=True)
    toks, labels = lake["tokens"], lake["labels"]
    cb = codebook.to(device).float()
    cb_sq = cb.pow(2).sum(1)
    n = toks.shape[0]
    codes = torch.empty(n, N_POS, dtype=torch.int16)
    for i in range(0, n, 512):
        t = toks[i:i + 512].to(device).float().reshape(-1, cb.shape[1])
        d2 = t.pow(2).sum(1, keepdim=True) - 2 * t @ cb.T + cb_sq[None, :]
        codes[i:i + 512] = d2.argmin(1).reshape(-1, N_POS).to(torch.int16) \
            .cpu()
    torch.save({"codes": codes, "labels": labels.clone()}, out_path)
    print(f"[codes] wrote {out_path.name}: {tuple(codes.shape)}")
    return codes, labels


def get_codebook(args, device) -> torch.Tensor:
    local = Path(args.codebook_file)
    if local.exists():
        return torch.load(local, map_location="cpu")
    import wandb
    art = wandb.Api().artifact(args.codebook_artifact)
    d = Path(art.download())
    return torch.load(d / local.name, map_location="cpu")


# ---------------------------------------------------------------------------
# Instruments
# ---------------------------------------------------------------------------

@torch.no_grad()
def admission_figure(model, codes, val_blob_path, out_png, device, n=8):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    blob = torch.load(val_blob_path, map_location="cpu", mmap=True)
    g = torch.Generator().manual_seed(5)
    sel = torch.randperm(codes.shape[0], generator=g)[:n]
    c = codes[sel].long().to(device)
    gen = torch.Generator(device=device).manual_seed(0)
    _, _, covered, admit_round = model(c, generator=gen)
    fig, axes = plt.subplots(2, n, figsize=(2.2 * n, 4.6))
    for j in range(n):
        img = blob["images"][int(sel[j])].permute(1, 2, 0).numpy()
        axes[0, j].imshow(img); axes[0, j].axis("off")
        if admit_round is not None:
            mp = admit_round[j].reshape(GRID, GRID).cpu().float()
            mp[mp < 0] = float("nan")
            axes[1, j].imshow(mp, cmap="viridis", interpolation="nearest",
                              vmin=0, vmax=max(model.rounds - 1, 1))
        else:
            axes[1, j].imshow(covered[j].reshape(GRID, GRID).cpu(),
                              cmap="gray", interpolation="nearest")
        axes[1, j].axis("off")
    fig.suptitle(f"{model.arm} B={model.budget} — covered/admission-round")
    fig.tight_layout(); fig.savefig(out_png, dpi=130); plt.close(fig)


@torch.no_grad()
def freq_overlap(model, codes, device, n_batches=5, batch=512):
    """T-learned only: fraction of admitted codes shared with the
    frequency-ranked set (is the scorer just counting?)."""
    if model.arm != "T-learned":
        return None
    overl = []
    for i in range(n_batches):
        c = codes[i * batch:(i + 1) * batch].long().to(device)
        covered, _, _ = model._admit_learned(c)
        one = torch.ones_like(c, dtype=torch.float)
        counts = torch.zeros(c.shape[0], model.k_codes, device=device) \
            .scatter_add_(1, c, one)
        freq_idx = counts.topk(model.budget, dim=1).indices
        freq_adm = torch.zeros(c.shape[0], model.k_codes, dtype=torch.bool,
                               device=device).scatter_(1, freq_idx, True)
        # learned admitted set = unique codes among covered positions;
        # scatter uncovered into a sentinel bin K, then drop it
        sent = torch.where(covered, c, torch.full_like(c, model.k_codes))
        adm = torch.zeros(c.shape[0], model.k_codes + 1, dtype=torch.bool,
                          device=device).scatter_(1, sent, True)[:, :-1]
        inter = (adm & freq_adm).sum(1).float()
        overl.append((inter / model.budget).mean().item())
    return float(np.mean(overl))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_arm(args, codebook, train_codes, train_labels, val_codes,
              val_labels, budget, device, run):
    torch.manual_seed(args.seed)
    model = VocabNeocore(codebook.shape[0], args.arm, budget,
                         rounds=args.rounds, codebook=codebook).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{args.arm} B={budget}] params {n_params / 1e6:.2f}M "
          f"rounds={model.rounds}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=0.05)
    total = args.num_epochs
    warm = torch.optim.lr_scheduler.LinearLR(opt, 1e-2, 1.0, 2)
    cos = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total - 2)
    sched = torch.optim.lr_scheduler.SequentialLR(opt, [warm, cos], [2])
    n = train_codes.shape[0]
    tag = f"B{budget}"
    best = {"top1": 0.0}
    for ep in range(total):
        model.train()
        order = torch.randperm(n)
        t0, losses = time.time(), []
        for i in range(0, n, args.batch_size):
            c = train_codes[order[i:i + args.batch_size]].long().to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, m, _, _ = model(c)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach()))
        sched.step()
        model.eval()
        vm, vloss = [], []
        gen = torch.Generator(device=device).manual_seed(123)
        with torch.no_grad():
            for i in range(0, val_codes.shape[0], args.batch_size):
                c = val_codes[i:i + args.batch_size].long().to(device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss, m, _, _ = model(c, generator=gen)
                vloss.append(float(loss)); vm.append(m)
        top1 = float(np.mean([m.get("top1", 0) for m in vm]))
        cov = float(np.mean([m["coverage"] for m in vm]))
        soft = float(np.mean([m.get("soft_cos", 0) for m in vm]))
        if top1 > best["top1"]:
            best = {"top1": top1, "soft": soft, "cov": cov,
                    "loss": float(np.mean(vloss)), "epoch": ep}
        run.log({f"{tag}/train_loss": float(np.mean(losses)),
                 f"{tag}/val_loss": float(np.mean(vloss)),
                 f"{tag}/val_top1": top1, f"{tag}/val_soft_cos": soft,
                 f"{tag}/coverage": cov, f"{tag}/epoch": ep,
                 f"{tag}/epoch_sec": time.time() - t0})
        if ep % 5 == 0 or ep == total - 1:
            print(f"[{args.arm} B={budget}] ep {ep + 1}/{total} "
                  f"loss {np.mean(vloss):.4f} top1 {top1:.4f} "
                  f"soft {soft:.4f} cov {cov:.3f} "
                  f"({time.time() - t0:.0f}s)")
    return model, best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", type=str, required=True, choices=ARMS)
    p.add_argument("--budgets", type=int, nargs="+", default=None,
                   help="default: 8 16 32 for T arms, 25 49 98 for P arms")
    p.add_argument("--rounds", type=int, default=4)
    p.add_argument("--num_epochs", type=int, default=40)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--codebook_artifact", type=str,
                   default="luckymushy-individual/neocore/"
                           "vocab-6duv9qzw:v0")
    p.add_argument("--codebook_file", type=str, default="codebook_k2048.pt")
    p.add_argument("--lake_dir", type=str, default="./data/dino_lake")
    p.add_argument("--jpeg_cache_dir",    type=str, default="./jpeg_cache")
    p.add_argument("--dataset_name",      type=str,
                   default="clane9/imagenet-100")
    p.add_argument("--dataset_cache_dir", type=str, default="./data")
    p.add_argument("--wandb_project", type=str, default="neocore")
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if args.budgets is None:
        args.budgets = [8, 16, 32] if args.arm.startswith("T") \
            else [25, 49, 98]
    if args.run_name is None:
        args.run_name = f"S2_{args.arm.replace('-', '')}"
    device = "cuda"

    import wandb
    run = wandb.init(project=args.wandb_project, name=args.run_name,
                     config=vars(args))
    run_dir = Path("runs") / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (Path("runs") / "LATEST").write_text(str(run_dir))

    # lake (rebuild if missing) -> codebook -> code grids
    lake = Path(args.lake_dir)
    if not (lake / "train_tokens.pt").exists():
        import train_vocab as tv
        tv.stage0(args, device)
    codebook = get_codebook(args, device)
    print(f"[codes] codebook {tuple(codebook.shape)}")
    train_codes, train_labels = assign_lake(
        lake / "train_tokens.pt", codebook, device,
        lake / f"train_codes_k{codebook.shape[0]}.pt")
    val_codes, val_labels = assign_lake(
        lake / "val_tokens.pt", codebook, device,
        lake / f"val_codes_k{codebook.shape[0]}.pt")

    val_blob = Path(args.jpeg_cache_dir) / "ram256" / "validation.pt"
    summary = {}
    for budget in args.budgets:
        model, best = train_arm(args, codebook, train_codes, train_labels,
                                val_codes, val_labels, budget, device, run)
        summary[budget] = best
        run.log({f"final/B{budget}_top1": best["top1"],
                 f"final/B{budget}_soft": best.get("soft", 0),
                 f"final/B{budget}_coverage": best.get("cov", 0)})
        admission_figure(model, val_codes, val_blob,
                         run_dir / f"admission_B{budget}.png", device)
        fo = freq_overlap(model, val_codes.long(), device)
        if fo is not None:
            run.log({f"final/B{budget}_freq_overlap": fo})
            print(f"[{args.arm} B={budget}] freq_overlap {fo:.3f}")
        torch.save(model.state_dict(), run_dir / f"model_B{budget}.pt")
        print(f"[{args.arm} B={budget}] BEST top1 {best['top1']:.4f} "
              f"soft {best.get('soft', 0):.4f} cov {best.get('cov', 0):.3f}")

    art = wandb.Artifact(f"stage2-{args.arm}-{run.id}", type="stage2")
    art.add_dir(str(run_dir))
    run.log_artifact(art)
    art.wait()
    print("ARTIFACT_VERIFIED")
    run.finish()


if __name__ == "__main__":
    main()
