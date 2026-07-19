"""
Vocabulary classification sweep (2026-07-19): how many DINO codes does
ImageNet-100 need, and does recursive selective memory help?

Architecture (user design): codes -> embedding + pos (256 tokens). Per
round: 2 shallow blocks attend over [256 image tokens + M memory tokens]
(workspace broadcast: memory conditions perception); a score head ranks
the whole pool; exactly B tokens are admitted (hard detached top-k,
(1+sigmoid) gate for scorer gradient) into a 6-block deep core that MIXES
them; outputs + mem embedding become next round's memory tokens. Memory
competes with image tokens for admission — retention is LEARNED, and
logged per round. After R rounds: attentive pool over the final B memory
tokens -> 100-way CE.

Mixing is legal again: classification can't be gamed through a decoder.
Deep compute lives inside the bottleneck (6 blocks x B tokens), so the
loop is cheap even at R=4.

Arms: --arch dense (same 2+6 blocks over all 256, no admission — the
saturation reference), or --arch loop with --policy learned|random.
Budgets swept into the starved regime (4/8/16/64) where selection can
actually bind.

wandb project: neocore-cls (one run per config — dashboard stays legible).
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

N_POS = 256


class Block(nn.Module):
    def __init__(self, d, heads, mlp_ratio=3.0):
        super().__init__()
        self.n1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.n2 = nn.LayerNorm(d)
        h = int(d * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Linear(h, d))

    def forward(self, x):
        y = self.n1(x)
        a, _ = self.attn(y, y, y, need_weights=False)
        x = x + a
        return x + self.mlp(self.n2(x))


class VocabClassifier(nn.Module):
    """Recursive selective-workspace classifier over frozen code grids."""

    def __init__(self, k_codes: int, arch: str = "loop",
                 policy: str = "learned", budget: int = 8, rounds: int = 4,
                 d: int = 256, heads: int = 8, shallow: int = 2,
                 deep: int = 6, n_classes: int = 100):
        super().__init__()
        assert arch in ("loop", "dense") and policy in ("learned", "random")
        self.arch, self.policy = arch, policy
        self.budget, self.rounds = budget, rounds
        self.emb = nn.Embedding(k_codes, d)
        self.pos = nn.Parameter(torch.randn(1, N_POS, d) * 0.02)
        self.mem_emb = nn.Parameter(torch.zeros(1, 1, d))
        self.shallow = nn.ModuleList(Block(d, heads) for _ in range(shallow))
        self.deep = nn.ModuleList(Block(d, heads) for _ in range(deep))
        self.score_head = nn.Linear(d, 1)
        self.pool_q = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.pool_attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.pool_norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, n_classes)

    def _run(self, blocks, x):
        for b in blocks:
            x = b(x)
        return x

    def _pool(self, tok):
        q = self.pool_q.expand(tok.shape[0], -1, -1)
        p, _ = self.pool_attn(q, tok, tok, need_weights=False)
        return self.head(self.pool_norm(p.squeeze(1)))

    def forward(self, codes: torch.Tensor):
        """codes (B,256) long -> (logits, instruments dict)."""
        tok_img = self.emb(codes) + self.pos
        if self.arch == "dense":
            h = self._run(self.deep, self._run(self.shallow, tok_img))
            return self._pool(h), {"retention": [0.0] * self.rounds}

        B = codes.shape[0]
        mem = None
        retention = []
        for r in range(self.rounds):
            pool_in = tok_img if mem is None \
                else torch.cat([tok_img, mem], dim=1)
            h = self._run(self.shallow, pool_in)
            s = self.score_head(h).squeeze(-1)              # (B, 256[+M])
            if self.policy == "random":
                key = torch.rand_like(s)
            else:
                key = s.detach()
            top = key.topk(self.budget, dim=1).indices      # (B, budget)
            sel = h.gather(1, top.unsqueeze(-1).expand(-1, -1, h.shape[-1]))
            gate = torch.sigmoid(
                s.gather(1, top)).unsqueeze(-1)             # scorer grad
            m = self._run(self.deep, sel * (1 + gate))
            mem = m + self.mem_emb
            retention.append(float((top >= N_POS).float().mean()))
        return self._pool(mem), {"retention": retention}


# ---------------------------------------------------------------------------

def train_config(cfg, train_codes, train_labels, val_codes, val_labels,
                 k_codes, device, args):
    import wandb
    name = (f"CLS_{cfg['arch']}" if cfg["arch"] == "dense" else
            f"CLS_{cfg['policy']}_B{cfg['budget']}_R{cfg['rounds']}")
    run = wandb.init(project=args.wandb_project, name=name,
                     config={**cfg, "epochs": args.num_epochs,
                             "batch": args.batch_size, "lr": args.lr},
                     reinit=True)
    torch.manual_seed(args.seed)
    model = VocabClassifier(k_codes, **cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=0.05)
    warm = torch.optim.lr_scheduler.LinearLR(opt, 1e-2, 1.0, 2)
    cos = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.num_epochs - 2)
    sched = torch.optim.lr_scheduler.SequentialLR(opt, [warm, cos], [2])
    n = train_codes.shape[0]
    y_tr = train_labels.long()
    y_va = val_labels.long().to(device)
    best = 0.0
    for ep in range(args.num_epochs):
        model.train()
        order = torch.randperm(n)
        t0, losses = time.time(), []
        for i in range(0, n, args.batch_size):
            sel = order[i:i + args.batch_size]
            c = train_codes[sel].long().to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, inst = model(c)
                loss = F.cross_entropy(logits, y_tr[sel].to(device))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach()))
        sched.step()
        model.eval()
        correct = tot = 0
        rets = []
        with torch.no_grad():
            for i in range(0, val_codes.shape[0], args.batch_size):
                c = val_codes[i:i + args.batch_size].long().to(device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits, inst = model(c)
                pred = logits.argmax(1)
                yv = y_va[i:i + args.batch_size]
                correct += int((pred == yv).sum()); tot += len(yv)
                rets.append(inst["retention"])
        acc = 100 * correct / tot
        best = max(best, acc)
        ret_mean = np.mean(np.array(rets), axis=0).tolist()
        log = {"val_top1": acc, "train_loss": float(np.mean(losses)),
               "epoch": ep, "epoch_sec": time.time() - t0}
        for r_i, rv in enumerate(ret_mean):
            log[f"retention_r{r_i}"] = rv
        run.log(log)
        if ep % 10 == 0 or ep == args.num_epochs - 1:
            print(f"[{name}] ep {ep + 1}/{args.num_epochs} "
                  f"top1 {acc:.2f} (best {best:.2f}) "
                  f"ret {['%.2f' % r for r in ret_mean]} "
                  f"({time.time() - t0:.0f}s)")
    run.summary["best_top1"] = best
    run.finish()
    return name, best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--budgets", type=int, nargs="+", default=[4, 8, 16, 64])
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
    p.add_argument("--wandb_project", type=str, default="neocore-cls")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    device = "cuda"

    from train_stage2 import assign_lake, get_codebook
    lake = Path(args.lake_dir)
    if not (lake / "train_tokens.pt").exists():
        import train_vocab as tv
        tv.stage0(args, device)
    codebook = get_codebook(args, device)
    k_codes = codebook.shape[0]
    train_codes, train_labels = assign_lake(
        lake / "train_tokens.pt", codebook, device,
        lake / f"train_codes_k{k_codes}.pt")
    val_codes, val_labels = assign_lake(
        lake / "val_tokens.pt", codebook, device,
        lake / f"val_codes_k{k_codes}.pt")

    run_dir = Path("runs") / "CLS_sweep"
    run_dir.mkdir(parents=True, exist_ok=True)
    (Path("runs") / "LATEST").write_text(str(run_dir))

    configs = [{"arch": "dense", "policy": "learned", "budget": 0,
                "rounds": args.rounds}]
    for b in args.budgets:
        for pol in ("learned", "random"):
            configs.append({"arch": "loop", "policy": pol, "budget": b,
                            "rounds": args.rounds})
    # recursion control at the sweet-spot budget
    for pol in ("learned", "random"):
        configs.append({"arch": "loop", "policy": pol, "budget": 8,
                        "rounds": 1})

    results = {}
    for cfg in configs:
        name, best = train_config(cfg, train_codes, train_labels,
                                  val_codes, val_labels, k_codes, device,
                                  args)
        results[name] = best
        print(f"=== {name}: best top1 {best:.2f}")

    import wandb, json
    (run_dir / "summary.json").write_text(json.dumps(results, indent=1))
    run = wandb.init(project=args.wandb_project, name="CLS_sweep_summary",
                     reinit=True)
    for k, v in results.items():
        run.summary[k] = v
    art = wandb.Artifact(f"cls-sweep-{run.id}", type="cls_sweep")
    art.add_dir(str(run_dir))
    run.log_artifact(art)
    art.wait()
    print("ARTIFACT_VERIFIED")
    run.finish()
    print("SWEEP_RESULTS " + json.dumps(results))


if __name__ == "__main__":
    main()
