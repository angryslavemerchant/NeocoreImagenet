"""
QK-admission classifier (2026-07-19): admission BEFORE any mixing.

The globality-law fix (user design): no encoder touches the image tokens.
Each token's key is a pure per-token function of (code, position) —
key_proj(emb + pos), no attention, so keys are CONSTANT across rounds.
A bank of learned query heads scores tokens by dot product (max over
heads: admitted if any search template matches); exactly B are admitted
(hard detached top-k, (1+sigmoid) gate for scorer gradient) into a
6-block deep core that mixes ONLY the admitted set. The round's output
(+ mem marker) rejoins the candidate pool AND modulates the queries via
cross-attention (q <- norm(q + MHA(q, m, m))) — what the model has seen
shifts where it looks next. All global knowledge lives in the query
state and memory, both of which only ever passed through the bottleneck.

Arms: --arch dense (6 blocks over all 256 raw tokens — the mixing
reference at matched deep depth), or loop with policy learned|random and
modulate on|off. static (learned + no modulation) re-admits the same
top-B forever by construction — the learned-vs-static gap is exactly the
value of the query shift.

Instruments per round: retention (memory re-admission), coverage
(cumulative unique image positions), new_frac (fresh admissions),
shift_overlap (round-r picks vs the round-0 static ranking's top-(r+1)B
— 1.0 = modulation does nothing beyond next-in-line), query_drift
(cosine to round-0 queries), head_share entropy (query specialization).

wandb project: neocore-cls (one run per config; WANDB_RUN_ID popped —
run_training.sh exports a fixed one that merged the last sweep).
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


class QKClassifier(nn.Module):
    """Foveated classifier: QK admission over unmixed tokens."""

    def __init__(self, k_codes: int, arch: str = "loop",
                 policy: str = "learned", modulate: bool = True,
                 budget: int = 64, rounds: int = 4, d: int = 256,
                 heads: int = 8, n_query: int = 8, deep: int = 6,
                 n_classes: int = 100, explore_frac: float = 0.125):
        super().__init__()
        assert arch in ("loop", "dense") and policy in ("learned", "random")
        self.arch, self.policy, self.modulate = arch, policy, modulate
        self.budget, self.rounds, self.n_query = budget, rounds, n_query
        self.explore_frac = explore_frac
        self.emb = nn.Embedding(k_codes, d)
        self.pos = nn.Parameter(torch.randn(1, N_POS, d) * 0.02)
        self.mem_emb = nn.Parameter(torch.zeros(1, 1, d))
        self.key_proj = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d))
        self.query0 = nn.Parameter(torch.randn(1, n_query, d) * 0.02)
        self.q_update = nn.MultiheadAttention(d, heads, batch_first=True)
        self.q_norm = nn.LayerNorm(d)
        self.deep = nn.ModuleList(Block(d, heads) for _ in range(deep))
        self.pool_q = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.pool_attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.pool_norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, n_classes)
        self.scale = 1.0 / math.sqrt(d)

    def _run_deep(self, x):
        for b in self.deep:
            x = b(x)
        return x

    def _pool(self, tok):
        q = self.pool_q.expand(tok.shape[0], -1, -1)
        p, _ = self.pool_attn(q, tok, tok, need_weights=False)
        return self.head(self.pool_norm(p.squeeze(1)))

    def _score(self, q, pool):
        k = self.key_proj(pool)                          # (B,N,d) per-token
        s = torch.einsum("bqd,bnd->bqn", q, k) * self.scale
        return s.max(dim=1)                              # values (B,N), heads

    def forward(self, codes: torch.Tensor):
        """codes (B,256) long -> (logits, instruments dict)."""
        tok_img = self.emb(codes) + self.pos             # NEVER mixed
        if self.arch == "dense":
            return self._pool(self._run_deep(tok_img)), {}

        B = codes.shape[0]
        dev = codes.device
        q = self.query0.expand(B, -1, -1)
        mem = None
        seen = torch.zeros(B, N_POS, dtype=torch.bool, device=dev)
        static_rank = None                               # round-0 img ranking
        inst = {k: [] for k in ("retention", "coverage", "new_frac",
                                "shift_overlap", "query_drift",
                                "head_entropy")}
        for r in range(self.rounds):
            pool = tok_img if mem is None \
                else torch.cat([tok_img, mem], dim=1)
            s, s_head = self._score(q, pool)             # (B,N)
            if static_rank is None:
                static_rank = s[:, :N_POS].detach().argsort(
                    dim=1, descending=True)              # (B,256)
            if self.policy == "random":
                key = torch.rand_like(s)
            else:
                key = s.detach().float()
            n_exp = 0
            if (self.training and self.policy == "learned"
                    and self.explore_frac > 0):
                # epsilon-greedy: never-admitted tokens can't be discovered
                # through a gate they never pass — reserve slots for uniform
                # random picks (training only; eval is pure top-k).
                n_exp = max(1, int(self.budget * self.explore_frac))
            top_g = key.topk(self.budget - n_exp, dim=1).indices
            if n_exp:
                rnd = torch.rand_like(key)
                rnd.scatter_(1, top_g, -1.0)             # exclude greedy picks
                top_e = rnd.topk(n_exp, dim=1).indices
                top = torch.cat([top_g, top_e], dim=1)   # (B,budget)
            else:
                top = top_g
            sel = pool.gather(
                1, top.unsqueeze(-1).expand(-1, -1, pool.shape[-1]))
            gate = torch.sigmoid(s.gather(1, top)).unsqueeze(-1)
            m = self._run_deep(sel * (1 + gate))
            mem = m + self.mem_emb

            with torch.no_grad():
                is_mem = top >= N_POS
                img_idx = top.clamp(max=N_POS - 1)       # mem slots masked out
                hit = torch.zeros(B, N_POS, dtype=torch.bool, device=dev)
                hit.scatter_(1, img_idx, ~is_mem)
                new = (hit & ~seen).sum(1).float()
                n_img = (~is_mem).sum(1).float().clamp(min=1)
                inst["retention"].append(float(is_mem.float().mean()))
                inst["new_frac"].append(float((new / n_img).mean()))
                seen |= hit
                inst["coverage"].append(float(seen.float().mean()))
                static_top = static_rank[:, :(r + 1) * self.budget]
                in_static = torch.zeros(B, N_POS, dtype=torch.bool,
                                        device=dev)
                in_static.scatter_(1, static_top, True)
                ov = (hit & in_static).sum(1).float() / n_img
                inst["shift_overlap"].append(float(ov.mean()))
                inst["query_drift"].append(float(F.cosine_similarity(
                    q, self.query0.expand_as(q), dim=-1).mean()))
                heads_win = s_head.gather(1, top).flatten().float()
                counts = torch.bincount(
                    heads_win.long(), minlength=self.n_query).float()
                p = counts / counts.sum()
                ent = -(p[p > 0] * p[p > 0].log()).sum() \
                    / math.log(self.n_query)
                inst["head_entropy"].append(float(ent))

            if self.modulate:
                dq, _ = self.q_update(q, m, m, need_weights=False)
                q = self.q_norm(q + dq)
        return self._pool(mem), inst


# ---------------------------------------------------------------------------

def cfg_name(cfg):
    if cfg["arch"] == "dense":
        return "QK_dense6"
    tag = cfg["policy"] if cfg["policy"] == "random" else \
        ("learned" if cfg["modulate"] else "static")
    return f"QK_{tag}_B{cfg['budget']}_R{cfg['rounds']}"


def train_config(cfg, train_codes, train_labels, val_codes, val_labels,
                 k_codes, device, args):
    import wandb
    os.environ.pop("WANDB_RUN_ID", None)   # else all configs merge into one
    name = cfg_name(cfg)
    run = wandb.init(project=args.wandb_project, name=name,
                     config={**cfg, "epochs": args.num_epochs,
                             "batch": args.batch_size, "lr": args.lr},
                     reinit=True)
    torch.manual_seed(args.seed)
    model = QKClassifier(k_codes, **cfg).to(device)
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
                logits, _ = model(c)
                loss = F.cross_entropy(logits, y_tr[sel].to(device))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach()))
        sched.step()
        model.eval()
        correct = tot = 0
        insts = []
        with torch.no_grad():
            for i in range(0, val_codes.shape[0], args.batch_size):
                c = val_codes[i:i + args.batch_size].long().to(device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits, inst = model(c)
                pred = logits.argmax(1)
                yv = y_va[i:i + args.batch_size]
                correct += int((pred == yv).sum()); tot += len(yv)
                if inst:
                    insts.append(inst)
        acc = 100 * correct / tot
        best = max(best, acc)
        log = {"val_top1": acc, "train_loss": float(np.mean(losses)),
               "epoch": ep, "epoch_sec": time.time() - t0}
        line = ""
        if insts:
            for key in insts[0]:
                per_r = np.mean([b[key] for b in insts], axis=0)
                for r_i, v in enumerate(per_r):
                    log[f"{key}_r{r_i}"] = float(v)
            last = cfg["rounds"] - 1
            line = (f" cov {log['coverage_r%d' % last]:.2f}"
                    f" ret {log['retention_r%d' % last]:.2f}"
                    f" shift {log['shift_overlap_r%d' % last]:.2f}")
        run.log(log)
        if ep % 10 == 0 or ep == args.num_epochs - 1:
            print(f"[{name}] ep {ep + 1}/{args.num_epochs} "
                  f"top1 {acc:.2f} (best {best:.2f}){line} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    run.summary["best_top1"] = best
    run.finish()
    return name, best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--budgets", type=int, nargs="+", default=[16, 64])
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

    run_dir = Path("runs") / "QK_sweep"
    run_dir.mkdir(parents=True, exist_ok=True)
    (Path("runs") / "LATEST").write_text(str(run_dir))

    configs = [{"arch": "dense", "policy": "learned", "modulate": False,
                "budget": 0, "rounds": args.rounds}]
    for b in args.budgets:
        configs.append({"arch": "loop", "policy": "learned",
                        "modulate": True, "budget": b,
                        "rounds": args.rounds})
        configs.append({"arch": "loop", "policy": "learned",
                        "modulate": False, "budget": b,
                        "rounds": args.rounds})
        configs.append({"arch": "loop", "policy": "random",
                        "modulate": False, "budget": b,
                        "rounds": args.rounds})

    results = {}
    for cfg in configs:
        name, best = train_config(cfg, train_codes, train_labels,
                                  val_codes, val_labels, k_codes, device,
                                  args)
        results[name] = best
        print(f"=== {name}: best top1 {best:.2f}", flush=True)

    import wandb, json
    (run_dir / "summary.json").write_text(json.dumps(results, indent=1))
    os.environ.pop("WANDB_RUN_ID", None)
    run = wandb.init(project=args.wandb_project, name="QK_sweep_summary",
                     reinit=True)
    for k, v in results.items():
        run.summary[k] = v
    art = wandb.Artifact(f"qk-sweep-{run.id}", type="cls_sweep")
    art.add_dir(str(run_dir))
    run.log_artifact(art)
    art.wait()
    print("ARTIFACT_VERIFIED")
    run.finish()
    print("SWEEP_RESULTS " + json.dumps(results))


if __name__ == "__main__":
    main()
