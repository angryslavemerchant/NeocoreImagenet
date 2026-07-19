"""
Episodic label-binding over the code lake (2026-07-19): the first task
in this project where the answer CANNOT be smeared.

Episode: 6 stream images of 6 distinct classes, EACH carrying one label
token (a learned embedding for a per-episode random label permutation —
the harder "labels on everything" default: no stimulus-level relevance
cue, every image is a potential target). After EVERY step, a probe query
(a DIFFERENT image of a class seen so far): predict ITS episode label.
Probes read memory but never write it. Dense reward is a credit-
assignment necessity, smoke-verified: with only a final query, retention
is a multi-hop chain of ~3%-probability exploration events and no arm
ever leaves chance. The headline metric is the FINAL probe (any of the
6 slots, full horizon). Chance = 1/6.
DINO codes give class recognition in-weights for free, but the class ->
label binding exists only inside the episode, attached to one token that
scrolled past under a memory budget. No retrieval = chance, exactly.

Model: QK admission as per-step PERCEPTION BANDWIDTH over an
ACCUMULATING memory (the only memory regime that has ever worked in
this project; per-step re-competition was smoke-tested here and
reproduced the reselect failure — retention becomes a chain of
exploration events that never bootstraps). Per step: pure per-token QK
scores over the image's 257 tokens (max over 8 query heads, no mixing
before admission); the label token is ALWAYS admitted (supervision is
architecturally salient, identical across arms) + B-1 chosen content
tokens (epsilon-greedy in training, pure top-k at eval); the deep core
encodes the admitted set in the context of current memory and their
outputs are APPENDED (+mem marker) — sticky, nothing re-competes.
Memory modulates the queries (learned arm): look for what distinguishes
THIS image given what is already stored. Probes admit B content tokens
of the query image and read ALL of memory (memory != perception).
Selection is therefore purely: which tokens best represent each image
for later retrieval — foveation with a persistent episodic store.

Class holdout: 80 train classes / 20 val classes (episodes at val are
over classes never episodically trained; codes/vocab shared).

Instruments: per-probe accuracy (acc_probe_s — retrieval vs horizon),
final-probe accuracy by target slot (acc_slot_s — primacy/recency).

Arms: dense (full attention over the concatenated episode, segment
embeddings — the no-bottleneck ceiling) and {learned, static, random}
x B{4,16}. wandb project: neocore-icl.
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
N_STREAM = 6
N_LABELS = 6


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


class EpisodeSampler:
    """GPU-resident episode construction by pure indexing over code grids."""

    def __init__(self, codes, labels, class_ids, device):
        self.device = device
        self.codes = codes.to(device)                    # (N,256) int16
        labels = labels.long()
        self.class_ids = class_ids.long().to(device)     # eligible classes
        idx_lists = [torch.nonzero(labels == int(c), as_tuple=True)[0]
                     for c in class_ids]
        self.counts = torch.tensor([len(t) for t in idx_lists],
                                   device=device)
        assert int(self.counts.min()) >= 2, "need >=2 imgs per class"
        pad = torch.zeros(len(idx_lists), int(self.counts.max()),
                          dtype=torch.long)
        for i, t in enumerate(idx_lists):
            pad[i, :len(t)] = t
        self.pad = pad.to(device)

    def sample(self, n_ep, generator=None):
        dev, g = self.device, generator
        nc = len(self.counts)
        slot_cls = torch.rand(n_ep, nc, device=dev,
                              generator=g).argsort(1)[:, :N_STREAM]
        cnt = self.counts[slot_cls]                       # (E,6)
        r = (torch.rand(n_ep, N_STREAM, device=dev,
                        generator=g) * cnt).long()
        r = torch.minimum(r, cnt - 1)
        img_idx = self.pad[slot_cls, r]                   # (E,6) lake rows
        stream_codes = self.codes[img_idx].long()         # (E,6,256)
        perm = torch.rand(n_ep, N_STREAM, device=dev,
                          generator=g).argsort(1)         # episode labels
        # a probe query after EVERY step s, targeting a slot <= s (dense
        # reward: 1-hop retention chains bootstrap longer ones); the final
        # step's probe targets any of the 6 slots — the headline metric.
        ar = torch.arange(n_ep, device=dev)
        targets = (torch.rand(n_ep, N_STREAM, device=dev, generator=g)
                   * torch.arange(1, N_STREAM + 1, device=dev)).long()
        t_cls = slot_cls.gather(1, targets)               # (E,6)
        t_cnt = self.counts[t_cls]
        r_sup = r.gather(1, targets)
        r2 = (torch.rand(n_ep, N_STREAM, device=dev,
                         generator=g) * (t_cnt - 1).clamp(min=1)).long()
        r2 = r2 + (r2 >= r_sup).long()                    # distinct image
        r2 = torch.minimum(r2, t_cnt - 1)
        query_codes = self.codes[self.pad[t_cls, r2]].long()  # (E,6,256)
        y = perm.gather(1, targets)                       # (E,6)
        return stream_codes, perm, query_codes, y, targets


class ICLModel(nn.Module):
    """Foveated episodic binder: QK admission with memory across a stream."""

    def __init__(self, k_codes: int, arch: str = "loop",
                 policy: str = "learned", modulate: bool = True,
                 budget: int = 16, d: int = 256, heads: int = 8,
                 n_query: int = 8, deep: int = 6,
                 explore_frac: float = 0.125):
        super().__init__()
        assert arch in ("loop", "dense") and policy in (
            "learned", "random", "oracle")
        # oracle: caller sets self.oracle_fn(codes)->(E,N) score mask —
        # rank-by-known-informativeness reference (toy: signature tokens;
        # real data: mutual information I(code; class) from the lake).
        self.oracle_fn = None
        self.arch, self.policy, self.modulate = arch, policy, modulate
        self.budget, self.n_query = budget, n_query
        self.explore_frac = explore_frac
        self.emb = nn.Embedding(k_codes, d)
        self.pos = nn.Parameter(torch.randn(1, N_POS, d) * 0.02)
        self.label_emb = nn.Embedding(N_LABELS, d)
        self.label_pos = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.mem_emb = nn.Parameter(torch.zeros(1, 1, d))
        self.seg_emb = nn.Parameter(
            torch.randn(1, N_STREAM + 1, 1, d) * 0.02)   # dense only
        self.key_proj = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d))
        self.query0 = nn.Parameter(torch.randn(1, n_query, d) * 0.02)
        self.q_update = nn.MultiheadAttention(d, heads, batch_first=True)
        self.q_norm = nn.LayerNorm(d)
        self.deep = nn.ModuleList(Block(d, heads) for _ in range(deep))
        self.pool_q = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.pool_attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.pool_norm = nn.LayerNorm(d)
        self.read_norm = nn.LayerNorm(d)
        # cosine read with learnable temperature: raw dot products against
        # unnormalized residual-stream bindings saturate the softmax at
        # init (conf 1.0 -> zero gradient -> read frozen; smoke-verified)
        self.read_temp = nn.Parameter(torch.tensor(10.0))
        self.head = nn.Linear(d, N_LABELS)
        self.scale = 1.0 / math.sqrt(d)

    def _run_deep(self, x):
        for b in self.deep:
            x = b(x)
        return x

    def _pool_vec(self, tok):
        q = self.pool_q.expand(tok.shape[0], -1, -1)
        p, _ = self.pool_attn(q, tok, tok, need_weights=False)
        return self.pool_norm(p.squeeze(1))

    def _pool(self, tok):
        return self.head(self._pool_vec(tok))

    def _score(self, q, pool):
        k = self.key_proj(pool)                          # per-token, no mix
        s = torch.einsum("bqd,bnd->bqn", q, k) * self.scale
        return s.max(dim=1).values

    def _admit(self, s, force_idx=None, mem_start=None, oracle_key=None):
        # force_idx: supervision tokens are architecturally salient — the
        # current step's label token is ALWAYS admitted (counts against B,
        # identical for every arm). Without it the label is a 1/257 needle
        # payable only through a multi-step grab->retain->retrieve chain
        # that no gradient can bootstrap (smoke-verified); with it, the
        # experiment measures what we actually ask: retention and eviction.
        if self.policy == "random":
            key = torch.rand_like(s)
        elif self.policy == "oracle":
            key = oracle_key + torch.rand_like(s) * 1e-3   # tie-break
        else:
            key = s.detach().float()
        n_force = 0
        if force_idx is not None:
            n_force = force_idx.shape[1]
            key = key.scatter(1, force_idx, float("-inf"))
        n_exp = 0
        if (self.training and self.policy == "learned"
                and self.explore_frac > 0):
            n_exp = max(1, int(self.budget * self.explore_frac))
            if mem_start is not None:
                n_exp = max(2, n_exp)   # room for a memory-biased slot
        top_g = key.topk(self.budget - n_force - n_exp, dim=1).indices
        parts = [top_g]
        if force_idx is not None:
            parts.insert(0, force_idx)
        if n_exp:
            rnd = torch.rand_like(key)
            rnd.scatter_(1, top_g, -1.0)
            if force_idx is not None:
                rnd.scatter_(1, force_idx, -1.0)
            if mem_start is not None and n_exp >= 2:
                # half the explore slots sample memory specifically:
                # retention chains can't bootstrap from uniform exploration
                # alone (p(mem) ~ B/N per hop — smoke-verified dead)
                n_me = n_exp // 2
                rnd_m = rnd.clone()
                rnd_m[:, :mem_start] = -1.0
                top_m = rnd_m.topk(n_me, dim=1).indices
                rnd.scatter_(1, top_m, -1.0)
                parts += [top_m, rnd.topk(n_exp - n_me, dim=1).indices]
            else:
                parts.append(rnd.topk(n_exp, dim=1).indices)
        return torch.cat(parts, dim=1)

    def _step_tokens(self, codes, ep_label=None):
        tok = self.emb(codes) + self.pos                 # (E,256,d)
        if ep_label is None:
            return tok
        lt = self.label_emb(ep_label).unsqueeze(1) + self.label_pos
        return torch.cat([tok, lt], dim=1)               # (E,257,d)

    def forward(self, stream_codes, stream_labels, query_codes, targets):
        """query_codes (E,6,256): one probe per step, targeting a slot <= s.
        Loop arms return logits (E,6,N_LABELS) — one prediction per probe;
        dense answers the final probe only, returning (E,N_LABELS)."""
        E = stream_codes.shape[0]
        dev = stream_codes.device
        if self.arch == "dense":
            parts = [self._step_tokens(stream_codes[:, s],
                                       stream_labels[:, s])
                     + self.seg_emb[:, s]
                     for s in range(N_STREAM)]
            parts.append(self._step_tokens(query_codes[:, -1])
                         + self.seg_emb[:, N_STREAM])
            h = self._run_deep(torch.cat(parts, dim=1))
            return self._pool(h), {}

        # Memory ACCUMULATES (the only regime that has ever worked in this
        # project — reselect/rewrite memories lost every head-to-head, and
        # the smoke reproduced that here: per-step re-competition makes
        # retention a chain of exploration events that never bootstraps).
        # B is per-step perception bandwidth: the label token + B-1 chosen
        # content tokens are deep-encoded in the context of current memory
        # and their outputs APPENDED; nothing re-competes. Selection is
        # therefore purely: which B-1 of 256 tokens best represent this
        # image for later retrieval.
        q = self.query0.expand(E, -1, -1)
        mem = None
        logits_all = []
        inst = {}
        aux = torch.zeros((), device=dev)
        for s in range(N_STREAM):
            tok = self._step_tokens(stream_codes[:, s], stream_labels[:, s])
            sc = self._score(q, tok)                     # (E,257)
            force = torch.full((E, 1), N_POS, dtype=torch.long, device=dev)
            ok_s = None
            if self.policy == "oracle":
                ok_s = F.pad(self.oracle_fn(stream_codes[:, s]), (0, 1))
            top = self._admit(sc, force_idx=force, oracle_key=ok_s)
            sel = tok.gather(
                1, top.unsqueeze(-1).expand(-1, -1, tok.shape[-1]))
            gate = torch.sigmoid(sc.gather(1, top)).unsqueeze(-1)
            x = sel * (1 + gate)
            if mem is not None:
                x = torch.cat([x, mem], dim=1)
            h = self._run_deep(x)
            # write ONE fused token per image: the label slot's output
            # (force_idx puts it at index 0), which co-encoded the chosen
            # content. The class->label binding lives inside a single
            # vector BY CONSTRUCTION, collapsing retrieval from a two-hop
            # induction circuit (which the dense diagnostic showed does
            # not form at this scale) to a one-hop similarity match.
            new = h[:, :1] + self.mem_emb                # sticky
            mem = new if mem is None else torch.cat([mem, new], dim=1)
            if self.modulate:
                dq, _ = self.q_update(q, mem, mem, need_weights=False)
                q = self.q_norm(q + dq)

            # probe: budget applies to PERCEIVING the query image; the
            # model's own memory is read by an ARCHITECTURAL lookup —
            # query summary dotted against binding tokens, softmax, read
            # (NTM-style content addressing). The dense diagnostic showed
            # the retrieval circuit does NOT emerge from plain attention
            # at this scale; per the project's law, the mechanism is
            # built in and only the REPRESENTATIONS are learned. Probes
            # write nothing.
            tokq = self._step_tokens(query_codes[:, s])  # (E,256,d)
            scq = self._score(q, tokq)
            ok_q = None if self.policy != "oracle" \
                else self.oracle_fn(query_codes[:, s])
            topq = self._admit(scq, oracle_key=ok_q)
            selq = tokq.gather(
                1, topq.unsqueeze(-1).expand(-1, -1, tokq.shape[-1]))
            gq = torch.sigmoid(scq.gather(1, topq)).unsqueeze(-1)
            hq = self._run_deep(selq * (1 + gq))
            p = self._pool_vec(hq)                       # (E,d)
            att_logits = torch.einsum(
                "bd,bmd->bm", F.normalize(p, dim=-1),
                F.normalize(mem, dim=-1)) * self.read_temp
            att = torch.softmax(att_logits, dim=-1)
            read = torch.einsum("bm,bmd->bd", att, mem)
            logits_all.append(self.head(self.read_norm(p + read)))
            # auxiliary retrieval supervision: the correct slot is known
            # at training time; CE on the attention logits teaches the
            # match directly (unsupervised matching locks onto a fixed
            # slot and saturates — smoke-verified). With retrieval
            # supervised, read_hit measures exactly whether the STORED
            # CONTENT can support matching — the admission question.
            aux = aux + F.cross_entropy(att_logits.float(), targets[:, s])
            if s == N_STREAM - 1:
                with torch.no_grad():
                    inst["read_hit"] = float(
                        (att.argmax(-1) == targets[:, -1]).float().mean())
                    inst["read_conf"] = float(
                        att.max(-1).values.mean())
        inst["aux_loss"] = aux / N_STREAM               # tensor, has grad
        return torch.stack(logits_all, dim=1), inst


# ---------------------------------------------------------------------------

def cfg_name(cfg):
    if cfg["arch"] == "dense":
        return "ICL_dense"
    tag = cfg["policy"] if cfg["policy"] in ("random", "oracle") else \
        ("learned" if cfg["modulate"] else "static")
    return f"ICL_{tag}_B{cfg['budget']}"


def build_mi_table(codes, labels, class_ids, k_codes, device):
    """Per-code informativeness: I contribution = KL(p(class|code)||p(class))
    over the episodic-TRAIN classes only (held-out generalization is part
    of what the oracle arm measures)."""
    keep = torch.isin(labels.long(), class_ids.long())
    c = codes[keep].long().flatten()
    y = labels[keep].long().repeat_interleave(codes.shape[1])
    cls_of = torch.full((int(labels.max()) + 1,), -1, dtype=torch.long)
    cls_of[class_ids.long()] = torch.arange(len(class_ids))
    y = cls_of[y]
    counts = torch.zeros(k_codes, len(class_ids))
    counts.index_put_((c, y), torch.ones_like(c, dtype=torch.float),
                      accumulate=True)
    p_ck = (counts + 1e-3) / (counts.sum(1, keepdim=True) + 1e-3 * counts.shape[1])
    p_c = counts.sum(0) / counts.sum()
    mi = (p_ck * (p_ck.log() - p_c.log())).sum(1)
    return mi.to(device)                                  # (k_codes,)


def train_config(cfg, samp_tr, samp_va, val_set, k_codes, device, args,
                 mi_table=None):
    import wandb
    os.environ.pop("WANDB_RUN_ID", None)
    name = cfg_name(cfg)
    batch = args.batch_size // 2 if cfg["arch"] == "dense" \
        else args.batch_size
    run = wandb.init(project=args.wandb_project, name=name,
                     config={**cfg, "epochs": args.num_epochs,
                             "batch": batch, "lr": args.lr,
                             "batches_per_epoch": args.batches_per_epoch},
                     reinit=True)
    torch.manual_seed(args.seed)
    model = ICLModel(k_codes, **cfg).to(device)
    if cfg["policy"] == "oracle":
        model.oracle_fn = lambda c: mi_table[c]
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=0.05)
    warm = torch.optim.lr_scheduler.LinearLR(opt, 1e-2, 1.0, 2)
    cos = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.num_epochs - 2)
    sched = torch.optim.lr_scheduler.SequentialLR(opt, [warm, cos], [2])
    best = 0.0
    for ep in range(args.num_epochs):
        model.train()
        t0, losses = time.time(), []
        for _ in range(args.batches_per_epoch):
            sc_, pm_, qc_, y_, ts_ = samp_tr.sample(batch)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, inst = model(sc_, pm_, qc_, ts_)
                if cfg["arch"] == "dense":
                    loss = F.cross_entropy(logits, y_[:, -1])
                else:
                    loss = (F.cross_entropy(logits.flatten(0, 1),
                                            y_.flatten())
                            + args.aux_w * inst["aux_loss"])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach()))
        sched.step()
        model.eval()
        correct = tot = 0
        insts, slot_hit, slot_n = [], np.zeros(N_STREAM), np.zeros(N_STREAM)
        probe_acc = np.zeros(N_STREAM)
        with torch.no_grad():
            for i in range(0, val_set[0].shape[0], batch):
                sl = slice(i, i + batch)
                sc_, pm_, qc_, y_, ts_ = (v[sl] for v in val_set)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits, inst = model(sc_, pm_, qc_, ts_)
                if cfg["arch"] == "dense":
                    ok = logits.argmax(-1) == y_[:, -1]
                else:
                    pred = logits.argmax(-1)             # (E,6)
                    ok_all = (pred == y_)
                    probe_acc += ok_all.float().sum(0).cpu().numpy()
                    ok = ok_all[:, -1]
                correct += int(ok.sum()); tot += len(y_)
                if inst:
                    insts.append(inst)
                ts_np = ts_[:, -1].cpu().numpy()
                ok_np = ok.cpu().numpy()
                for sl_i in range(N_STREAM):
                    m_ = ts_np == sl_i
                    slot_hit[sl_i] += ok_np[m_].sum(); slot_n[sl_i] += m_.sum()
        acc = 100 * correct / tot
        best = max(best, acc)
        log = {"val_top1": acc, "train_loss": float(np.mean(losses)),
               "epoch": ep, "epoch_sec": time.time() - t0}
        line = ""
        if cfg["arch"] == "loop":
            for s_i in range(N_STREAM):
                log[f"acc_slot{s_i}"] = float(
                    100 * slot_hit[s_i] / max(slot_n[s_i], 1))
                log[f"acc_probe{s_i}"] = float(100 * probe_acc[s_i] / tot)
            for key in ("read_hit", "read_conf"):
                log[key] = float(np.mean([b[key] for b in insts]))
            log["val_aux"] = float(np.mean(
                [float(b["aux_loss"]) for b in insts]))
            line = (f" hit {log['read_hit']:.2f}"
                    f" conf {log['read_conf']:.2f}"
                    f" slot0 {log['acc_slot0']:.0f}")
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
    p.add_argument("--budgets", type=int, nargs="+", default=[4, 16])
    p.add_argument("--num_epochs", type=int, default=40)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--batches_per_epoch", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--aux_w", type=float, default=0.5)
    p.add_argument("--n_val_episodes", type=int, default=2048)
    p.add_argument("--codebook_artifact", type=str,
                   default="luckymushy-individual/neocore/"
                           "vocab-6duv9qzw:v0")
    p.add_argument("--codebook_file", type=str, default="codebook_k2048.pt")
    p.add_argument("--lake_dir", type=str, default="./data/dino_lake")
    p.add_argument("--jpeg_cache_dir",    type=str, default="./jpeg_cache")
    p.add_argument("--dataset_name",      type=str,
                   default="clane9/imagenet-100")
    p.add_argument("--dataset_cache_dir", type=str, default="./data")
    p.add_argument("--wandb_project", type=str, default="neocore-icl")
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

    # class holdout: 80 episodic-train classes, 20 never-episodically-seen
    split = torch.randperm(100, generator=torch.Generator().manual_seed(42))
    samp_tr = EpisodeSampler(train_codes, train_labels, split[:80], device)
    samp_va = EpisodeSampler(val_codes, val_labels, split[80:], device)
    g = torch.Generator(device=device).manual_seed(123)
    val_set = samp_va.sample(args.n_val_episodes, generator=g)
    print(f"holdout val classes: {sorted(split[80:].tolist())}", flush=True)

    run_dir = Path("runs") / "ICL_sweep"
    run_dir.mkdir(parents=True, exist_ok=True)
    (Path("runs") / "LATEST").write_text(str(run_dir))

    mi_table = build_mi_table(train_codes, train_labels, split[:80],
                              k_codes, device)
    print(f"MI table: mean {float(mi_table.mean()):.3f} "
          f"max {float(mi_table.max()):.3f}", flush=True)

    configs = []
    for b in args.budgets:
        configs.append({"arch": "loop", "policy": "oracle",
                        "modulate": False, "budget": b})  # upper reference
        configs.append({"arch": "loop", "policy": "learned",
                        "modulate": True, "budget": b})
        configs.append({"arch": "loop", "policy": "learned",
                        "modulate": False, "budget": b})
        configs.append({"arch": "loop", "policy": "random",
                        "modulate": False, "budget": b})
    configs.append({"arch": "dense", "policy": "learned",
                    "modulate": False, "budget": 0})   # ceiling runs LAST

    results = {}
    for cfg in configs:
        name, bst = train_config(cfg, samp_tr, samp_va, val_set,
                                 k_codes, device, args, mi_table=mi_table)
        results[name] = bst
        print(f"=== {name}: best top1 {bst:.2f}", flush=True)

    import wandb, json
    (run_dir / "summary.json").write_text(json.dumps(results, indent=1))
    os.environ.pop("WANDB_RUN_ID", None)
    run = wandb.init(project=args.wandb_project, name="ICL_sweep_summary",
                     reinit=True)
    for k, v in results.items():
        run.summary[k] = v
    art = wandb.Artifact(f"icl-sweep-{run.id}", type="cls_sweep")
    art.add_dir(str(run_dir))
    run.log_artifact(art)
    art.wait()
    print("ARTIFACT_VERIFIED")
    run.finish()
    print("SWEEP_RESULTS " + json.dumps(results))


if __name__ == "__main__":
    main()
