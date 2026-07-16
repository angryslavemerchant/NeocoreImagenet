"""
model_asfnet_ae_ladder.py — N-stage border-retention autoencoder starting
from FINE patches (default 4x4 -> 56x56 = 3136 tokens), with a HARD budget
at every stage.

WHY (2026-07-16 checkpoint): 16x16 patches pre-chunk the image with a fixed
grid before the router sees anything — most learnable boundaries are inside
patches, and above 16x16 there is barely one level of structure left for a
second stage to find. Meanwhile every unpinned two-stage variant was
dynamically unstable, and the budgeted one was stable. This model combines
both lessons: start fine (room for real hierarchy) and pin the rate at each
stage (stability + predictable memory).

Design:
  PatchEmbed 4x4                      3136 tokens, d 64
  stage 1: 2 blocks (d  64) -> route -> border-keep -> budget  784
  stage 2: 2 blocks (d 128) -> route -> border-keep -> budget  196
  stage 3: 2 blocks (d 256) -> route -> border-keep -> budget   49
  main:    4 blocks (d 256) on the 49 survivors
  decoder: survivors scattered at EXACT grid coords into the 3136 grid,
           3 thin blocks (d 64), loss on DROPPED positions (~98.4% of the
           image — dense by construction).

Mechanics per stage (mirrors ASFNetBR / ASFNetBR2R):
  - attention runs on the COMPACTED survivor set (rate-pinned, so the
    sequence length per stage is bounded by the previous budget);
  - routing runs on the full 56x56 grid layout (features scattered back),
    via MaskedGridRouter with validity = both endpoints still kept;
  - each router gets its gradient through a confidence residual on its
    valid-edge soft evidence s_k; budgets rank by ACCUMULATED evidence
    (sum of s_k over stages so far), the ladder analogue of AE2R's s1+s2;
  - stage-1 blocks and the decoder see no padding -> attn_mask None ->
    FlashAttention fast path at 3136 tokens.

gpu_connected_components is deliberately never called (its label-propagation
loop is O(N) iterations — prohibitive at N=3136 and diagnostics-only).
"""

import torch
import torch.nn as nn

from model_asfnet import PatchEmbed, TransformerBlock
from model_asfnet_br import (
    MaskedGridRouter,
    compact_survivors,
    masked_border_keep,
)


class FieldRouter(nn.Module):
    """
    Potential-field router: emits a SCALAR FIELD u per token; a (valid) edge
    is cut iff the endpoints' quantized field values differ
    (round(u_i) != round(u_j)).

    WHY (2026-07-16 pause): with per-edge cut probabilities, cuts need not
    close — a mesh of slits makes every token "border" without enclosing
    anything, and every threshold invites parking. Level-set boundaries are
    closed curves BY TOPOLOGY: two tokens in different bins cannot be
    connected by any same-bin path, so EVERY cut separates two chunks and
    interiors exist by construction. This restores in 2D the 1D identity
    H-Net relies on (every cut is a real chunk boundary).

    Soft evidence for the gradient path: p_e = sigmoid(4*(|du| - 0.5)) —
    high when the field jumps by more than half a bin across the edge. The
    hard cut is a detached routing fact, as everywhere else in ASFNet; u
    learns through the confidence residual on accumulated evidence. There
    is no ratio loss and no free per-edge threshold: with hard budgets
    downstream, collapsing u to a constant gives an ARBITRARY top-k and a
    high reconstruction loss, so — unlike threshold parking — collapse is
    not a low-loss corner.
    """

    def __init__(self, d_model: int, proj_dim: int, grid_size: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, 1),
        )
        # same fixed grid-edge buffer layout as BoundaryRouter
        idx = torch.arange(grid_size * grid_size).reshape(grid_size, grid_size)
        right = torch.stack([idx[:, :-1].flatten(), idx[:, 1:].flatten()], dim=1)
        down  = torch.stack([idx[:-1, :].flatten(), idx[1:, :].flatten()], dim=1)
        self.register_buffer("edge_indices", torch.cat([right, down], dim=0))

    def forward(
        self,
        h:                 torch.Tensor,   # (B, N, D) full-grid layout
        keep:              torch.Tensor,   # (B, N) bool current survivors
        target_group_size: float,          # unused; signature parity
    ):
        u = self.head(h).squeeze(-1)                    # (B, N)
        idx_i = self.edge_indices[:, 0]
        idx_j = self.edge_indices[:, 1]

        du    = u[:, idx_i] - u[:, idx_j]               # (B, E)
        probs = torch.sigmoid(4.0 * (du.abs() - 0.5))
        hard  = (torch.round(u[:, idx_i].detach())
                 != torch.round(u[:, idx_j].detach())).float()

        valid   = keep[:, idx_i] & keep[:, idx_j]
        l_ratio = h.new_zeros(())                        # no auxiliary loss
        return hard, probs, valid, l_ratio


@torch.compiler.disable
def grid_component_labels(
    connected:    torch.Tensor,   # (B, E) bool — valid AND uncut
    edge_indices: torch.Tensor,   # (E, 2)
    n_tokens:     int,
    keep:         torch.Tensor,   # (B, N) bool
    iters:        int = 64,
) -> torch.Tensor:
    """
    Fixed-iteration min-label propagation over the survivor subgraph
    (compile-disabled: a data-dependent loop with no sync'd early exit).
    `iters` bounds the merge diameter — regions with graph diameter beyond
    it may keep >1 label, which only creates phantom borders (extra keep
    candidates), never lost ones. Dropped tokens get label -1.
    Returns (B, N) long.
    """
    B = connected.shape[0]
    device = connected.device
    idx_i, idx_j = edge_indices[:, 0], edge_indices[:, 1]

    labels = torch.arange(n_tokens, device=device).unsqueeze(0).expand(B, -1).clone()
    labels = labels.masked_fill(~keep, -1)

    INF = n_tokens
    idx_i_exp = idx_i.unsqueeze(0).expand(B, -1)
    idx_j_exp = idx_j.unsqueeze(0).expand(B, -1)
    for _ in range(iters):
        li, lj = labels[:, idx_i], labels[:, idx_j]
        mn = torch.minimum(li, lj)
        prop = torch.where(connected, mn, mn.new_full((), INF))
        new = labels.clone()
        new.scatter_reduce_(1, idx_j_exp, prop, reduce="amin", include_self=True)
        new.scatter_reduce_(1, idx_i_exp, prop, reduce="amin", include_self=True)
        labels = new.masked_fill(~keep, -1)
    return labels


def component_border_keep(
    hard:         torch.Tensor,   # (B, E)
    valid:        torch.Tensor,   # (B, E) bool
    edge_indices: torch.Tensor,
    n_tokens:     int,
    keep_prev:    torch.Tensor,   # (B, N) bool
) -> torch.Tensor:
    """
    Enclosure-aware border retention: a cut edge counts only if it actually
    SEPARATES two connected components of the survivor subgraph — slits
    (cuts whose endpoints remain connected around them) confer no
    border-ness. Islands (no valid edges) are singleton chunks: all-border,
    kept, as everywhere else in ASFNet.
    """
    idx_i, idx_j = edge_indices[:, 0], edge_indices[:, 1]
    connected = valid & (hard.detach() < 0.5)
    labels = grid_component_labels(connected, edge_indices, n_tokens, keep_prev)

    separating = (valid & (hard.detach() > 0.5)
                  & (labels[:, idx_i] != labels[:, idx_j])).float()
    b = separating.new_zeros(separating.shape[0], n_tokens)
    b = b.index_add(1, idx_i, separating).index_add(1, idx_j, separating)

    vf = valid.float()
    per_tok = vf.new_zeros(vf.shape[0], n_tokens)
    per_tok = per_tok.index_add(1, idx_i, vf).index_add(1, idx_j, vf)
    island = per_tok < 0.5

    return keep_prev & ((b > 0.5) | island)


class LadderStage(nn.Module):
    """proj (widen) -> blocks on compacted survivors -> router. The keep /
    budget arithmetic lives in ASFNetAELadder.forward_features, which owns
    the full-grid layout."""

    def __init__(self, d_in: int, d_model: int, num_heads: int, blocks: int,
                 budget_tokens: int, grid_size: int, router_proj_dim: int,
                 mlp_ratio: float, router_kind: str = "edge"):
        super().__init__()
        assert (d_model // num_heads) % 4 == 0, \
            "head_dim must be divisible by 4 for 2D RoPE"
        assert router_kind in ("edge", "field", "component")
        self.budget_tokens = budget_tokens
        self.router_kind = router_kind
        self.proj = nn.Linear(d_in, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, mlp_ratio)
            for _ in range(blocks)
        ])
        if router_kind == "field":
            self.router = FieldRouter(d_model, router_proj_dim, grid_size)
        else:
            self.router = MaskedGridRouter(d_model, router_proj_dim, grid_size)


class ASFNetAELadder(nn.Module):
    def __init__(
        self,
        image_size:        int   = 224,
        patch_size:        int   = 4,
        in_channels:       int   = 3,
        stage_dims:        tuple = (64, 128, 256),
        stage_heads:       tuple = (4, 4, 8),
        stage_blocks:      tuple = (2, 2, 2),
        stage_budgets:     tuple = (784, 196, 49),
        main_blocks:       int   = 4,
        mlp_ratio:         float = 3.0,
        target_group_size: float = 3.0,
        router_proj_dim:   int   = 64,
        decoder_d_model:   int   = 64,
        decoder_blocks:    int   = 3,
        decoder_heads:     int   = 4,
        norm_pix_loss:     bool  = True,
        router_kind:       str   = "edge",
        budget_floor:      bool  = False,
    ):
        super().__init__()
        assert len(stage_dims) == len(stage_heads) == len(stage_blocks) \
            == len(stage_budgets)
        self.router_kind = router_kind
        # budget_floor: keep EXACTLY K per stage — border tokens first, then
        # highest-evidence survivors fill any deficit. Added 2026-07-16 after
        # two under-supply collapses: budgets-as-caps let routers go zero-cut
        # and ride the island guard down to a handful of tokens, at which
        # point later routers receive zero gradient (no valid edges) and
        # freeze. A floor makes the rate a true architectural constant.
        self.budget_floor = budget_floor
        assert all(a > b for a, b in zip(stage_budgets, stage_budgets[1:])), \
            "budgets must strictly decrease"
        assert (decoder_d_model // decoder_heads) % 4 == 0

        self.patch_size    = patch_size
        self.in_channels   = in_channels
        self.grid_size     = image_size // patch_size
        self.n_patches     = self.grid_size ** 2
        self.norm_pix_loss = norm_pix_loss
        self.target_group_size = target_group_size
        assert stage_budgets[0] < self.n_patches

        self.patch_embed = PatchEmbed(image_size, patch_size, in_channels,
                                      stage_dims[0])

        dims_in = (stage_dims[0],) + tuple(stage_dims[:-1])
        self.stages = nn.ModuleList([
            LadderStage(d_in, d, h, b, k, self.grid_size,
                        router_proj_dim, mlp_ratio, router_kind=router_kind)
            for d_in, d, h, b, k in zip(dims_in, stage_dims, stage_heads,
                                        stage_blocks, stage_budgets)
        ])

        d_final = stage_dims[-1]
        self.main_net = nn.ModuleList([
            TransformerBlock(d_final, stage_heads[-1], mlp_ratio)
            for _ in range(main_blocks)
        ])
        self.norm = nn.LayerNorm(d_final)

        # ---- Decoder (single-stage recipe at ladder scale) ----
        self.decoder_embed = nn.Linear(d_final, decoder_d_model)
        self.mask_token    = nn.Parameter(torch.zeros(1, 1, decoder_d_model))
        nn.init.normal_(self.mask_token, std=0.02)
        self.decoder = nn.ModuleList([
            TransformerBlock(decoder_d_model, decoder_heads, mlp_ratio)
            for _ in range(decoder_blocks)
        ])
        self.decoder_norm = nn.LayerNorm(decoder_d_model)
        self.decoder_pred = nn.Linear(decoder_d_model,
                                      patch_size ** 2 * in_channels)

    # ------------------------------------------------------------------
    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        B, C, H, W = imgs.shape
        p = self.patch_size
        g = H // p
        x = imgs.reshape(B, C, g, p, g, p)
        return x.permute(0, 2, 4, 3, 5, 1).reshape(B, g * g, p * p * C)

    def unpatchify(self, pred: torch.Tensor) -> torch.Tensor:
        B, N, _ = pred.shape
        p, g, C = self.patch_size, self.grid_size, self.in_channels
        x = pred.reshape(B, g, g, p, p, C)
        return x.permute(0, 5, 1, 3, 2, 4).reshape(B, C, g * p, g * p)

    # ------------------------------------------------------------------
    def forward_features(self, x: torch.Tensor):
        """
        Returns:
            feats:     (B, K_last, D) post-norm final survivors
            pad_mask:  (B, K_last)
            sel:       (B, K_last) grid index per slot
            keep:      (B, N) final retention mask
            l_ratio:   summed router losses
            kept_per_stage: list of 0-dim GPU tensors (mean kept after each
                            stage's budget)
            stage_keeps: list of (B, N) bool masks — retention AFTER each
                            stage's budget (for visualisation / probing)
        """
        tokens, coords = self.patch_embed(x)          # (B, N, d0), (N, 2)
        B, N, _ = tokens.shape
        ei = self.stages[0].router.edge_indices
        idx_i, idx_j = ei[:, 0], ei[:, 1]

        keep    = torch.ones(B, N, dtype=torch.bool, device=x.device)
        s_total = tokens.new_zeros(B, N)
        l_ratio = tokens.new_zeros(())
        kept_per_stage = []
        stage_keeps = []

        tok_c, coord_c, pad_mask, sel = tokens, coords, None, None

        for stage in self.stages:
            tok_c = stage.proj(tok_c)
            for blk in stage.blocks:
                tok_c = blk(tok_c, coord_c, pad_mask)   # None mask -> flash
            d = tok_c.shape[-1]

            # Full-grid layout for routing (stage 1 already is full-grid).
            if sel is None:
                full = tok_c
            else:
                full = tok_c.new_zeros(B, N, d)
                full = full.scatter(
                    1, sel.unsqueeze(-1).expand(-1, -1, d), tok_c)

            hard, probs, valid, l_r = stage.router(
                full, keep, self.target_group_size)
            l_ratio = l_ratio + l_r

            # Differentiable evidence on valid edges (router's grad path).
            pw  = probs * valid.to(probs.dtype)
            s_k = pw.new_zeros(B, N)
            s_k = s_k.index_add(1, idx_i, pw).index_add(1, idx_j, pw)
            s_total = s_total + s_k

            s_slot = s_k if sel is None else s_k.gather(1, sel)
            tok_c  = tok_c + s_slot.unsqueeze(-1) * tok_c

            # Border retention on the survivor subgraph, then the budget.
            # edge/field: any incident (valid) cut confers border-ness —
            #   for the field router every cut separates BY CONSTRUCTION.
            # component: only cuts that separate true connected components
            #   count (slits confer nothing).
            if self.router_kind == "component":
                new_keep = component_border_keep(hard, valid, ei, N, keep)
            else:
                new_keep = masked_border_keep(hard, valid, ei, N, keep)
            new_keep = new_keep | (keep & (new_keep.sum(dim=1, keepdim=True) == 0))

            K = stage.budget_tokens
            if self.budget_floor:
                # Exact-K: borders outrank everything (BIG bonus), evidence
                # breaks ties and fills the deficit from the remaining
                # survivors. keep_prev >= K holds by the ladder's strictly
                # decreasing budgets, so exactly K survive every stage.
                scores = (s_total.detach()
                          + 1e6 * new_keep.float()).masked_fill(
                              ~keep, float("-inf"))
                top  = scores.topk(K, dim=1).indices
                keep = torch.zeros_like(keep).scatter(1, top, True)
            else:
                scores = s_total.detach().masked_fill(~new_keep, float("-inf"))
                top    = scores.topk(K, dim=1).indices
                capped = torch.zeros_like(new_keep)
                capped = capped.scatter(1, top, True) & new_keep
                over   = new_keep.sum(dim=1, keepdim=True) > K
                keep   = torch.where(over, capped, new_keep)

            # Compact the (post-residual) survivors for the next stage —
            # sequence length from here on is bounded by K: memory is
            # rate-pinned by construction.
            if sel is None:
                full_post = tok_c
            else:
                full_post = tok_c.new_zeros(B, N, d)
                full_post = full_post.scatter(
                    1, sel.unsqueeze(-1).expand(-1, -1, d), tok_c)
            tok_c, coord_c, pad_mask, sel, n_keep = compact_survivors(
                full_post, coords, keep)

            kept_per_stage.append(n_keep.float().mean().detach())
            stage_keeps.append(keep)

        for blk in self.main_net:
            tok_c = blk(tok_c, coord_c, pad_mask)
        tok_c = self.norm(tok_c)

        return tok_c, pad_mask, sel, keep, l_ratio, kept_per_stage, stage_keeps

    # ------------------------------------------------------------------
    def _decode(self, feats, pad_mask, sel):
        """ASFNetAE recipe: survivors at exact grid coords, mask token
        everywhere else, thin decoder over the full fine grid (no pads ->
        flash)."""
        B  = feats.shape[0]
        dd = self.mask_token.shape[-1]

        enc      = self.decoder_embed(feats)
        mask_tok = self.mask_token.to(enc.dtype)
        enc      = torch.where(pad_mask.unsqueeze(-1),
                               mask_tok.expand_as(enc), enc)

        base = mask_tok.expand(B, self.n_patches, dd)
        x    = base.scatter(1, sel.unsqueeze(-1).expand(-1, -1, dd), enc)

        coords = self.patch_embed.coords
        for blk in self.decoder:
            x = blk(x, coords)
        return self.decoder_pred(self.decoder_norm(x))

    # ------------------------------------------------------------------
    def forward(self, imgs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """
        Standard 6-tuple. Slot semantics:
            mean_kept:   FINAL survivors (post last budget, ~49)
            mean_groups: stage-1 survivors (post first budget, ~784)
            drop_frac:   final dropped fraction (loss support, ~0.984)
        """
        feats, pad_mask, sel, keep, l_ratio, kept_per_stage, _stage_keeps = \
            self.forward_features(imgs)

        pred = self._decode(feats, pad_mask, sel)      # (B, N, p*p*C)

        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mu  = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mu) / (var + 1e-6) ** 0.5

        loss_patch = ((pred.float() - target.float()) ** 2).mean(dim=-1)

        m = (~keep).float()
        loss_rec = (loss_patch * m).sum() / m.sum().clamp(min=1.0)

        drop_frac = m.mean().detach()
        return (loss_rec, l_ratio, imgs.new_zeros(()),
                kept_per_stage[-1], kept_per_stage[0], drop_frac)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def reconstruct(self, imgs: torch.Tensor):
        feats, pad_mask, sel, keep, _, _, _ = self.forward_features(imgs)
        pred = self._decode(feats, pad_mask, sel)
        return self.unpatchify(pred.float()), keep

    def count_parameters(self) -> dict:
        def n(m): return sum(p.numel() for p in m.parameters())
        return {
            "patch_embed": n(self.patch_embed),
            "stages":      n(self.stages),
            "main_net":    n(self.main_net),
            "decoder":     n(self.decoder_embed) + n(self.decoder)
                           + n(self.decoder_norm) + n(self.decoder_pred)
                           + self.mask_token.numel(),
            "total":       n(self),
        }
