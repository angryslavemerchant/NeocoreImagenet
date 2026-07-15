"""
model_asfnet_br.py — Border-Retention ASFNet (single- and two-stage).

STRATEGY CHANGE vs model_asfnet / model_asfnet2:
Stage 1 no longer pools chunks into single tokens. Instead it KEEPS every
border token (any token with at least one incident cut edge) and DROPS all
interior tokens. Consequences:

  - Every surviving token keeps its exact integer grid coordinate, so 2D
    rotary position stays true per token instead of a centroid average.
  - Compression scales like interior/area: big smooth chunks shed many
    tokens, small detailed chunks shed almost none — adaptive by mechanics.
  - Stage 2 (ASFNetBR2) routes on TRUE GRID ADJACENCY restricted to
    survivors. k-NN on centroids is gone entirely: no non-contiguous
    shortcut edges, planar percolation semantics restored.
  - Islands (retained tokens with no retained grid neighbour) have no valid
    stage-2 edges, so they remain singleton groups automatically.

GRADIENT PATH (replaces the weighted pool):
The router's soft probs reach the task loss through a confidence residual
on every retained token:

    token_out = token + s * token,   s = sum of incident soft edge probs

d(token_out)/d(probs) = token per incident edge — per-token placement
gradient, and no token can be scaled to zero. This is H-Net's DeChunk
confidence-multiply in its natural 2D form.

Shared components (PatchEmbed, TransformerBlock, BoundaryRouter,
GroupMerge, gpu_connected_components, load_balancing_loss) are imported
from model_asfnet — this file inherits any fixes made there. Nothing in
the existing files is modified.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_asfnet import (
    PatchEmbed,
    TransformerBlock,
    BoundaryRouter,
    GroupMerge,
    gpu_connected_components,
    load_balancing_loss,
)


# ---------------------------------------------------------------------------
# Retention helpers
# ---------------------------------------------------------------------------

def token_boundary_evidence(
    probs:        torch.Tensor,   # (B, E) soft boundary probs
    edge_indices: torch.Tensor,   # (E, 2)
    n_tokens:     int,
) -> torch.Tensor:
    """
    s_m = sum of soft boundary probabilities over the token's incident grid
    edges. Differentiable in probs — this is the tensor that carries the
    router's placement gradient into the confidence residual.
    Returns (B, N).
    """
    idx_i, idx_j = edge_indices[:, 0], edge_indices[:, 1]
    s = probs.new_zeros(probs.shape[0], n_tokens)
    s = s.index_add(1, idx_i, probs)
    s = s.index_add(1, idx_j, probs)
    return s


def border_keep_mask(
    hard:         torch.Tensor,   # (B, E) hard boundary decisions (0/1)
    edge_indices: torch.Tensor,   # (E, 2)
    n_tokens:     int,
) -> torch.Tensor:
    """
    keep[b, m] = True iff token m has at least one incident cut edge.
    Built from hard decisions (detached — a routing fact, not a gradient
    path). Singleton chunks are all-border, so they are always retained.
    Returns (B, N) bool.
    """
    idx_i, idx_j = edge_indices[:, 0], edge_indices[:, 1]
    hb = hard.detach()
    b  = hb.new_zeros(hb.shape[0], n_tokens)
    b  = b.index_add(1, idx_i, hb)
    b  = b.index_add(1, idx_j, hb)
    return b > 0.5


def compact_survivors(
    tokens: torch.Tensor,   # (B, N, D)
    coords: torch.Tensor,   # (N, 2) shared grid coords, or (B, N, 2)
    keep:   torch.Tensor,   # (B, N) bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Pack retained tokens to the front of the sequence, padded to the batch
    max. Stable sort preserves raster order among survivors (deterministic,
    and keeps RoPE-friendly locality in the sequence, though RoPE itself
    only reads coords).

    Returns:
        out_tokens: (B, max_K, D)
        out_coords: (B, max_K, 2)  — exact integer grid coords per survivor
        pad_mask:   (B, max_K) bool, True = padding slot
        sel:        (B, max_K) long — original grid index of each slot
                    (pad slots point at dropped positions; mask before use)
        n_keep:     (B,) survivor count per image
    """
    B, N, D = tokens.shape
    device  = tokens.device

    n_keep = keep.sum(dim=1)                          # (B,)
    max_K  = int(n_keep.max().item())                 # sync point, same class as GroupMerge's max_G

    order = torch.argsort((~keep).to(torch.uint8), dim=1, stable=True)  # keep first
    sel   = order[:, :max_K]                          # (B, max_K)

    out_tokens = tokens.gather(1, sel.unsqueeze(-1).expand(-1, -1, D))

    coords_b   = coords.unsqueeze(0).expand(B, -1, -1) if coords.dim() == 2 else coords
    out_coords = coords_b.gather(1, sel.unsqueeze(-1).expand(-1, -1, 2))

    arange   = torch.arange(max_K, device=device).unsqueeze(0)   # (1, max_K)
    pad_mask = arange >= n_keep.unsqueeze(1)                     # (B, max_K)

    return out_tokens, out_coords, pad_mask, sel, n_keep


# ---------------------------------------------------------------------------
# Stage 2 router — fixed grid edges, per-image survivor mask
# ---------------------------------------------------------------------------

class MaskedGridRouter(BoundaryRouter):
    """
    Stage 2 router for the retention model. Identical parameter-free
    H-Net-style scoring as BoundaryRouter (identity-initialised d_model
    projections, p = clamp((1 - cos)/2, 0, 1)), but on the ORIGINAL grid
    edge buffer with per-image validity: an edge is valid iff both endpoints
    survived stage 1. Any retained-retained grid edge is routable —
    including edges that were stage-1 cuts — so stage 2 can consolidate
    across stage-1 chunk boundaries.

    The load-balancing target is per-image spanning-forest arithmetic on the
    survivor subgraph (n_tok = survivors, n_edge = valid edges), same shape
    as BoundaryRouter2's per-image loss. The survivor subgraph is a subgraph
    of the planar grid, so it stays planar — the percolation reasoning that
    motivated the topology-aware target applies again, unlike the k-NN graph.
    If islands make the group target unreachable (components > G_target) the
    loss saturates harmlessly.
    """

    def forward(  # signature differs from parent by design
        self,
        h:                 torch.Tensor,   # (B, N, D)
        keep:              torch.Tensor,   # (B, N) bool — stage 1 survivors
        target_group_size: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            hard:    (B, E) hard decisions (meaningful only on valid edges)
            probs:   (B, E) soft probabilities
            valid:   (B, E) bool — both endpoints retained
            l_ratio: scalar per-image-averaged load-balancing loss
        """
        idx_i = self.edge_indices[:, 0]
        idx_j = self.edge_indices[:, 1]

        q = self.q_proj(h)
        k = self.k_proj(h)

        cos   = F.cosine_similarity(q[:, idx_i, :], k[:, idx_j, :], dim=-1).clamp(-1.0, 1.0)
        probs = ((1.0 - cos) / 2.0).clamp(0.0, 1.0)   # (B, E)
        hard  = (probs > 0.5).float()                  # (B, E)

        valid   = keep[:, idx_i] & keep[:, idx_j]      # (B, E)
        valid_f = valid.float()

        n_edge    = valid_f.sum(dim=1)                 # (B,)
        n_tok     = keep.float().sum(dim=1)            # (B,)
        safe_edge = n_edge.clamp(min=1.0)

        F_rate = (hard.detach() * valid_f).sum(dim=1) / safe_edge
        G_rate = (probs         * valid_f).sum(dim=1) / safe_edge

        g_target = n_tok / target_group_size
        f_target = ((n_edge - n_tok + g_target) / safe_edge).clamp(min=1e-6)
        N        = 1.0 / f_target
        N_safe   = N.clamp(min=1.0 + 1e-4)

        per_image = load_balancing_loss(F_rate, G_rate, N_safe)   # (B,)

        good = (N > 1.0) & (n_edge >= 2) & (n_tok >= 2)
        l_ratio = per_image[good].mean() if good.any() else h.new_zeros(())

        return hard, probs, valid, l_ratio


# ---------------------------------------------------------------------------
# Stage 2 connected components — grid edges, survivor validity, trash group
# ---------------------------------------------------------------------------

def gpu_connected_components_masked(
    hard:         torch.Tensor,   # (B, E) hard decisions
    valid:        torch.Tensor,   # (B, E) both-endpoints-retained mask
    edge_indices: torch.Tensor,   # (E, 2) fixed grid buffer
    n_tokens:     int,
    keep:         torch.Tensor,   # (B, N) bool
) -> torch.Tensor:
    """
    Label propagation identical to gpu_connected_components (see
    model_asfnet.py) with two changes:

      1. An edge is connected iff valid AND not a boundary. Edges touching
         a dropped token never propagate.
      2. All DROPPED tokens are pre-labelled -1. They have no valid edges,
         so -1 never spreads and they never receive a label — after the
         contiguous remap every dropped token lands in ONE shared "trash"
         group (id 0 whenever any token was dropped, since -1 sorts first).
         The caller masks that group out via a retained-count check, so the
         padded group tensor stays small instead of carrying one singleton
         group per dropped token.

    Returns (B, N) contiguous group IDs.
    """
    B      = hard.shape[0]
    device = hard.device

    idx_i = edge_indices[:, 0]
    idx_j = edge_indices[:, 1]

    connected = valid & (hard < 0.5)   # (B, E)

    labels = (
        torch.arange(n_tokens, device=device)
        .unsqueeze(0).expand(B, -1).clone()
    )
    labels[~keep] = -1                 # shared trash label for dropped tokens

    INF       = n_tokens
    idx_i_exp = idx_i.unsqueeze(0).expand(B, -1)
    idx_j_exp = idx_j.unsqueeze(0).expand(B, -1)

    for _ in range(n_tokens):
        label_i = labels[:, idx_i]
        label_j = labels[:, idx_j]

        min_ij    = torch.minimum(label_i, label_j)
        propagate = torch.where(connected, min_ij, min_ij.new_full((), INF))

        new_labels = labels.clone()
        new_labels.scatter_reduce_(1, idx_j_exp, propagate, reduce="amin", include_self=True)
        new_labels.scatter_reduce_(1, idx_i_exp, propagate, reduce="amin", include_self=True)

        if torch.equal(new_labels, labels):
            labels = new_labels
            break
        labels = new_labels

    # Contiguous remap (same sort + cumsum as stage 1)
    sorted_labels, sort_idx = labels.sort(dim=1)
    is_new = torch.cat([
        torch.ones(B, 1, device=device, dtype=torch.bool),
        sorted_labels[:, 1:] != sorted_labels[:, :-1],
    ], dim=1)
    contiguous_sorted = is_new.long().cumsum(dim=1) - 1
    contiguous_labels = torch.empty_like(labels)
    contiguous_labels.scatter_(1, sort_idx, contiguous_sorted)

    return contiguous_labels


def masked_edge_probs_to_token_weights(
    probs:        torch.Tensor,   # (B, E)
    hard:         torch.Tensor,   # (B, E)
    valid:        torch.Tensor,   # (B, E)
    edge_indices: torch.Tensor,   # (E, 2)
    n_tokens:     int,
    keep:         torch.Tensor,   # (B, N) bool
) -> torch.Tensor:
    """
    Border-only linear merge weights for the stage 2 pool, restricted to
    valid (survivor-survivor) edges — the retention analogue of
    edge_probs_to_token_weights.

    GUARD: a stage-2 group whose perimeter is entirely gaps (dropped-token
    holes) or an island has ZERO incident cut edges, so pure border
    weighting would give it a zero weight-sum and GroupMerge would emit a
    clamped junk token for a REAL group. A tiny uniform base (+1e-3) on
    every retained token makes such groups degrade to a uniform mean while
    leaving border-weighted groups (s ~ O(1)) essentially unchanged.

    Dropped tokens get exactly 0 — they only inhabit the trash group, which
    is masked downstream anyway.
    """
    idx_i, idx_j = edge_indices[:, 0], edge_indices[:, 1]
    valid_f = valid.to(probs.dtype)

    pw = probs * valid_f
    s  = probs.new_zeros(probs.shape[0], n_tokens)
    s  = s.index_add(1, idx_i, pw)
    s  = s.index_add(1, idx_j, pw)

    hb     = hard.detach() * valid_f
    border = probs.new_zeros(probs.shape[0], n_tokens)
    border = border.index_add(1, idx_i, hb)
    border = border.index_add(1, idx_j, hb)
    is_border = (border > 0.5).to(probs.dtype)

    keep_f = keep.to(probs.dtype)
    return (s * is_border + 1e-3) * keep_f


# ---------------------------------------------------------------------------
# ASFNetBR — single-stage border retention
# ---------------------------------------------------------------------------

class ASFNetBR(nn.Module):
    """
    Single-stage border-retention ASFNet.

    Pipeline:
      PatchEmbed                       N tokens (grid × grid)
      → encoder_blocks × TransformerBlock
      → BoundaryRouter                 (unchanged, incl. ratio target)
      → keep = border tokens           interior tokens dropped
      → confidence residual            token += s · token   (grad → probs)
      → compact survivors              (B, max_K, D), true grid coords
      → stage_proj                     Linear at the stage boundary
                                       (counterpart of GroupMerge.proj)
      → main_blocks × TransformerBlock over survivors
      → masked GAP → classifier

    num_classes=0 builds no classifier head (used by the autoencoder).
    forward_features() exposes everything the AE / linear probe needs.
    """

    def __init__(
        self,
        image_size:        int   = 224,
        patch_size:        int   = 16,
        in_channels:       int   = 3,
        d_model:           int   = 256,
        num_heads:         int   = 8,
        encoder_blocks:    int   = 2,
        main_blocks:       int   = 6,
        mlp_ratio:         float = 3.0,
        num_classes:       int   = 100,
        target_group_size: float = 3.0,
        router_proj_dim:   int   = 64,   # accepted for parity, ignored by router
    ):
        super().__init__()
        self.target_group_size = target_group_size
        grid_size = image_size // patch_size

        self.patch_embed = PatchEmbed(image_size, patch_size, in_channels, d_model)

        self.encoder = nn.ModuleList([
            TransformerBlock(d_model, num_heads, mlp_ratio)
            for _ in range(encoder_blocks)
        ])

        self.router     = BoundaryRouter(d_model, router_proj_dim, grid_size)
        self.stage_proj = nn.Linear(d_model, d_model)

        self.main_net = nn.ModuleList([
            TransformerBlock(d_model, num_heads, mlp_ratio)
            for _ in range(main_blocks)
        ])

        self.norm       = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes) if num_classes > 0 else None

    def forward_features(self, x: torch.Tensor, keep_all: bool = False):
        """
        keep_all=True bypasses retention (keep = all tokens) while leaving
        everything else — router, ratio loss, confidence residual —
        identical. Used by the joint-embedding teacher so the uncompressed
        branch is the same function as the student minus the drop.
        Default False = behaviour unchanged.

        Returns:
            feats:       (B, max_K, D) post-norm survivor tokens
            coord_c:     (B, max_K, 2) exact grid coords per survivor
            pad_mask:    (B, max_K) True = padding
            sel:         (B, max_K) original grid index per slot
            keep:        (B, N) bool retention mask (True = kept)
            l_ratio:     scalar ratio loss
            mean_kept:   0-dim GPU tensor — avg retained tokens per image
            mean_groups: 0-dim GPU tensor — avg stage-1 group count
                         (both lazy: call .item() only at logging time —
                         an .item() here would force a CPU/GPU sync every
                         training step)
            s:           (B, N) differentiable per-token boundary evidence
                         (sum of incident soft edge probs) — the AE's
                         bottleneck ranks survivors with this
            probs:       (B, E) soft edge boundary probabilities — the AE's
                         token-level keep-rate loss is built from these
        """
        tokens, coords = self.patch_embed(x)
        N = tokens.shape[1]

        for block in self.encoder:
            tokens = block(tokens, coords)

        hard, probs, l_ratio = self.router(tokens, self.target_group_size)

        # Diagnostics only — grouping no longer feeds a pool.
        group_ids = gpu_connected_components(hard.detach(), self.router.edge_indices, N)

        s    = token_boundary_evidence(probs, self.router.edge_indices, N)
        keep = border_keep_mask(hard, self.router.edge_indices, N)
        # Guard: an image with zero cut edges has zero border tokens.
        # Keep everything for that image (it is simply uncompressed).
        keep = keep | (keep.sum(dim=1, keepdim=True) == 0)

        if keep_all:
            keep = torch.ones_like(keep)

        # Confidence residual — the placement gradient path.
        tokens = tokens + s.unsqueeze(-1) * tokens

        tok_c, coord_c, pad_mask, sel, n_keep = compact_survivors(tokens, coords, keep)
        tok_c = self.stage_proj(tok_c)

        for block in self.main_net:
            tok_c = block(tok_c, coord_c, pad_mask)

        tok_c = self.norm(tok_c)

        mean_kept   = n_keep.float().mean().detach()
        mean_groups = (group_ids.max(dim=1).values + 1).float().mean().detach()

        return tok_c, coord_c, pad_mask, sel, keep, l_ratio, mean_kept, mean_groups, s, probs

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, float, float]:
        """
        Returns: logits, l_ratio, mean_kept, mean_groups
        """
        feats, _, pad_mask, _, _, l_ratio, mean_kept, mean_groups, _s, _probs = \
            self.forward_features(x)

        real_mask   = (~pad_mask).float()
        token_sum   = (feats * real_mask.unsqueeze(-1)).sum(dim=1)
        token_count = real_mask.sum(dim=1, keepdim=True).clamp(min=1)
        pooled      = token_sum / token_count

        logits = self.classifier(pooled)
        # Classifier callers (train_asfnet_br.py) expect plain floats.
        return logits, l_ratio, float(mean_kept.item()), float(mean_groups.item())

    def count_parameters(self) -> dict:
        def n(m): return sum(p.numel() for p in m.parameters()) if m is not None else 0
        return {
            "patch_embed": n(self.patch_embed),
            "encoder":     n(self.encoder),
            "router":      n(self.router),
            "stage_proj":  n(self.stage_proj),
            "main_net":    n(self.main_net),
            "norm+head":   n(self.norm) + n(self.classifier),
            "total":       n(self),
        }


# ---------------------------------------------------------------------------
# ASFNetBR2R — two-stage DOUBLE retention: border-keep at BOTH stages
# ---------------------------------------------------------------------------

def masked_border_keep(
    hard:         torch.Tensor,   # (B, E) hard decisions
    valid:        torch.Tensor,   # (B, E) both-endpoints-retained mask
    edge_indices: torch.Tensor,   # (E, 2)
    n_tokens:     int,
    keep_prev:    torch.Tensor,   # (B, N) stage-1 retention
) -> torch.Tensor:
    """
    Stage-2 analogue of border_keep_mask, restricted to valid
    (survivor-survivor) edges: keep a survivor iff it has >=1 incident valid
    cut edge OR it is an island (no valid edges at all — the survivor-graph
    singleton chunk, all-border by the same argument as stage 1's singletons).
    Returns (B, N) bool, always a subset of keep_prev.
    """
    idx_i, idx_j = edge_indices[:, 0], edge_indices[:, 1]
    valid_f = valid.float()

    cut = (hard.detach() * valid_f)
    cuts_per_tok = cut.new_zeros(cut.shape[0], n_tokens)
    cuts_per_tok = cuts_per_tok.index_add(1, idx_i, cut).index_add(1, idx_j, cut)

    valid_per_tok = valid_f.new_zeros(valid_f.shape[0], n_tokens)
    valid_per_tok = valid_per_tok.index_add(1, idx_i, valid_f).index_add(1, idx_j, valid_f)

    border = cuts_per_tok > 0.5
    island = valid_per_tok < 0.5
    return keep_prev & (border | island)


class ASFNetBR2R(nn.Module):
    """
    Two-stage border-retention ASFNet where stage 2 is ALSO retention —
    no pooling anywhere. Stage 1 drops the interiors of fine chunks;
    stage 2 routes the survivor subgraph into coarser chunks and drops
    THEIR interiors. Every surviving token keeps its exact grid coordinate
    all the way into the main network (and, in the AE, into the decoder).

    Gradient paths mirror stage 1's: each router reaches the loss through
    its own confidence residual (s = sum of incident soft probs on that
    stage's routable edges); retention masks are hard/detached routing facts.

    Compression is compound: keep2 ⊆ keep1. The known failure mode is the
    same keep-everything corner as the single-stage AE — with small target
    group sizes almost every token is some chunk's border. Whether the
    coarser stage-2 chunks develop real interiors (and therefore real drops)
    is exactly what the AE experiment observes; watch mean_kept2 vs
    mean_kept1.
    """

    def __init__(
        self,
        image_size:          int   = 224,
        patch_size:          int   = 16,
        in_channels:         int   = 3,
        d_model:             int   = 256,
        num_heads:           int   = 8,
        encoder1_blocks:     int   = 3,
        encoder2_blocks:     int   = 3,
        main_blocks:         int   = 6,
        mlp_ratio:           float = 3.0,
        num_classes:         int   = 100,
        target_group_size_1: float = 3.0,
        target_group_size_2: float = 3.0,
        router_proj_dim:     int   = 64,
    ):
        super().__init__()
        self.target_group_size_1 = target_group_size_1
        self.target_group_size_2 = target_group_size_2
        grid_size = image_size // patch_size

        self.patch_embed = PatchEmbed(image_size, patch_size, in_channels, d_model)

        self.encoder1 = nn.ModuleList([
            TransformerBlock(d_model, num_heads, mlp_ratio)
            for _ in range(encoder1_blocks)
        ])
        self.router1     = BoundaryRouter(d_model, router_proj_dim, grid_size)
        self.stage1_proj = nn.Linear(d_model, d_model)

        self.encoder2 = nn.ModuleList([
            TransformerBlock(d_model, num_heads, mlp_ratio)
            for _ in range(encoder2_blocks)
        ])
        self.router2     = MaskedGridRouter(d_model, router_proj_dim, grid_size)
        self.stage2_proj = nn.Linear(d_model, d_model)

        self.main_net = nn.ModuleList([
            TransformerBlock(d_model, num_heads, mlp_ratio)
            for _ in range(main_blocks)
        ])

        self.norm       = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes) if num_classes > 0 else None

    def forward_features(self, x: torch.Tensor):
        """
        Same return contract as ASFNetBR.forward_features (so the AE decoder
        scatters survivors at exact coords identically), with stage-2 extras:

        Returns:
            feats:       (B, max_K, D) post-norm final survivor tokens
            coord_c:     (B, max_K, 2) exact grid coords per survivor
            pad_mask:    (B, max_K) True = padding
            sel:         (B, max_K) original grid index per slot
            keep2:       (B, N) bool FINAL retention mask (subset of keep1)
            l_ratio1, l_ratio2: scalar router losses
            mean_kept1:  0-dim GPU tensor — stage-1 survivors
            mean_kept2:  0-dim GPU tensor — final survivors
            s1, s2:      (B, N) differentiable boundary evidence per stage
            probs1:      (B, E) stage-1 soft edge probabilities
        """
        tokens, coords = self.patch_embed(x)
        N = tokens.shape[1]

        # ---- Stage 1 (identical to ASFNetBR / ASFNetBR2) ----
        for block in self.encoder1:
            tokens = block(tokens, coords)

        hard1, probs1, l_ratio1 = self.router1(tokens, self.target_group_size_1)

        s1    = token_boundary_evidence(probs1, self.router1.edge_indices, N)
        keep1 = border_keep_mask(hard1, self.router1.edge_indices, N)
        keep1 = keep1 | (keep1.sum(dim=1, keepdim=True) == 0)

        tokens = tokens + s1.unsqueeze(-1) * tokens
        tokens = self.stage1_proj(tokens)

        # ---- Stage 2 encoder: full layout, dropped tokens masked as keys ----
        drop1 = ~keep1
        for block in self.encoder2:
            tokens = block(tokens, coords, drop1)

        # ---- Stage 2 retention on the survivor subgraph ----
        hard2, probs2, valid2, l_ratio2 = self.router2(
            tokens, keep1, self.target_group_size_2,
        )

        keep2 = masked_border_keep(hard2, valid2, self.router2.edge_indices, N, keep1)
        # Guard: an image with zero stage-2 cuts (and no islands) would keep
        # nothing — fall back to keep1 for that image (stage 2 uncompressed).
        keep2 = keep2 | (keep1 & (keep2.sum(dim=1, keepdim=True) == 0))

        # Stage-2 confidence residual on VALID edges only — probs2's
        # gradient path (there is no weighted merge in this variant).
        idx_i, idx_j = self.router2.edge_indices[:, 0], self.router2.edge_indices[:, 1]
        pw = probs2 * valid2.to(probs2.dtype)
        s2 = pw.new_zeros(pw.shape[0], N)
        s2 = s2.index_add(1, idx_i, pw).index_add(1, idx_j, pw)

        tokens = tokens + s2.unsqueeze(-1) * tokens

        tok_c, coord_c, pad_mask, sel, n_keep = compact_survivors(tokens, coords, keep2)
        tok_c = self.stage2_proj(tok_c)

        for block in self.main_net:
            tok_c = block(tok_c, coord_c, pad_mask)

        tok_c = self.norm(tok_c)

        mean_kept1 = keep1.float().sum(dim=1).mean().detach()
        mean_kept2 = n_keep.float().mean().detach()

        return (tok_c, coord_c, pad_mask, sel, keep2, l_ratio1, l_ratio2,
                mean_kept1, mean_kept2, s1, s2, probs1)

    def forward(self, x: torch.Tensor):
        """Returns: logits, l_ratio1, l_ratio2, mean_kept1, mean_kept2"""
        feats, _, pad_mask, _, _, l_ratio1, l_ratio2, mean_kept1, \
            mean_kept2, _s1, _s2, _probs1 = self.forward_features(x)

        real_mask   = (~pad_mask).float()
        token_sum   = (feats * real_mask.unsqueeze(-1)).sum(dim=1)
        token_count = real_mask.sum(dim=1, keepdim=True).clamp(min=1)
        pooled      = token_sum / token_count

        logits = self.classifier(pooled)
        return logits, l_ratio1, l_ratio2, \
            float(mean_kept1.item()), float(mean_kept2.item())

    def count_parameters(self) -> dict:
        def n(m): return sum(p.numel() for p in m.parameters()) if m is not None else 0
        return {
            "patch_embed": n(self.patch_embed),
            "encoder1":    n(self.encoder1),
            "router1":     n(self.router1),
            "stage1_proj": n(self.stage1_proj),
            "encoder2":    n(self.encoder2),
            "router2":     n(self.router2),
            "stage2_proj": n(self.stage2_proj),
            "main_net":    n(self.main_net),
            "norm+head":   n(self.norm) + n(self.classifier),
            "total":       n(self),
        }


# ---------------------------------------------------------------------------
# ASFNetBR2 — two-stage: retention, then grid-adjacent grouping of survivors
# ---------------------------------------------------------------------------

class ASFNetBR2(nn.Module):
    """
    Two-stage border-retention ASFNet.

    Pipeline:
      PatchEmbed
        → encoder1 × TransformerBlock          [full 196, global]
        → BoundaryRouter (grid edges)          [stage 1 cuts]
        → keep1 = border tokens, drop interiors
        → confidence residual (grad → probs1)
        → stage1_proj

        → encoder2 × TransformerBlock          [full 196 layout; DROPPED
                                                tokens masked out of
                                                attention as keys]
        → MaskedGridRouter                     [grid edges among survivors]
        → gpu_connected_components_masked      [dropped → one trash group]
        → GroupMerge (border-weighted pool)    [survivors → stage-2 groups]
        → main_blocks × TransformerBlock       [group tokens, centroid RoPE]
        → masked GAP → classifier

    Notes:
      - Stage 2 sees exact per-token grid positions (retention preserved
        them), and its adjacency is the true grid subgraph over survivors —
        k-NN is gone. Islands stay islands (no valid edges → singleton).
      - weighted_merge2 defaults TRUE: with a uniform stage-2 pool, probs2
        would have NO path to the task loss — the exact gradient
        disconnection this project already diagnosed once. Disable only to
        deliberately ablate (--uniform_merge2 in the train script).
      - encoder2 runs at the full 196 layout for simplicity (dropped tokens
        are masked, their outputs discarded). Compute optimisation
        (compacting before encoder2 with edge remap) is deliberately
        deferred — quality first, per project convention.
    """

    def __init__(
        self,
        image_size:          int   = 224,
        patch_size:          int   = 16,
        in_channels:         int   = 3,
        d_model:             int   = 256,
        num_heads:           int   = 8,
        encoder1_blocks:     int   = 2,
        encoder2_blocks:     int   = 2,
        main_blocks:         int   = 4,
        mlp_ratio:           float = 3.0,
        num_classes:         int   = 100,
        target_group_size_1: float = 3.0,
        target_group_size_2: float = 3.0,
        router_proj_dim:     int   = 64,
        weighted_merge2:     bool  = True,
    ):
        super().__init__()
        self.target_group_size_1 = target_group_size_1
        self.target_group_size_2 = target_group_size_2
        self.weighted_merge2     = weighted_merge2
        grid_size = image_size // patch_size

        self.patch_embed = PatchEmbed(image_size, patch_size, in_channels, d_model)

        self.encoder1 = nn.ModuleList([
            TransformerBlock(d_model, num_heads, mlp_ratio)
            for _ in range(encoder1_blocks)
        ])

        self.router1     = BoundaryRouter(d_model, router_proj_dim, grid_size)
        self.stage1_proj = nn.Linear(d_model, d_model)

        self.encoder2 = nn.ModuleList([
            TransformerBlock(d_model, num_heads, mlp_ratio)
            for _ in range(encoder2_blocks)
        ])

        self.router2 = MaskedGridRouter(d_model, router_proj_dim, grid_size)
        self.merge2  = GroupMerge(d_model)

        self.main_net = nn.ModuleList([
            TransformerBlock(d_model, num_heads, mlp_ratio)
            for _ in range(main_blocks)
        ])

        self.norm       = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes) if num_classes > 0 else None

    def forward_features(self, x: torch.Tensor):
        """
        Everything up to (and including) the post-norm stage-2 group tokens —
        the two-stage counterpart of ASFNetBR.forward_features, exposed for
        the autoencoder (ASFNetAE2).

        Returns:
            feats2:      (B, max_G, D) post-norm stage-2 group tokens
            coords2:     (B, max_G, 2) group centroid coords
            pad_mask2:   (B, max_G) True = padding / trash group
            group_ids2:  (B, N) contiguous stage-2 group id per grid position
                         (dropped tokens share the trash group)
            keep1:       (B, N) bool stage-1 retention mask
            l_ratio1, l_ratio2: scalar router losses
            mean_kept1:   0-dim GPU tensor — avg stage-1 survivors
            mean_groups2: 0-dim GPU tensor — avg real stage-2 groups
            s1:          (B, N) differentiable stage-1 boundary evidence
            probs1:      (B, E) stage-1 soft edge probabilities
        """
        B = x.shape[0]

        # ---- Stage 1: route, retain borders, confidence residual ----
        tokens, coords = self.patch_embed(x)
        N = tokens.shape[1]

        for block in self.encoder1:
            tokens = block(tokens, coords)

        hard1, probs1, l_ratio1 = self.router1(tokens, self.target_group_size_1)

        s1    = token_boundary_evidence(probs1, self.router1.edge_indices, N)
        keep1 = border_keep_mask(hard1, self.router1.edge_indices, N)
        keep1 = keep1 | (keep1.sum(dim=1, keepdim=True) == 0)

        tokens = tokens + s1.unsqueeze(-1) * tokens
        tokens = self.stage1_proj(tokens)

        # ---- Stage 2 encoder: full layout, dropped tokens masked as keys ----
        drop1 = ~keep1   # (B, N) True = masked, matches TransformerBlock's attn_mask
        for block in self.encoder2:
            tokens = block(tokens, coords, drop1)

        # ---- Stage 2 routing on the survivor subgraph of the grid ----
        hard2, probs2, valid2, l_ratio2 = self.router2(
            tokens, keep1, self.target_group_size_2,
        )

        group_ids2 = gpu_connected_components_masked(
            hard2.detach(), valid2, self.router2.edge_indices, N, keep1,
        )

        token_weights2 = None
        if self.weighted_merge2:
            token_weights2 = masked_edge_probs_to_token_weights(
                probs2, hard2, valid2, self.router2.edge_indices, N, keep1,
            )

        padded2, coords2, _, _ = self.merge2(
            tokens, coords, group_ids2, token_weights=token_weights2,
        )

        # ---- Mask groups that contain no retained token ----
        # (the shared trash group of dropped tokens, plus stride padding)
        max_G2   = padded2.shape[1]
        real_sum = torch.zeros(B, max_G2, device=x.device)
        real_sum.scatter_add_(1, group_ids2.clamp(0, max_G2 - 1), keep1.float())
        pad_mask2 = real_sum < 0.5

        # ---- Main network over group tokens ----
        for block in self.main_net:
            padded2 = block(padded2, coords2, pad_mask2)

        padded2 = self.norm(padded2)

        mean_kept1   = keep1.float().sum(dim=1).mean().detach()
        mean_groups2 = (~pad_mask2).sum(dim=1).float().mean().detach()

        return (padded2, coords2, pad_mask2, group_ids2, keep1,
                l_ratio1, l_ratio2, mean_kept1, mean_groups2, s1, probs1)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, float]:
        """
        Returns: logits, l_ratio1, l_ratio2, mean_kept1, mean_groups2
        """
        feats2, _, pad_mask2, _, _, l_ratio1, l_ratio2, mean_kept1, \
            mean_groups2, _s1, _probs1 = self.forward_features(x)

        real_mask   = (~pad_mask2).float()
        token_sum   = (feats2 * real_mask.unsqueeze(-1)).sum(dim=1)
        token_count = real_mask.sum(dim=1, keepdim=True).clamp(min=1)
        pooled      = token_sum / token_count

        logits = self.classifier(pooled)

        return logits, l_ratio1, l_ratio2, \
            float(mean_kept1.item()), float(mean_groups2.item())

    def count_parameters(self) -> dict:
        def n(m): return sum(p.numel() for p in m.parameters())
        return {
            "patch_embed": n(self.patch_embed),
            "encoder1":    n(self.encoder1),
            "router1":     n(self.router1),
            "stage1_proj": n(self.stage1_proj),
            "encoder2":    n(self.encoder2),
            "router2":     n(self.router2),
            "merge2":      n(self.merge2),
            "main_net":    n(self.main_net),
            "norm+head":   n(self.norm) + n(self.classifier),
            "total":       n(self),
        }
