"""
model_asfnet2.py — Two-stage hierarchical ASFNet.

Applies grouping twice in sequence:
  Stage 1: 196 tokens  →  196/target_group_size_1 groups  (fixed grid adjacency)
  Stage 2:  G1 tokens  →  G1/target_group_size_2  groups  (k-NN adjacency)

  Default (both=3.0):  196 → ~65 → ~22  (1/9 overall compression)
  Example (4.0, 2.0):  196 → ~49 → ~25  (1/8 overall compression)

Shared components (PatchEmbed, Attention, TransformerBlock, BoundaryRouter,
GroupMerge, gpu_connected_components, and the local-attention blocks) are
imported directly from model_asfnet to avoid duplication. New code here covers
only Stage 2-specific pieces:
  - build_knn_edges
  - knn_edges_to_mask
  - gpu_connected_components_dynamic
  - BoundaryRouter2
  - ASFNet2

Local attention (optional, off by default):
  - local_encoder1 : Stage 1 encoder uses a fixed grid-window mask
                     (LocalTransformerBlock, radius = local_radius).
  - local_encoder2 : Stage 2 encoder uses the per-image k-NN adjacency as its
                     attention mask (LocalTransformerBlock2). The same k-NN
                     edges are reused by BoundaryRouter2, so the encoder mixes
                     tokens over exactly the graph the router then cuts.
  The four combinations of these two flags give the global/global,
  local/global, global/local, local/local ablation grid.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# Shared components — imported to avoid duplication.
# model_asfnet2 inherits any bug-fixes made to these automatically.
from model_asfnet import *


# ---------------------------------------------------------------------------
# Stage 2 edge builder — k-nearest-neighbor on group centroids
# ---------------------------------------------------------------------------

def build_knn_edges(
    padded_coords: torch.Tensor,  # (B, G, 2) group centroid coordinates
    pad_mask: torch.Tensor,       # (B, G) True = padding slot
    k: int,                       # neighbours per token
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build a k-Nearest-Neighbor (k-NN — a graph where each node is connected
    to its k spatially closest nodes) edge set for Stage 2 tokens.

    At Stage 1, adjacency was taken from the regular patch grid because every
    token was exactly one patch and the grid IS the true topology. At Stage 2,
    tokens are irregular groups with fractional centroid coordinates, so the
    grid structure is gone. k-NN on centroids recovers a sensible adjacency:
    the closest groups in image space are almost always the ones that were
    actually neighboring before merging.

    Padding slots are excluded by setting their distances to infinity before
    taking the top-k, so they never appear as valid neighbors. Any edge
    involving a padding source token is also marked invalid.

    Distance is computed in float32 regardless of input dtype to avoid the
    low-precision rounding in bfloat16 scrambling the nearest-neighbor order
    for tightly-packed group centroids.

    Args:
        padded_coords: (B, G, 2)  group centroid (row, col) coordinates
        pad_mask:      (B, G)     True = padding
        k:             neighbours per token (recommended: 6)

    Returns:
        src:   (B, G*k)  source token index per edge
        dst:   (B, G*k)  destination token index per edge
        valid: (B, G*k)  True if both endpoints are real (non-padding) tokens
    """
    B, G, _ = padded_coords.shape
    device   = padded_coords.device

    # Pairwise L2 distance between all (real + padding) token centroids
    coords_f = padded_coords.float()
    diff = coords_f.unsqueeze(2) - coords_f.unsqueeze(1)  # (B, G, G, 2)
    dist = diff.norm(dim=-1)                               # (B, G, G)

    # Mask out self-connections and any pair where either endpoint is padding
    INF = 1e9
    eye       = torch.eye(G, device=device, dtype=torch.bool)
    pad_either = pad_mask.unsqueeze(2) | pad_mask.unsqueeze(1)  # (B, G, G)
    dist = dist.masked_fill(eye.unsqueeze(0), INF)
    dist = dist.masked_fill(pad_either, INF)

    # k-NN: for each token, take the k closest others
    k_clamped = min(k, G - 1)
    _, top_k_idx = dist.topk(k_clamped, dim=-1, largest=False)  # (B, G, k)

    # Build flat edge lists: (B, G*k)
    src = (
        torch.arange(G, device=device)
        .view(1, G, 1)
        .expand(B, G, k_clamped)
        .reshape(B, G * k_clamped)
    )
    dst = top_k_idx.reshape(B, G * k_clamped)

    # Mark edges invalid if the source token is a padding slot.
    # (Destination padding is already excluded via INF distances, but padding
    # sources still get arbitrary top-k indices from the INF rows — mask them.)
    src_real = ~pad_mask.gather(1, src)   # (B, G*k)
    dst_real = ~pad_mask.gather(1, dst)   # (B, G*k)
    valid    = src_real & dst_real        # (B, G*k)

    return src, dst, valid

def edge_probs_to_token_weights_dynamic(
    probs:    torch.Tensor,   # (B, E) soft boundary probs on k-NN edges
    hard:     torch.Tensor,   # (B, E) hard decisions
    valid:    torch.Tensor,   # (B, E) real-edge mask from build_knn_edges
    src:      torch.Tensor,   # (B, E) source token idx
    dst:      torch.Tensor,   # (B, E) destination token idx
    n_tokens: int,            # max_G1 — tokens per image (incl. padding)
) -> torch.Tensor:
    """
    Stage 2 border-only linear merge weights — the per-image k-NN analogue of
    edge_probs_to_token_weights (model_asfnet.py).

    A token is 'border' if any of its VALID incident k-NN edges is a cut.
    Interior tokens get weight 0 → dropped. Padding tokens have no valid edges,
    so they also get 0 (which is what we want — they were masked anyway). Border
    tokens get w = s = sum of their valid incident edge probs, so higher boundary
    prob → higher weight. Gradient to `probs` flows through border tokens only.

    Uses per-image scatter_add (indices are (B, E), not a shared buffer), and
    multiplies by `valid` so invalid/padding edges contribute nothing regardless
    of the arbitrary indices they carry. Returns (B, n_tokens).
    """
    B       = probs.shape[0]
    device  = probs.device
    valid_f = valid.to(probs.dtype)                      # (B, E)

    # Padding-source rows carry arbitrary dst indices; clamp to stay in-bounds.
    # Their contribution is zeroed by valid_f, so the clamp target is irrelevant.
    src_c = src.clamp(0, n_tokens - 1)
    dst_c = dst.clamp(0, n_tokens - 1)

    # s_m: soft boundary evidence per token over valid incident edges (grad→probs)
    pw = probs * valid_f                                 # (B, E)
    s  = torch.zeros(B, n_tokens, device=device, dtype=probs.dtype)
    s  = s.scatter_add(1, src_c, pw)
    s  = s.scatter_add(1, dst_c, pw)

    # border indicator from hard (detached — routing fact, not a gradient path)
    hb     = hard.detach() * valid_f                     # (B, E)
    border = torch.zeros(B, n_tokens, device=device, dtype=probs.dtype)
    border = border.scatter_add(1, src_c, hb)
    border = border.scatter_add(1, dst_c, hb)
    is_border = (border > 0.5).to(probs.dtype)           # (B, n_tokens)

    return s * is_border                                 # (B, n_tokens)


# ---------------------------------------------------------------------------
# Stage 2 local-attention mask — dense adjacency from the k-NN edge lists
# ---------------------------------------------------------------------------

def knn_edges_to_mask(
    src:      torch.Tensor,   # (B, E) source token index per edge
    dst:      torch.Tensor,   # (B, E) destination token index per edge
    valid:    torch.Tensor,   # (B, E) True = real edge (not involving padding)
    n_tokens: int,            # G = max_G1, tokens per image (incl. padding)
) -> torch.Tensor:
    """
    Convert the per-image k-NN edge lists from build_knn_edges into a dense
    (B, G, G) boolean attention mask for Stage 2 local attention.

    mask[b, i, j] = True means query token i may attend to key token j.
    Set True for every valid edge (i = src, j = dst). Self-loops are added on
    the diagonal so that:
      - every real token attends to itself plus its k nearest real neighbours;
      - every padding token attends only to itself, which keeps its query row
        non-empty (an all-False row would make scaled_dot_product_attention
        produce NaN). The padding token's output is discarded downstream.

    Because padding destinations are already excluded from the edge lists
    (INF distance in build_knn_edges) and padding-source edges are marked
    invalid, no real token ever has a padding token as a key. So this single
    mask fully replaces the separate padding mask the global encoder2 used.

    The mask is built from the SAME edges later passed to BoundaryRouter2, so
    the encoder mixes tokens over exactly the graph the router then cuts.
    """
    B, E   = src.shape
    device = src.device
    G      = n_tokens

    mask = torch.zeros(B, G, G, dtype=torch.bool, device=device)

    bb  = torch.arange(B, device=device).view(B, 1).expand(B, E)  # (B, E)
    sel = valid                                                   # (B, E) bool
    # Flatten only the valid edges into 1D index tensors and set them True.
    mask[bb[sel], src[sel], dst[sel]] = True

    # Self-loops (prevents all-False rows → NaN; gives real tokens self-access).
    diag = torch.arange(G, device=device)
    mask[:, diag, diag] = True

    return mask


# ---------------------------------------------------------------------------
# Stage 2 connected components — per-image edge sets
# ---------------------------------------------------------------------------

def gpu_connected_components_dynamic(
    hard:     torch.Tensor,   # (B, E) hard boundary decisions, 1=boundary 0=connected
    valid:    torch.Tensor,   # (B, E) True = real edge (not involving padding)
    src:      torch.Tensor,   # (B, E) source token index per edge (per image)
    dst:      torch.Tensor,   # (B, E) destination token index per edge (per image)
    n_tokens: int,            # max_G1 — number of tokens per image (including padding)
) -> torch.Tensor:
    """
    Same iterative label-propagation algorithm as gpu_connected_components
    (see model_asfnet.py for the full explanation), adapted for per-image
    edge sets.

    The only difference from the Stage 1 version: edge_indices are (B, E)
    tensors that vary per image rather than a single (E, 2) buffer shared
    across the batch. This is necessary because Stage 2 tokens are irregular
    groups with k-NN adjacency that depends on each image's specific group
    centroid layout.

    An edge is "connected" only if it is both valid (not involving padding)
    AND not a boundary (hard < 0.5). Padding tokens stay isolated: they have
    no valid edges, so they never receive a propagated label, and each becomes
    its own singleton component.

    Args:
        hard:     (B, E) hard router decisions
        valid:    (B, E) edge validity mask from build_knn_edges
        src:      (B, E) source token indices
        dst:      (B, E) destination token indices
        n_tokens: number of tokens per image (= max_G1)

    Returns:
        labels: (B, n_tokens) int64, contiguous group IDs in [0, n_groups)
    """
    B      = hard.shape[0]
    device = hard.device

    connected = valid & (hard < 0.5)   # (B, E)

    labels = (
        torch.arange(n_tokens, device=device)
        .unsqueeze(0).expand(B, -1).clone()
    )   # (B, n_tokens) — each token starts as its own group

    INF   = n_tokens
    src_c = src.clamp(0, n_tokens - 1)   # safe index clamping (padding
    dst_c = dst.clamp(0, n_tokens - 1)   # sources may have arbitrary dst)

    for _ in range(n_tokens):
        label_src = labels.gather(1, src_c)   # (B, E)
        label_dst = labels.gather(1, dst_c)   # (B, E)

        min_ij    = torch.minimum(label_src, label_dst)
        propagate = torch.where(connected, min_ij, min_ij.new_full((), INF))

        new_labels = labels.clone()
        new_labels.scatter_reduce_(1, dst_c, propagate, reduce='amin', include_self=True)
        new_labels.scatter_reduce_(1, src_c, propagate, reduce='amin', include_self=True)

        if torch.equal(new_labels, labels):
            labels = new_labels
            break
        labels = new_labels

    # Remap non-contiguous root labels → contiguous IDs [0, n_groups)
    sorted_labels, sort_idx = labels.sort(dim=1)
    is_new = torch.cat([
        torch.ones(B, 1, device=device, dtype=torch.bool),
        sorted_labels[:, 1:] != sorted_labels[:, :-1],
    ], dim=1)
    contiguous_sorted  = is_new.long().cumsum(dim=1) - 1
    contiguous_labels  = torch.empty_like(labels)
    contiguous_labels.scatter_(1, sort_idx, contiguous_sorted)

    return contiguous_labels


# ---------------------------------------------------------------------------
# Stage 2 boundary router — dynamic edge version
# ---------------------------------------------------------------------------

class BoundaryRouter2(nn.Module):
    """
    Stage 2 router — the same H-Net-faithful RoutingModule as Stage 1
    (identity-initialised d_model projections, parameter-free boundary
    probability p = clamp((1 - cos_sim)/2, 0, 1)), adapted to dynamic per-image
    k-NN edges instead of a fixed grid buffer.

    See BoundaryRouter in model_asfnet.py for the full rationale on why the
    router is parameter-free and identity-initialised, and for which parts are
    faithful to H-Net vs. deliberate 2D deviations. This class differs only in
    edge handling and in that the load-balancing target is computed per image,
    because Stage 1 emits a variable number of real groups per image (so
    n_tokens and the valid edge count vary across the batch).

    The per-image loss is fully vectorised — no Python loop over the batch.

    proj_dim is accepted for backward compatibility but IGNORED (projections are
    always full d_model, per H-Net).
    """

    def __init__(self, d_model: int, proj_dim: int = None):
        super().__init__()
        self.d_model = d_model

        # Identity-initialised projections (H-Net RoutingModule).
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        with torch.no_grad():
            self.q_proj.weight.copy_(torch.eye(d_model))
            self.k_proj.weight.copy_(torch.eye(d_model))
        self.q_proj.weight._no_reinit = True
        self.k_proj.weight._no_reinit = True

    def forward(
        self,
        h:                  torch.Tensor,   # (B, G1, D) Stage 2 input tokens
        src:                torch.Tensor,   # (B, E) source indices from build_knn_edges
        dst:                torch.Tensor,   # (B, E) destination indices
        valid:              torch.Tensor,   # (B, E) real-edge mask
        target_group_size:  float,
        n_real_per_image:   torch.Tensor,   # (B,) real token count per image
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            hard:    (B, E) binary boundary decisions (caller detaches for CC)
            probs:   (B, E) soft boundary probabilities
            l_ratio: scalar load-balancing loss averaged across valid images
        """
        q = self.q_proj(h)   # (B, G1, D)
        k = self.k_proj(h)   # (B, G1, D)

        D_dim = q.shape[-1]
        q_i = q.gather(1, src.unsqueeze(-1).expand(-1, -1, D_dim))   # (B, E, D)
        k_j = k.gather(1, dst.unsqueeze(-1).expand(-1, -1, D_dim))   # (B, E, D)

        cos   = F.cosine_similarity(q_i, k_j, dim=-1).clamp(-1.0, 1.0)   # (B, E)
        probs = ((1.0 - cos) / 2.0).clamp(0.0, 1.0)                       # (B, E)
        hard  = (probs > 0.5).float()                                     # (B, E)

        # --- Per-image load-balancing loss (vectorised) ---
        #
        # Spanning-forest target per image (n_tokens, n_edges are image-specific
        # at Stage 2 because Stage 1 emits a variable number of real groups):
        #     G_target = n_tokens / target_group_size
        #     F_target = (n_edges - n_tokens + G_target) / n_edges
        #     N        = 1 / F_target
        # then fed into the shared load_balancing_loss. F_rate and G_rate are
        # masked means over each image's VALID edges only.
        valid_f = valid.float()                                  # (B, E)
        n_edge  = valid_f.sum(dim=1)                             # (B,)
        n_tok   = n_real_per_image.float()                      # (B,)

        safe_edge = n_edge.clamp(min=1.0)
        F_rate = (hard.detach() * valid_f).sum(dim=1) / safe_edge   # (B,)
        G_rate = (probs        * valid_f).sum(dim=1) / safe_edge    # (B,)

        g_target = n_tok / target_group_size
        f_target = ((n_edge - n_tok + g_target) / safe_edge).clamp(min=1e-6)
        N        = (1.0 / f_target)                                 # (B,)
        # Guard N > 1 for the loss (images that can't form >1 group are dropped
        # from the average via the `good` mask below, but clamp first so the
        # elementwise loss never hits a divide-by-zero).
        N_safe   = N.clamp(min=1.0 + 1e-4)

        per_image = load_balancing_loss(F_rate, G_rate, N_safe)    # (B,)

        good = (N > 1.0) & (n_edge >= 2) & (n_tok >= 2)            # (B,) bool
        if good.any():
            l_ratio = per_image[good].mean()
        else:
            l_ratio = h.new_zeros(())

        return hard, probs, l_ratio


# ---------------------------------------------------------------------------
# ASFNet2
# ---------------------------------------------------------------------------

class ASFNet2(nn.Module):
    """
    Two-stage hierarchical Attention-Shifted Focus Network.

    Pipeline:
      PatchEmbed
        → encoder1_blocks × TransformerBlock  [Stage 1 representation]
        → BoundaryRouter  (fixed grid edges)  [Stage 1 grouping]
        → gpu_connected_components
        → GroupMerge1                          [~196/3 ≈ 65 tokens]

        → encoder2_blocks × TransformerBlock  [Stage 2 representation]
        → build_knn_edges (k-NN on centroids) [Stage 2 adjacency]
        → BoundaryRouter2 (dynamic edges)     [Stage 2 grouping]
        → gpu_connected_components_dynamic
        → GroupMerge2                          [~65/3 ≈ 22 tokens]

        → main_blocks × TransformerBlock      [final reasoning]
        → masked Global Average Pool
        → Linear classifier

    Local attention (optional):
      local_encoder1 swaps the Stage 1 encoder for grid-window local blocks
      (radius local_radius). local_encoder2 swaps the Stage 2 encoder for
      k-NN-masked local blocks; the k-NN adjacency is built once before the
      encoder and reused by BoundaryRouter2. Both default off, giving the plain
      global/global model unless enabled.

    Why k-NN for Stage 2 and not for Stage 1?
    At Stage 1 every token is exactly one patch, so the true spatial
    adjacency IS the regular grid — k-NN on integer grid coordinates
    recovers the exact same edges at extra cost. At Stage 2 groups have
    irregular fractional centroid coordinates, so the grid is gone and
    k-NN is the natural replacement.

    Padding bookkeeping:
    GroupMerge1 pads its output to the batch-maximum group count. Those
    padding slots propagate into Stage 2. The Stage 2 connected-components
    keeps them isolated (no valid edges connect to padding). After GroupMerge2
    a corrected Stage 2 padding mask is computed by checking whether each
    Stage 2 group received any contribution from a real Stage 1 token.
    """

    def __init__(
        self,
        image_size:           int   = 224,
        patch_size:           int   = 16,
        in_channels:          int   = 3,
        d_model:              int   = 256,
        num_heads:            int   = 8,
        encoder1_blocks:      int   = 2,
        encoder2_blocks:      int   = 2,
        main_blocks:          int   = 4,
        mlp_ratio:            float = 3.0,
        num_classes:          int   = 100,
        target_group_size_1:  float = 3.0,  # Stage 1 compression: N → N/target_group_size_1
        target_group_size_2:  float = 3.0,  # Stage 2 compression: G1 → G1/target_group_size_2
        router_proj_dim:      int   = 64,
        knn_k:                int   = 6,
        weighted_merge: bool = False,
        local_encoder1: bool = False,
        local_radius: int = 1,
        local_encoder2: bool = False,
        local_encoder2_safe: bool = False,
    ):
        super().__init__()
        self.target_group_size_1 = target_group_size_1
        self.target_group_size_2 = target_group_size_2
        self.knn_k               = knn_k
        grid_size              = image_size // patch_size

        self.patch_embed = PatchEmbed(image_size, patch_size, in_channels, d_model)

        self.local_encoder1 = local_encoder1
        if local_encoder1:
            self.encoder1 = nn.ModuleList([
                LocalTransformerBlock(d_model, num_heads, grid_size, local_radius, mlp_ratio)
                for _ in range(encoder1_blocks)
            ])
        else:
            self.encoder1 = nn.ModuleList([
                TransformerBlock(d_model, num_heads, mlp_ratio)
                for _ in range(encoder1_blocks)
            ])

        # Stage 1: same fixed-grid router as original ASFNet
        self.router1  = BoundaryRouter(d_model, router_proj_dim, grid_size)
        self.merge1   = GroupMerge(d_model)

        self.local_encoder2 = local_encoder2
        if local_encoder2:
            self.encoder2 = nn.ModuleList([
                LocalTransformerBlock2(d_model, num_heads, mlp_ratio,
                                       safe_attn=local_encoder2_safe)
                for _ in range(encoder2_blocks)
            ])
        else:
            self.encoder2 = nn.ModuleList([
                TransformerBlock(d_model, num_heads, mlp_ratio)
                for _ in range(encoder2_blocks)
            ])

        # Stage 2: dynamic k-NN router, no grid_size needed
        self.router2  = BoundaryRouter2(d_model, router_proj_dim)
        self.merge2   = GroupMerge(d_model)

        self.main_net = nn.ModuleList([
            TransformerBlock(d_model, num_heads, mlp_ratio)
            for _ in range(main_blocks)
        ])

        self.norm       = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)

        self.knn_k = knn_k
        self.weighted_merge = weighted_merge


    def forward(
            self,
            x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, float]:
        # ---- Stage 1 ----
        tokens, coords = self.patch_embed(x)  # (B, N1, D)

        for block in self.encoder1:
            tokens = block(tokens, coords)

        hard1, probs1, l_ratio1 = self.router1(tokens, self.target_group_size_1)

        group_ids1 = gpu_connected_components(
            hard1.detach(), self.router1.edge_indices, tokens.shape[1]
        )

        token_weights1 = None
        if self.weighted_merge:
            token_weights1 = edge_probs_to_token_weights(
                probs1, hard1, self.router1.edge_indices, tokens.shape[1],
            )

        padded_tokens1, padded_coords1, pad_mask1, mean_groups1 = self.merge1(
            tokens, coords, group_ids1, token_weights=token_weights1,
        )

        # ---- Stage 2 adjacency ----
        n_real1 = (~pad_mask1).sum(dim=1)  # (B,) real token count per image
        src2, dst2, valid2 = build_knn_edges(padded_coords1, pad_mask1, self.knn_k)
        max_G1 = padded_tokens1.shape[1]

        # ---- Stage 2 encoder ----
        if self.local_encoder2:
            adj2 = knn_edges_to_mask(src2, dst2, valid2, max_G1)
            for block in self.encoder2:
                padded_tokens1 = block(padded_tokens1, padded_coords1, adj2)
        else:
            for block in self.encoder2:
                padded_tokens1 = block(padded_tokens1, padded_coords1, pad_mask1)

        # ---- Stage 2 routing ----
        hard2, probs2, l_ratio2 = self.router2(
            padded_tokens1, src2, dst2, valid2,
            self.target_group_size_2, n_real1,
        )

        group_ids2 = gpu_connected_components_dynamic(
            hard2.detach(), valid2, src2, dst2, max_G1
        )

        token_weights2 = None
        if self.weighted_merge:
            token_weights2 = edge_probs_to_token_weights_dynamic(
                probs2, hard2, valid2, src2, dst2, max_G1,
            )

        padded_tokens2, padded_coords2, _, mean_groups2 = self.merge2(
            padded_tokens1, padded_coords1, group_ids2, token_weights=token_weights2,
        )

        # ---- Stage 2 padding mask (unchanged) ----
        max_G2 = padded_tokens2.shape[1]
        real1 = (~pad_mask1).float()
        real2_sum = torch.zeros(x.shape[0], max_G2, device=x.device)
        ids2_clamped = group_ids2.clamp(0, max_G2 - 1)
        real2_sum.scatter_add_(1, ids2_clamped, real1)
        pad_mask2 = real2_sum < 0.5
        mean_groups2 = float((~pad_mask2).sum(dim=1).float().mean().item())

        # ---- Main network + classifier (unchanged) ----
        for block in self.main_net:
            padded_tokens2 = block(padded_tokens2, padded_coords2, pad_mask2)

        padded_tokens2 = self.norm(padded_tokens2)

        real_mask = (~pad_mask2).float()
        token_sum = (padded_tokens2 * real_mask.unsqueeze(-1)).sum(dim=1)
        token_count = real_mask.sum(dim=1, keepdim=True).clamp(min=1)
        pooled = token_sum / token_count

        logits = self.classifier(pooled)
        return logits, l_ratio1, l_ratio2, mean_groups1, mean_groups2

    def count_parameters(self) -> dict:
        def n(m): return sum(p.numel() for p in m.parameters())
        return {
            "patch_embed":  n(self.patch_embed),
            "encoder1":     n(self.encoder1),
            "router1":      n(self.router1),
            "merge1":       n(self.merge1),
            "encoder2":     n(self.encoder2),
            "router2":      n(self.router2),
            "merge2":       n(self.merge2),
            "main_net":     n(self.main_net),
            "norm+head":    n(self.norm) + n(self.classifier),
            "total":        n(self),
        }

#ughhhh.
#new strat. instead of meanpooling border tokens, we keep all of them.
# with all kept border tokens, we can compare borders against borders on the grid still.
# that way, we maintain all spatial congruity.
# islands are basically always kept, and maybe should always be kept down the line.
# inspiration is because currently stage 2 basically destabilizes stage 1 chunking.
# stage 1 chunking is beautiful, but add on stage 2, and its not even stage 2 that grouping too hard,
# but the influence of having that second stage is degenerating stage 1.
# might be related to graph noding, maybe mean pooling. anyhoo, we're moving forward with this.