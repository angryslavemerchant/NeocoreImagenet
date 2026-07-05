"""
model_asfnet2.py — Two-stage hierarchical ASFNet.

Applies grouping twice in sequence:
  Stage 1: 196 tokens  →  196/target_group_size_1 groups  (fixed grid adjacency)
  Stage 2:  G1 tokens  →  G1/target_group_size_2  groups  (k-NN adjacency)

  Default (both=3.0):  196 → ~65 → ~22  (1/9 overall compression)
  Example (4.0, 2.0):  196 → ~49 → ~25  (1/8 overall compression)

Shared components (PatchEmbed, Attention, TransformerBlock, BoundaryRouter,
GroupMerge, gpu_connected_components) are imported directly from model_asfnet
to avoid duplication. New code here covers only Stage 2-specific pieces:
  - build_knn_edges
  - gpu_connected_components_dynamic
  - BoundaryRouter2
  - ASFNet2
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
    Stage 2 boundary router. Identical learned weights to BoundaryRouter
    (Stage 1) but accepts dynamic per-image edges instead of a fixed grid
    buffer, and computes a topology-aware ratio loss that accounts for
    the varying token and edge counts at Stage 2.

    Because token count (n_tokens) and valid edge count (n_edges) vary
    per image at Stage 2 (unlike Stage 1 where every image has exactly
    N=196 tokens and 364 edges), the ratio loss is computed per image
    and averaged — a small Python loop over B images on scalar tensors,
    negligible cost.
    """

    def __init__(self, d_model: int, proj_dim: int = 64):
        super().__init__()
        self.W_q           = nn.Linear(d_model, proj_dim, bias=False)
        self.W_k           = nn.Linear(d_model, proj_dim, bias=False)
        self.score_to_prob = nn.Linear(1, 1)

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
            hard:    (B, E) binary boundary decisions (STE — Straight-Through
                     Estimator, meaning the forward pass uses hard 0/1 decisions
                     while gradients flow through the soft probabilities)
            probs:   (B, E) soft boundary probabilities
            l_ratio: scalar ratio loss averaged across the batch
        """
        q = self.W_q(h)   # (B, G1, proj_dim)
        k = self.W_k(h)   # (B, G1, proj_dim)

        proj_dim = q.shape[-1]
        q_i = q.gather(1, src.unsqueeze(-1).expand(-1, -1, proj_dim))   # (B, E, proj_dim)
        k_j = k.gather(1, dst.unsqueeze(-1).expand(-1, -1, proj_dim))   # (B, E, proj_dim)

        sim   = F.cosine_similarity(q_i, k_j, dim=-1)     # (B, E)
        D     = (1.0 - sim).unsqueeze(-1)                 # (B, E, 1)
        probs = torch.sigmoid(self.score_to_prob(D)).squeeze(-1)   # (B, E)

        # Straight-Through Estimator: hard decision forward, soft gradient backward
        hard = (probs > 0.5).float() + probs - probs.detach()   # (B, E)

        # --- Per-image ratio loss ---
        #
        # Same spanning-forest formula as Stage 1 (see BoundaryRouter in
        # model_asfnet.py for the full derivation), but n_tokens and n_edges
        # are image-specific at Stage 2 because Stage 1 produces a variable
        # number of real groups per image.
        #
        # F_target = (n_edges - n_tokens + G_target) / n_edges
        # where G_target = n_tokens / target_group_size
        #
        # This keeps F_target above the percolation threshold regardless of
        # how many real tokens this particular image has at Stage 2.
        B = h.shape[0]
        l_ratio_total = h.new_zeros(())
        n_valid_images = 0

        for b in range(B):
            n_tok  = float(n_real_per_image[b].item())
            n_edge = float(valid[b].sum().item())

            if n_edge < 2 or n_tok < 2:
                continue

            g_target = n_tok / target_group_size
            f_target = max((n_edge - n_tok + g_target) / n_edge, 1e-6)
            N        = 1.0 / f_target

            if N <= 1.0:
                continue

            # Only compute F_rate and G_rate over valid edges for this image
            valid_b  = valid[b]                           # (E,) bool
            F_rate   = hard[b, valid_b].detach().mean()
            G_rate   = probs[b, valid_b].mean()

            l_ratio_total = l_ratio_total + (N / (N - 1)) * (
                (N - 1) * F_rate * G_rate + (1 - F_rate) * (1 - G_rate)
            )
            n_valid_images += 1

        l_ratio = l_ratio_total / max(n_valid_images, 1)
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
        local_encoder1: bool = False,
        local_radius: int = 1,
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

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, float]:
        """
        Args:
            x: (B, C, H, W)

        Returns:
            logits:       (B, num_classes)
            l_ratio1:     scalar — Stage 1 ratio loss
            l_ratio2:     scalar — Stage 2 ratio loss
            mean_groups1: float  — avg Stage 1 group count (diagnostic)
            mean_groups2: float  — avg Stage 2 group count (diagnostic)
        """
        # ---- Stage 1 ----
        tokens, coords = self.patch_embed(x)   # (B, N1, D)

        for block in self.encoder1:
            tokens = block(tokens, coords)

        hard1, _, l_ratio1 = self.router1(tokens, self.target_group_size_1)

        group_ids1 = gpu_connected_components(
            hard1.detach(), self.router1.edge_indices, tokens.shape[1]
        )

        # padded_tokens1: (B, max_G1, D) — real slots followed by padding
        # pad_mask1:      (B, max_G1)    — True = padding slot
        padded_tokens1, padded_coords1, pad_mask1, mean_groups1 = self.merge1(
            tokens, coords, group_ids1
        )

        # ---- Stage 2 ----
        for block in self.encoder2:
            padded_tokens1 = block(padded_tokens1, padded_coords1, pad_mask1)

        # Real token count per image going into Stage 2 routing.
        # Needed for the per-image ratio loss formula.
        n_real1 = (~pad_mask1).sum(dim=1)   # (B,)

        src2, dst2, valid2 = build_knn_edges(padded_coords1, pad_mask1, self.knn_k)

        hard2, _, l_ratio2 = self.router2(
            padded_tokens1, src2, dst2, valid2,
            self.target_group_size_2, n_real1,
        )

        max_G1     = padded_tokens1.shape[1]
        group_ids2 = gpu_connected_components_dynamic(
            hard2.detach(), valid2, src2, dst2, max_G1
        )

        # padded_tokens2: (B, max_G2, D)
        # GroupMerge's own pad_mask (based on group count) doesn't account for
        # Stage 1 padding propagating through — we recompute it below.
        padded_tokens2, padded_coords2, _, mean_groups2 = self.merge2(
            padded_tokens1, padded_coords1, group_ids2
        )

        # ---- Stage 2 padding mask ----
        #
        # GroupMerge2's built-in pad_mask just marks slots beyond the per-image
        # group count. That's not sufficient here: some of those groups were
        # formed by merging Stage 1 PADDING tokens (isolated because they had no
        # valid edges). Those Stage 2 groups are also padding, even if they sit
        # within the max-group count for the batch.
        #
        # Fix: scatter the real/fake indicator from Stage 1 into Stage 2 group
        # slots. A Stage 2 group is real iff at least one of its contributing
        # Stage 1 tokens was real.
        #
        # real1[b, i] = 1.0  iff Stage 1 slot i is a real group for image b
        # real2_sum[b, g2] = sum of real1[b, i] for all i mapping to g2
        # → pad_mask2[b, g2] = True iff real2_sum == 0 (all-padding group)
        max_G2          = padded_tokens2.shape[1]
        real1           = (~pad_mask1).float()             # (B, max_G1)
        real2_sum       = torch.zeros(x.shape[0], max_G2, device=x.device)
        ids2_clamped    = group_ids2.clamp(0, max_G2 - 1)
        real2_sum.scatter_add_(1, ids2_clamped, real1)
        pad_mask2 = real2_sum < 0.5                        # (B, max_G2)
        mean_groups2 = float((~pad_mask2).sum(dim=1).float().mean().item())

        # ---- Main network + classifier ----
        for block in self.main_net:
            padded_tokens2 = block(padded_tokens2, padded_coords2, pad_mask2)

        padded_tokens2 = self.norm(padded_tokens2)

        real_mask   = (~pad_mask2).float()
        token_sum   = (padded_tokens2 * real_mask.unsqueeze(-1)).sum(dim=1)
        token_count = real_mask.sum(dim=1, keepdim=True).clamp(min=1)
        pooled      = token_sum / token_count

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

