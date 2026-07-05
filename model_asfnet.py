import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ---------------------------------------------------------------------------
# Patch Embedding
# ---------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    """
    Splits a 224×224 image into non-overlapping patches and linearly
    projects each to d_model dimensions.

    Produces a grid_size×grid_size token sequence with fixed (row, col) grid
    coordinates. Coordinates are registered as a buffer so they follow the
    model to whatever device.
    """

    def __init__(
        self,
        image_size:  int = 224,
        patch_size:  int = 16,
        in_channels: int = 3,
        d_model:     int = 256,
    ):
        super().__init__()
        self.grid_size = image_size // patch_size
        self.n_patches = self.grid_size ** 2

        self.proj = nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(d_model)

        rows = torch.arange(self.grid_size).float()
        cols = torch.arange(self.grid_size).float()
        grid_row, grid_col = torch.meshgrid(rows, cols, indexing="ij")
        coords = torch.stack([grid_row.flatten(), grid_col.flatten()], dim=-1)
        self.register_buffer("coords", coords)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.proj(x)                    # (B, D, grid, grid)
        x = x.flatten(2).transpose(1, 2)   # (B, N, D)
        x = self.norm(x)
        return x, self.coords               # tokens: (B, N, D),  coords: (N, 2)


# ---------------------------------------------------------------------------
# 2D Rotary Positional Encoding (RoPE)
# ---------------------------------------------------------------------------

def _build_rope_2d(
    coords: torch.Tensor,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute per-position rotation factors for 2D Rotary Positional Encoding.

    Rotary Positional Encoding encodes position by rotating token vectors in
    embedding space rather than adding a fixed lookup table. This preserves
    relative distance information and works naturally with fractional
    coordinates (which appear after group merging).

    The head dimension is split in two halves:
      dims [0 : head_dim//2]  encode row position
      dims [head_dim//2 : ]   encode column position

    coords:   (..., 2)  (row, col) — any leading dims, can be fractional after group merge
    head_dim: must be divisible by 4

    Returns cos, sin of shape (..., head_dim).
    """
    assert head_dim % 4 == 0, "head_dim must be divisible by 4 for 2D RoPE"
    half    = head_dim // 2
    n_freqs = half // 2

    device = coords.device

    freqs = 1.0 / (
        10000 ** (torch.arange(n_freqs, device=device).float() / n_freqs)
    )

    row = coords[..., 0]
    col = coords[..., 1]

    row_angles = row.unsqueeze(-1) * freqs
    col_angles = col.unsqueeze(-1) * freqs

    row_cos = torch.cos(row_angles).repeat_interleave(2, dim=-1)
    row_sin = torch.sin(row_angles).repeat_interleave(2, dim=-1)
    col_cos = torch.cos(col_angles).repeat_interleave(2, dim=-1)
    col_sin = torch.sin(col_angles).repeat_interleave(2, dim=-1)

    cos = torch.cat([row_cos, col_cos], dim=-1)
    sin = torch.cat([row_sin, col_sin], dim=-1)
    return cos, sin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate adjacent dimension pairs: [x0, x1, x2, x3] → [−x1, x0, −x3, x2]."""
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return torch.stack([-x2, x1], dim=-1).flatten(-2)


def apply_rope_2d(
    q: torch.Tensor,
    k: torch.Tensor,
    coords: torch.Tensor,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply 2D Rotary Positional Encoding rotations to query and key tensors.

    q, k:   (B, num_heads, N, head_dim)
    coords: (N, 2)    shared across batch — used in encoder (integer patch coords)
         or (B, N, 2) per-image — used in main network (fractional group centroid coords)
    """
    cos, sin = _build_rope_2d(coords, head_dim)

    if cos.dim() == 2:
        cos = cos[None, None]
        sin = sin[None, None]
    else:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

    cos = cos.to(q.dtype)
    sin = sin.to(q.dtype)

    q = q * cos + _rotate_half(q) * sin
    k = k * cos + _rotate_half(k) * sin
    return q, k


# ---------------------------------------------------------------------------
# Multi-Head Self-Attention with 2D RoPE
# ---------------------------------------------------------------------------

class Attention(nn.Module):
    """
    Multi-head self-attention where 2D Rotary Positional Encoding is applied
    directly to queries and keys.

    Uses F.scaled_dot_product_attention which dispatches to FlashAttention
    when available.
    """

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = d_model // num_heads

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out  = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        coords: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, D = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        def to_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        q, k, v = to_heads(q), to_heads(k), to_heads(v)

        q, k = apply_rope_2d(q, k, coords, self.head_dim)

        bias = None
        if attn_mask is not None:
            bias = torch.zeros(B, 1, 1, N, device=x.device, dtype=q.dtype)
            bias = bias.masked_fill(attn_mask[:, None, None, :], float("-inf"))

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        out = out.transpose(1, 2).reshape(B, N, D)
        return self.out(out)


# ---------------------------------------------------------------------------
# Transformer Block (pre-norm)
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """
    Standard Vision Transformer block using pre-norm.

    Pre-norm means LayerNorm (Layer Normalization — a technique that
    normalizes the activations within each token to have zero mean and
    unit variance, stabilizing training) is applied before each sub-layer
    rather than after.

    Structure: LN → Attention → residual → LN → FFN → residual
    """

    def __init__(self, d_model: int, num_heads: int, mlp_ratio: float = 3.0):
        super().__init__()
        ffn_dim = int(d_model * mlp_ratio)

        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = Attention(d_model, num_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, d_model),
        )

    def forward(
        self,
        x: torch.Tensor,
        coords: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), coords, attn_mask)
        x = x + self.ffn(self.norm2(x))
        return x

# =============================================================================
# Local-neighbourhood attention for encoder1 (additive — paste into
# model_asfnet.py, below the existing Attention / TransformerBlock defs).
#
# These reuse the existing apply_rope_2d and mirror Attention/TransformerBlock
# exactly, differing only in that each token may attend to a fixed square
# window around it on the patch grid instead of the whole image.
#
# Nothing here touches existing classes, so current ASFNet / ASFNet2 runs and
# checkpoints are unaffected. ASFNet2 opts in via a flag (see the __init__ diff
# in the accompanying message).
# =============================================================================


def _build_neighborhood_mask(grid_size: int, radius: int) -> torch.Tensor:
    """
    Boolean (N, N) attention mask for a grid_size x grid_size patch grid.

    Entry (i, j) is True iff patch j lies within Chebyshev distance `radius`
    of patch i — i.e. inside the (2*radius+1) square window centred on i.
    True = "j participates in i's attention" (SDPA bool-mask convention).
    Self is always included (distance 0), so every row has at least one True
    and no token can produce a NaN from an all-masked row.

    Token ordering matches PatchEmbed: index = row * grid_size + col
    (row-major flatten of the (grid, grid) feature map), so the mask lines up
    with the token sequence without any reindexing.
    """
    coords = torch.stack(
        torch.meshgrid(
            torch.arange(grid_size),
            torch.arange(grid_size),
            indexing="ij",
        ),
        dim=-1,
    ).reshape(-1, 2)                                   # (N, 2) integer (row, col)

    diff = (coords[:, None, :] - coords[None, :, :]).abs()   # (N, N, 2)
    cheb = diff.max(dim=-1).values                           # (N, N) Chebyshev dist
    return cheb <= radius                                    # (N, N) bool


class LocalAttention(nn.Module):
    """
    Multi-head self-attention restricted to a local grid neighbourhood.

    Structurally identical to Attention (same qkv/out projections, same 2D
    rotary positional encoding applied to q and k) except each token may only
    attend to tokens inside a fixed square window on the patch grid. The window
    is supplied as a boolean (N, N) mask (True = allowed) and broadcast over
    batch and heads.

    Position is still encoded globally via RoPE — every token carries its true
    grid coordinate. The mask only restricts *which pairs interact*, not where
    each token thinks it is.
    """

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = d_model // num_heads

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        coords: torch.Tensor,
        neighborhood: torch.Tensor,
    ) -> torch.Tensor:
        B, N, D = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        def to_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        q, k, v = to_heads(q), to_heads(k), to_heads(v)
        q, k = apply_rope_2d(q, k, coords, self.head_dim)

        # neighborhood: (N, N) bool → (1, 1, N, N), broadcasts over (B, heads).
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=neighborhood[None, None])
        out = out.transpose(1, 2).reshape(B, N, D)
        return self.out(out)


class LocalTransformerBlock(nn.Module):
    """
    Pre-norm transformer block whose attention is restricted to a local grid
    neighbourhood (see LocalAttention). Feed-forward is unchanged.

    Owns its neighbourhood mask as a non-persistent buffer, built once from
    (grid_size, radius). Non-persistent so it is *not* written into the
    checkpoint — it is fully derivable from config, so checkpoints stay clean
    and changing `radius` later can't cause a state_dict shape mismatch.

    forward takes (x, coords) only. encoder1 processes all 196 patch tokens
    with no padding, so there is no padding mask to thread through — which is
    exactly why this is a drop-in for the existing `block(tokens, coords)` call.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        grid_size: int,
        radius: int,
        mlp_ratio: float = 3.0,
    ):
        super().__init__()
        ffn_dim = int(d_model * mlp_ratio)

        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = LocalAttention(d_model, num_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, d_model),
        )

        self.register_buffer(
            "neighborhood",
            _build_neighborhood_mask(grid_size, radius),
            persistent=False,
        )

    def forward(self, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), coords, self.neighborhood)
        x = x + self.ffn(self.norm2(x))
        return x



# ---------------------------------------------------------------------------
# Boundary Router
# ---------------------------------------------------------------------------

def _build_edge_indices(grid_size: int) -> torch.Tensor:
    """
    Precompute all 4-connected adjacent token pairs for a grid_size×grid_size grid.
    Only right and down neighbours are stored (each undirected edge appears once).
    Returns (E, 2) tensor of (i, j) index pairs.
    """
    edges = []
    for r in range(grid_size):
        for c in range(grid_size):
            idx = r * grid_size + c
            if c + 1 < grid_size:
                edges.append((idx, idx + 1))
            if r + 1 < grid_size:
                edges.append((idx, idx + grid_size))
    return torch.tensor(edges, dtype=torch.long)


class BoundaryRouter(nn.Module):
    """
    Scores every adjacent token pair and decides where group boundaries fall.

    Per edge (i, j):
      1. Project:  q_i = W_q · h_i,   k_j = W_k · h_j
      2. Score:    D(i,j) = 1 − cosine_similarity(q_i, k_j)
      3. Prob:     p(i,j) = sigmoid(linear(D(i,j)))

    Uses a Straight-Through Estimator (forward pass makes a hard 0/1
    boundary decision, backward pass lets gradients flow through the soft
    probability as if the hard threshold never happened, keeping the router
    trainable despite the discrete decision).

    The ratio loss prevents the router collapsing to all-boundaries or
    one-giant-group. Its target boundary fraction is derived from the actual
    grid topology rather than a fixed fraction so it works correctly across
    different patch sizes — see the comment in forward() for the derivation.
    """

    def __init__(self, d_model: int, proj_dim: int = 64, grid_size: int = 14):
        super().__init__()
        self.W_q           = nn.Linear(d_model, proj_dim, bias=False)
        self.W_k           = nn.Linear(d_model, proj_dim, bias=False)
        self.score_to_prob = nn.Linear(1, 1)

        edges = _build_edge_indices(grid_size)
        self.register_buffer("edge_indices", edges)

    def forward(
        self,
        h: torch.Tensor,
        target_group_size: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        h:                 (B, N, D) encoder hidden states
        target_group_size: desired average tokens per group

        Returns:
            hard:    (B, E) binary boundary decisions with Straight-Through gradient
            probs:   (B, E) soft boundary probabilities
            l_ratio: scalar ratio loss
        """
        idx_i = self.edge_indices[:, 0]
        idx_j = self.edge_indices[:, 1]

        q = self.W_q(h)
        k = self.W_k(h)

        q_i = q[:, idx_i, :]
        k_j = k[:, idx_j, :]

        sim   = F.cosine_similarity(q_i, k_j, dim=-1)
        D     = (1.0 - sim).unsqueeze(-1)
        probs = torch.sigmoid(self.score_to_prob(D)).squeeze(-1)

        hard = (probs > 0.5).float() + probs - probs.detach()

        # --- Ratio loss ---
        #
        # The original formulation used N = target_group_size directly,
        # targeting F = 1/N of edges as boundaries. This breaks on different
        # patch sizes because the same boundary fraction produces different
        # group counts depending on how many edges the grid has relative to
        # its token count.
        #
        # Fix: derive target boundary fraction from actual grid topology.
        #
        # Spanning forest argument: a forest with G trees on T nodes needs
        # exactly T-G internal (non-boundary) edges. Everything else is a
        # boundary. So:
        #
        #   F_target = (n_edges - n_tokens + G_target) / n_edges
        #   G_target = n_tokens / target_group_size
        #
        # This gives F_target ≈ 0.64 for both 14×14 and 28×28 grids.
        # Critically, this is above 0.5 — the percolation threshold (the
        # point below which a giant connected component forms and swallows
        # most tokens into a few huge groups). The original N=3 targeted
        # F=0.33, well below this threshold, which is why 8×8 patches
        # collapsed to ~100 groups instead of ~261.
        n_tokens = h.shape[1]
        n_edges  = self.edge_indices.shape[0]
        g_target = n_tokens / target_group_size
        f_target = (n_edges - n_tokens + g_target) / n_edges
        f_target = max(f_target, 1e-6)
        N        = 1.0 / f_target

        F_rate = hard.detach().mean(dim=-1)
        G_rate = probs.mean(dim=-1)

        l_ratio = (N / (N - 1)) * (
            (N - 1) * F_rate * G_rate
            + (1 - F_rate) * (1 - G_rate)
        )
        l_ratio = l_ratio.mean()

        return hard, probs, l_ratio


# ---------------------------------------------------------------------------
# GPU-native Connected Components via Iterative Label Propagation
# ---------------------------------------------------------------------------

def gpu_connected_components(
    hard: torch.Tensor,
    edge_indices: torch.Tensor,
    n_tokens: int,
) -> torch.Tensor:
    """
    Finds connected components (groups of tokens with no boundary between them)
    entirely on the GPU using iterative label propagation.

    Replaces the previous CPU union-find + ThreadPoolExecutor approach, which
    had two hard sync points (GPU→CPU transfer before union-find, CPU→GPU
    transfer after) that forced the GPU to completely stall every forward pass.
    This implementation never leaves the GPU.

    Algorithm:
      1. Each token starts labelled with its own index.
      2. On each iteration, every token takes the minimum label of its
         non-boundary neighbours via scatter_reduce with 'amin' (scatter-reduce
         is a GPU operation that accumulates values into target index positions
         — here taking the minimum across all edges pointing to each token).
      3. After enough iterations, all tokens in a connected component share the
         same label (the minimum token index in that component).
      4. Labels are remapped to contiguous IDs [0, n_groups) per image using
         a sort + cumsum — both are fast GPU primitives.

    Convergence: a label travels one edge per iteration, so a component whose
    longest internal path (diameter) is K needs up to K iterations to fully
    settle. The loop runs until no label changes anywhere in the batch, so
    every image is guaranteed fully converged before returning — this avoids
    the non-deterministic group splitting that a fixed iteration count caused.
    Small groups (target_group_size≈3) converge in 2-3 iterations, so the
    common case stays fast.

    Args:
        hard:         (B, E) hard boundary decisions — 1=boundary, 0=connected
        edge_indices: (E, 2) adjacency pairs, registered buffer from BoundaryRouter
        n_tokens:     N — number of tokens per image

    Returns:
        labels: (B, N) int64 — contiguous group IDs in [0, n_groups_per_image)
    """
    B      = hard.shape[0]
    device = hard.device

    idx_i     = edge_indices[:, 0]   # (E,)
    idx_j     = edge_indices[:, 1]
    connected = hard < 0.5           # (B, E) — True means same group (no boundary)

    # Each token starts as its own group, labelled by its own index.
    # After propagation, all tokens in a component converge to the minimum index.
    labels = (
        torch.arange(n_tokens, device=device)
        .unsqueeze(0)
        .expand(B, -1)
        .clone()
    )  # (B, N)

    # INF sentinel: larger than any valid label, so boundary edges contribute
    # nothing to the amin reduction (they get ignored).
    INF = n_tokens

    # Pre-expand edge index tensors once — reused every iteration
    idx_i_exp = idx_i.unsqueeze(0).expand(B, -1)  # (B, E)
    idx_j_exp = idx_j.unsqueeze(0).expand(B, -1)

    # Loop until convergence rather than a fixed count.
    #
    # A label travels at most one edge per iteration, so a stringy component
    # of diameter K (longest internal path) needs up to K iterations to fully
    # settle. On a 28×28 grid that diameter can be far larger than log2(N),
    # so a fixed iteration count silently under-converges some images —
    # splitting a single component into several groups on some steps but not
    # others. That non-determinism was the source of the training jitter.
    #
    # n_tokens is a hard upper bound on the diameter (a path can't be longer
    # than the token count), so this loop is guaranteed to terminate and
    # guaranteed to fully converge every image before returning.
    for _ in range(n_tokens):
        label_i = labels[:, idx_i]   # (B, E) — current label at each edge's source
        label_j = labels[:, idx_j]   # (B, E) — current label at each edge's dest

        # The label to spread through each edge: minimum of both endpoints.
        # For boundary edges, use INF so they don't affect the amin reduction.
        min_ij    = torch.minimum(label_i, label_j)                      # (B, E)
        propagate = torch.where(connected, min_ij, min_ij.new_full((), INF))  # (B, E)

        # Scatter the minimum label to both endpoints of every connected edge.
        # scatter_reduce_ with 'amin' and include_self=True takes the minimum
        # of the token's existing label and everything scattered to it this step.
        new_labels = labels.clone()
        new_labels.scatter_reduce_(1, idx_j_exp, propagate, reduce='amin', include_self=True)
        new_labels.scatter_reduce_(1, idx_i_exp, propagate, reduce='amin', include_self=True)

        # Stop as soon as no label changed anywhere in the batch — the
        # components have fully settled. This keeps the common case (small
        # groups, converges in 2-3 iterations) fast while still handling
        # worst-case stringy components correctly.
        if torch.equal(new_labels, labels):
            labels = new_labels
            break
        labels = new_labels

    # --- Remap non-contiguous root labels → contiguous IDs [0, n_groups) ---
    #
    # After propagation, labels are the minimum token index in each component
    # (e.g. [0, 0, 5, 5, 3, 3, ...]). We need contiguous IDs (e.g. [0, 0, 2, 2, 1, 1])
    # so that GroupMerge's scatter and pad-mask arithmetic work correctly.
    #
    # Method: sort tokens by label within each image. Wherever the label changes
    # in the sorted order, that's a new group. cumsum over those change-points
    # gives contiguous IDs in sorted order; scatter puts them back in token order.

    sorted_labels, sort_idx = labels.sort(dim=1)          # (B, N)

    is_new_group = torch.cat([
        torch.ones(B, 1, device=device, dtype=torch.bool),
        sorted_labels[:, 1:] != sorted_labels[:, :-1],
    ], dim=1)                                              # (B, N) — True at each group's first token

    contiguous_sorted = is_new_group.long().cumsum(dim=1) - 1  # (B, N) — IDs in sorted order

    # Unsort: put contiguous IDs back at original token positions
    contiguous_labels = torch.empty_like(labels)
    contiguous_labels.scatter_(1, sort_idx, contiguous_sorted)  # (B, N)

    return contiguous_labels


# ---------------------------------------------------------------------------
# Group Merging + Projection
# ---------------------------------------------------------------------------

# Fixed stride between images in the flattened group index space.
# Must exceed the maximum possible groups per image (= n_tokens, when every
# token is isolated). 1000 covers up to 28×28 = 784 tokens with a buffer.
_GROUP_STRIDE = 1000


class GroupMerge(nn.Module):
    """
    Merges each connected-component group into a single token via mean pooling,
    then applies a linear projection.

    Accepts a (B, N) integer tensor of contiguous group IDs directly from
    gpu_connected_components — no Python loops, no CPU involvement.

    Vectorized via a fixed-stride offset trick: image b's group IDs are shifted
    by b * _GROUP_STRIDE so all B*N tokens can be processed in a single
    scatter_add_ call instead of B separate ones.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        h: torch.Tensor,
        coords: torch.Tensor,
        group_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        h:         (B, N, D) encoder hidden states
        coords:    (N, 2)    patch grid coordinates
        group_ids: (B, N)    contiguous group IDs from gpu_connected_components

        Returns:
            padded_tokens: (B, max_G, D)
            padded_coords: (B, max_G, 2) — group centroid coords for RoPE
            pad_mask:      (B, max_G) bool — True = padding token
            mean_groups:   float — avg group count across batch (diagnostic)
        """
        B, N, D = h.shape
        device  = h.device

        n_groups_per_image = group_ids.max(dim=1).values + 1  # (B,)

        # Make group IDs globally unique across the batch to enable a single
        # scatter_add_ call for all B*N tokens at once.
        offsets    = torch.arange(B, device=device, dtype=torch.long) * _GROUP_STRIDE
        global_ids = group_ids + offsets.unsqueeze(1)  # (B, N)

        flat_h   = h.reshape(B * N, D)
        flat_ids = global_ids.reshape(B * N)

        feat_sum = torch.zeros(B * _GROUP_STRIDE, D, device=device, dtype=h.dtype)
        counts   = torch.zeros(B * _GROUP_STRIDE,    device=device, dtype=h.dtype)
        feat_sum.scatter_add_(0, flat_ids.unsqueeze(-1).expand(-1, D), flat_h)
        counts.scatter_add_(0, flat_ids, torch.ones(B * N, device=device, dtype=h.dtype))

        group_feats = (feat_sum / counts.unsqueeze(-1).clamp(min=1)).view(B, _GROUP_STRIDE, D)

        max_G         = int(n_groups_per_image.max().item())
        padded_tokens = self.proj(group_feats[:, :max_G, :])  # (B, max_G, D)

        # Pad mask: True = padding slot, False = real group.
        # Broadcasting comparison — no loop needed.
        arange   = torch.arange(max_G, device=device).unsqueeze(0)  # (1, max_G)
        pad_mask = arange >= n_groups_per_image.unsqueeze(1)         # (B, max_G)

        with torch.no_grad():
            coord_sum = torch.zeros(B * _GROUP_STRIDE, 2, device=device)
            # coords is (N, 2) at Stage 1 (shared patch grid, same for every image)
            # but (B, N, 2) at Stage 2 (per-image fractional group centroids).
            # Handle both cases before flattening to (B*N, 2) for scatter_add_.
            if coords.dim() == 2:
                coords_flat = coords.unsqueeze(0).expand(B, -1, -1).reshape(B * N, 2).float()
            else:
                coords_flat = coords.reshape(B * N, 2).float()
            coord_sum.scatter_add_(
                0,
                flat_ids.unsqueeze(-1).expand(-1, 2),
                coords_flat,
            )
            group_coords  = (coord_sum / counts.float().unsqueeze(-1).clamp(min=1)).view(B, _GROUP_STRIDE, 2)
            padded_coords = group_coords[:, :max_G, :]  # (B, max_G, 2)

        mean_groups = float(n_groups_per_image.float().mean().item())
        return padded_tokens, padded_coords, pad_mask, mean_groups


# ---------------------------------------------------------------------------
# ASFNet
# ---------------------------------------------------------------------------

class ASFNet(nn.Module):
    """
    Attention-Shifted Focus Network.

    Pipeline:
      PatchEmbed                          N tokens (grid_size × grid_size)
      → encoder_blocks × TransformerBlock build semantic representations
      → BoundaryRouter                    score every adjacent pair
      → gpu_connected_components          label propagation entirely on GPU
      → GroupMerge                        mean-pool + project → adaptive token count
      → main_blocks × TransformerBlock    reason over compressed token set
      → masked Global Average Pool
      → Linear classifier
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
        router_proj_dim:   int   = 64,
    ):
        super().__init__()
        self.target_group_size = target_group_size
        grid_size = image_size // patch_size

        self.patch_embed = PatchEmbed(image_size, patch_size, in_channels, d_model)

        self.encoder = nn.ModuleList([
            TransformerBlock(d_model, num_heads, mlp_ratio)
            for _ in range(encoder_blocks)
        ])

        self.router      = BoundaryRouter(d_model, router_proj_dim, grid_size)
        self.group_merge = GroupMerge(d_model)

        self.main_net = nn.ModuleList([
            TransformerBlock(d_model, num_heads, mlp_ratio)
            for _ in range(main_blocks)
        ])

        self.norm       = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        """
        x: (B, C, H, W)

        Returns:
            logits:      (B, num_classes)
            l_ratio:     scalar — ratio loss to add to task loss during training
            mean_groups: float  — avg group tokens per image (diagnostic)
        """
        tokens, coords = self.patch_embed(x)

        for block in self.encoder:
            tokens = block(tokens, coords)

        hard, probs, l_ratio = self.router(tokens, self.target_group_size)

        # Connected components entirely on GPU — no CPU transfers, no stalls
        group_ids = gpu_connected_components(
            hard.detach(),
            self.router.edge_indices,
            tokens.shape[1],
        )

        padded_tokens, padded_coords, pad_mask, mean_groups = self.group_merge(
            tokens, coords, group_ids,
        )

        for block in self.main_net:
            padded_tokens = block(padded_tokens, padded_coords, pad_mask)

        padded_tokens = self.norm(padded_tokens)

        real_mask   = (~pad_mask).float()
        token_sum   = (padded_tokens * real_mask.unsqueeze(-1)).sum(dim=1)
        token_count = real_mask.sum(dim=1, keepdim=True).clamp(min=1)
        pooled      = token_sum / token_count

        logits = self.classifier(pooled)
        return logits, l_ratio, mean_groups

    def count_parameters(self) -> dict:
        def n(m): return sum(p.numel() for p in m.parameters())
        return {
            "patch_embed":  n(self.patch_embed),
            "encoder":      n(self.encoder),
            "router":       n(self.router),
            "group_merge":  n(self.group_merge),
            "main_net":     n(self.main_net),
            "norm+head":    n(self.norm) + n(self.classifier),
            "total":        n(self),
        }