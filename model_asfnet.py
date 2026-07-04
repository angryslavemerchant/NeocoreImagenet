import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ---------------------------------------------------------------------------
# Patch Embedding
# ---------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    """
    Splits a 224×224 image into non-overlapping 16×16 patches and linearly
    projects each to d_model dimensions.

    Produces a 14×14 = 196 token sequence with fixed (row, col) grid coordinates.
    Coordinates are registered as a buffer so they follow the model to whatever device.
    """

    def __init__(
        self,
        image_size:  int = 224,
        patch_size:  int = 16,
        in_channels: int = 3,
        d_model:     int = 256,
    ):
        super().__init__()
        self.grid_size = image_size // patch_size  # 14
        self.n_patches = self.grid_size ** 2       # 196

        # Conv2d with kernel=stride=patch_size tiles the image without overlap —
        # equivalent to flatten-then-linear but faster
        self.proj = nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(d_model)

        # Precompute integer (row, col) grid coordinates once; shape (196, 2)
        rows = torch.arange(self.grid_size).float()
        cols = torch.arange(self.grid_size).float()
        grid_row, grid_col = torch.meshgrid(rows, cols, indexing="ij")
        coords = torch.stack([grid_row.flatten(), grid_col.flatten()], dim=-1)
        self.register_buffer("coords", coords)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, C, H, W)
        x = self.proj(x)                    # (B, D, grid, grid)
        x = x.flatten(2).transpose(1, 2)   # (B, 196, D)
        x = self.norm(x)
        return x, self.coords               # tokens: (B, 196, D),  coords: (196, 2)


# ---------------------------------------------------------------------------
# 2D Rotary Positional Encoding (RoPE)
# ---------------------------------------------------------------------------

def _build_rope_2d(
    coords: torch.Tensor,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute per-position rotation factors for 2D RoPE (Rotary Positional Encoding —
    encodes position by rotating token vectors in embedding space rather than adding
    a lookup table, which preserves relative distance information across any sequence length).

    The head dimension is split in two halves:
      dims [0 : head_dim//2]  encode row position
      dims [head_dim//2 : ]   encode column position

    coords:   (..., 2)  (row, col) — any leading dims, can be fractional after group merge
    head_dim: must be divisible by 4

    Returns cos, sin of shape (..., head_dim).
    """
    assert head_dim % 4 == 0, "head_dim must be divisible by 4 for 2D RoPE"
    half    = head_dim // 2   # dims allocated per spatial axis
    n_freqs = half // 2       # frequency pairs per axis

    device = coords.device

    # Geometric sequence of inverse frequencies: θ_i = 1 / 10000^(i / n_freqs)
    freqs = 1.0 / (
        10000 ** (torch.arange(n_freqs, device=device).float() / n_freqs)
    )

    row = coords[..., 0]   # (...)
    col = coords[..., 1]

    # (..., n_freqs) via broadcasting
    row_angles = row.unsqueeze(-1) * freqs
    col_angles = col.unsqueeze(-1) * freqs

    # repeat_interleave so adjacent dim pairs share one angle,
    # matching the _rotate_half pairing below
    row_cos = torch.cos(row_angles).repeat_interleave(2, dim=-1)  # (..., half)
    row_sin = torch.sin(row_angles).repeat_interleave(2, dim=-1)
    col_cos = torch.cos(col_angles).repeat_interleave(2, dim=-1)
    col_sin = torch.sin(col_angles).repeat_interleave(2, dim=-1)

    cos = torch.cat([row_cos, col_cos], dim=-1)  # (..., head_dim)
    sin = torch.cat([row_sin, col_sin], dim=-1)
    return cos, sin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate adjacent dimension pairs: [x0, x1, x2, x3] → [−x1, x0, −x3, x2]."""
    x1 = x[..., 0::2]  # even dims
    x2 = x[..., 1::2]  # odd dims
    return torch.stack([-x2, x1], dim=-1).flatten(-2)


def apply_rope_2d(
    q: torch.Tensor,
    k: torch.Tensor,
    coords: torch.Tensor,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply 2D RoPE rotations to query and key tensors.

    q, k:   (B, num_heads, N, head_dim)
    coords: (N, 2)    shared across batch — used in encoder (integer patch coords)
         or (B, N, 2) per-image — used in main network (fractional group centroid coords)

    Casts rotation factors to match q/k dtype (important for bfloat16 training).
    """
    cos, sin = _build_rope_2d(coords, head_dim)  # (N, D) or (B, N, D)

    # Broadcast over batch and head dims
    if cos.dim() == 2:
        cos = cos[None, None]   # (1, 1, N, D)
        sin = sin[None, None]
    else:
        cos = cos.unsqueeze(1)  # (B, 1, N, D)
        sin = sin.unsqueeze(1)

    # Cast to match the compute dtype (e.g. bfloat16 under autocast)
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
    Multi-head self-attention where 2D RoPE is applied directly to queries and keys.

    Positional information is encoded as relative rotations between q and k,
    not as additive embeddings on input tokens. This means position-encoding
    is always relative — a useful property after adaptive group merging where
    group centroid coordinates can be fractional and non-uniform.

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
        """
        x:         (B, N, D)
        coords:    (N, 2) or (B, N, 2)
        attn_mask: (B, N) bool — True marks padding tokens to be ignored in attention
        """
        B, N, D = x.shape

        qkv = self.qkv(x)               # (B, N, 3D)
        q, k, v = qkv.chunk(3, dim=-1)

        def to_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        q, k, v = to_heads(q), to_heads(k), to_heads(v)  # (B, heads, N, head_dim)

        # Apply 2D RoPE rotations to queries and keys
        q, k = apply_rope_2d(q, k, coords, self.head_dim)

        # Build additive float bias from boolean padding mask.
        # Shape (B, 1, 1, N): adds -inf to padding key positions so they attract no attention.
        bias = None
        if attn_mask is not None:
            bias = torch.zeros(B, 1, 1, N, device=x.device, dtype=q.dtype)
            bias = bias.masked_fill(attn_mask[:, None, None, :], float("-inf"))

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)  # (B, heads, N, head_dim)
        out = out.transpose(1, 2).reshape(B, N, D)                     # merge heads
        return self.out(out)


# ---------------------------------------------------------------------------
# Transformer Block (pre-norm)
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """
    Standard ViT (Vision Transformer) transformer block using pre-norm.

    Pre-norm means LayerNorm is applied before each sub-layer (not after),
    which is more stable to train than post-norm at moderate depth.

    Structure:  LN → Attention (with 2D RoPE) → residual → LN → FFN → residual

    The same block definition is reused in both the encoder and the main network.
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
                edges.append((idx, idx + 1))           # right
            if r + 1 < grid_size:
                edges.append((idx, idx + grid_size))   # down
    return torch.tensor(edges, dtype=torch.long)        # (E, 2)


class BoundaryRouter(nn.Module):
    """
    Scores every adjacent token pair and decides where group boundaries fall.

    Per edge (i, j):
      1. Project:  q_i = W_q · h_i,   k_j = W_k · h_j
      2. Score:    D(i,j) = 1 − cosine_similarity(q_i, k_j)   [scalar dissimilarity]
      3. Prob:     p(i,j) = sigmoid(linear(D(i,j)))

    W_q and W_k are kept small (proj_dim = 64 vs d_model = 256) so the router
    learns which subspace of the representation is relevant for boundary detection,
    without adding many parameters.

    Straight-Through Estimator (STE — forward pass uses the hard 0/1 threshold
    decision, backward pass lets gradients flow through as if the operation were
    continuous) converts soft probabilities to binary boundary decisions.

    Also computes L_ratio, the ratio loss that prevents the router collapsing to
    "keep all tokens" (no compression) or "one giant group" (over-compression).
    """

    def __init__(self, d_model: int, proj_dim: int = 64, grid_size: int = 14):
        super().__init__()
        self.W_q           = nn.Linear(d_model, proj_dim, bias=False)
        self.W_k           = nn.Linear(d_model, proj_dim, bias=False)
        # Learnable affine transform on the scalar cosine distance before sigmoid
        self.score_to_prob = nn.Linear(1, 1)

        edges = _build_edge_indices(grid_size)
        self.register_buffer("edge_indices", edges)  # (E, 2) — fixed structure

    def forward(
        self,
        h: torch.Tensor,
        target_group_size: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        h:                 (B, N, D) encoder hidden states
        target_group_size: target average tokens per group (N in the ratio loss)

        Returns:
            hard:    (B, E) binary boundary decisions with STE gradient attachment
            probs:   (B, E) soft boundary probabilities
            l_ratio: scalar ratio loss
        """
        idx_i = self.edge_indices[:, 0]   # (E,)
        idx_j = self.edge_indices[:, 1]

        # Project all token positions at once, then index into edge endpoints
        q = self.W_q(h)                   # (B, N, proj_dim)
        k = self.W_k(h)

        q_i = q[:, idx_i, :]             # (B, E, proj_dim)
        k_j = k[:, idx_j, :]

        # Dissimilarity score per edge: 0 = identical direction, 2 = opposite
        sim = F.cosine_similarity(q_i, k_j, dim=-1)   # (B, E)
        D   = (1.0 - sim).unsqueeze(-1)                # (B, E, 1)

        # Soft boundary probability via learned affine + sigmoid
        probs = torch.sigmoid(self.score_to_prob(D)).squeeze(-1)  # (B, E)

        # Hard decision via STE:
        #   forward:  hard = threshold(probs)  — actual 0 or 1
        #   backward: gradient flows through probs as if hard = probs
        hard = (probs > 0.5).float() + probs - probs.detach()    # (B, E)

        # --- Ratio loss ---
        # F = hard boundary rate — non-differentiable, used as a diagnostic signal
        # G = mean soft boundary probability — differentiable, this is the gradient dial
        # Coupling: if too many hard boundaries fire (F > 1/N_target), gradient on G
        # pushes it down → router produces fewer boundaries next step. And vice versa.
        # Formula adapted from H-Net; minimum is at F = G = 1 / target_group_size.
        N = target_group_size
        F_rate = hard.detach().mean(dim=-1)   # (B,) — per-image hard boundary rate
        G_rate = probs.mean(dim=-1)           # (B,) — per-image mean soft probability

        l_ratio = (N / (N - 1)) * (
            (N - 1) * F_rate * G_rate
            + (1 - F_rate) * (1 - G_rate)
        )
        l_ratio = l_ratio.mean()              # scalar

        return hard, probs, l_ratio


# ---------------------------------------------------------------------------
# Connected Components (union-find, CPU, per image)
# ---------------------------------------------------------------------------

def _union_find(n_tokens: int, edges: list, boundaries: list) -> list[int]:
    """
    Union-Find with path compression to find connected components.

    Two tokens are connected (same group) if the boundary between them is 0.
    Returns a list of length n_tokens where each entry is a contiguous group id.
    """
    parent = list(range(n_tokens))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]   # halving path compression
            x = parent[x]
        return x

    def union(x: int, y: int):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for (i, j), b in zip(edges, boundaries):
        if b == 0:          # no boundary → merge into same group
            union(i, j)

    # Remap roots to contiguous ids starting at 0
    root_to_id: dict[int, int] = {}
    result = []
    for i in range(n_tokens):
        root = find(i)
        if root not in root_to_id:
            root_to_id[root] = len(root_to_id)
        result.append(root_to_id[root])

    return result


def find_groups(
    hard: torch.Tensor,
    edge_indices: torch.Tensor,
    n_tokens: int,
) -> list[list[int]]:
    """
    Run connected components for every image in the batch.

    hard:         (B, E) detached hard boundary decisions (0 = same group, 1 = boundary)
    edge_indices: (E, 2)

    Returns list of B group-assignment lists, each of length n_tokens.
    The grid is small (196 tokens, 364 edges) so the Python loop is fast.
    """
    edges     = edge_indices.tolist()
    hard_list = hard.cpu().tolist()    # detach happens at call site in ASFNet.forward

    return [
        _union_find(n_tokens, edges, hard_list[b])
        for b in range(hard.shape[0])
    ]


# ---------------------------------------------------------------------------
# Group Merging + Projection
# ---------------------------------------------------------------------------

class GroupMerge(nn.Module):
    """
    Merges each connected-component group into a single token via mean pooling,
    then applies a linear projection to decouple the encoder and main-network spaces.

    Token value:    differentiable mean pool of constituent encoder hidden states
                    (gradients flow back through scatter_add → encoder)
    Token position: mean of constituent (row, col) coordinates
                    (not differentiable — only used for RoPE in the main network)

    Sequences are padded to the batch-maximum group count so downstream transformer
    blocks can process the whole batch in one forward pass.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        h: torch.Tensor,
        coords: torch.Tensor,
        all_groups: list[list[int]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        h:          (B, N, D) encoder hidden states
        coords:     (N, 2)    patch grid coordinates
        all_groups: B group-assignment lists, each of length N

        Returns:
            padded_tokens: (B, max_G, D)
            padded_coords: (B, max_G, 2) — group centroid coords for RoPE
            pad_mask:      (B, max_G) bool — True = padding token
            mean_groups:   float — avg group count across batch (diagnostic)
        """
        B, N, D = h.shape
        device   = h.device

        batch_feats:  list[torch.Tensor] = []
        batch_coords: list[torch.Tensor] = []

        for b in range(B):
            group_ids = torch.tensor(all_groups[b], device=device, dtype=torch.long)  # (N,)
            n_groups  = int(group_ids.max().item()) + 1

            # --- Differentiable mean pool of features ---
            feat_sum = torch.zeros(n_groups, D, device=device, dtype=h.dtype)
            counts   = torch.zeros(n_groups, device=device, dtype=h.dtype)
            feat_sum.scatter_add_(0, group_ids.unsqueeze(-1).expand(-1, D), h[b])
            counts.scatter_add_(0, group_ids, torch.ones(N, device=device, dtype=h.dtype))
            group_feats = feat_sum / counts.unsqueeze(-1)              # (n_groups, D)

            # --- Mean pool coordinates (no grad needed — only used for RoPE) ---
            with torch.no_grad():
                coord_sum = torch.zeros(n_groups, 2, device=device)
                coord_sum.scatter_add_(
                    0,
                    group_ids.unsqueeze(-1).expand(-1, 2),
                    coords.float(),
                )
                group_coords = coord_sum / counts.float().unsqueeze(-1)  # (n_groups, 2)

            batch_feats.append(group_feats)
            batch_coords.append(group_coords)

        # --- Pad to max group count in this batch ---
        group_sizes = [f.shape[0] for f in batch_feats]
        max_G       = max(group_sizes)
        mean_groups = sum(group_sizes) / B

        padded_tokens = torch.zeros(B, max_G, D, device=device, dtype=h.dtype)
        padded_coords = torch.zeros(B, max_G, 2, device=device, dtype=torch.float32)
        pad_mask      = torch.ones(B, max_G, device=device, dtype=torch.bool)

        for b, (feats, crds, G) in enumerate(zip(batch_feats, batch_coords, group_sizes)):
            padded_tokens[b, :G] = feats
            padded_coords[b, :G] = crds
            pad_mask[b, :G]      = False   # True = padding, False = real token

        # Single batched projection — applied to padding tokens too but they're masked downstream
        padded_tokens = self.proj(padded_tokens)

        return padded_tokens, padded_coords, pad_mask, mean_groups


# ---------------------------------------------------------------------------
# ASFNet
# ---------------------------------------------------------------------------

class ASFNet(nn.Module):
    """
    Attention-Shifted Focus Network.

    A ViT augmented with content-adaptive spatial grouping: similar adjacent
    patches are merged into single group tokens, reducing sequence length
    based on image content before the main transformer reasoning stage.

    Pipeline:
      PatchEmbed                          196 tokens, 14×14 grid
      → encoder_blocks × TransformerBlock build semantic representations
      → BoundaryRouter                    score every adjacent pair
      → ConnectedComponents               flood-fill groups from hard boundaries
      → GroupMerge                        mean-pool + project → adaptive token count
      → main_blocks × TransformerBlock    reason over compressed token set
      → masked Global Average Pool
      → Linear classifier

    ~5.58M parameters with defaults (d_model=256, enc=2, main=6, mlp_ratio=3).
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
        # Stage 1: Patch embedding
        tokens, coords = self.patch_embed(x)    # (B, 196, D), (196, 2)

        # Stage 2: Encoder — build semantic representations before routing
        for block in self.encoder:
            tokens = block(tokens, coords)       # coords shared, no mask needed

        # Stage 3: Routing — score every adjacent pair, compute ratio loss
        hard, probs, l_ratio = self.router(tokens, self.target_group_size)

        # Stage 4: Connected components — group assignments per image
        # hard is detached here: CC step is non-differentiable.
        # Gradients reach the router only through l_ratio → probs → W_q, W_k.
        all_groups = find_groups(
            hard.detach(),
            self.router.edge_indices,
            tokens.shape[1],
        )

        # Stage 5: Group merging — mean pool + project → variable-length token sequences
        padded_tokens, padded_coords, pad_mask, mean_groups = self.group_merge(
            tokens, coords, all_groups
        )
        # padded_tokens: (B, max_G, D)
        # padded_coords: (B, max_G, 2) — per-image fractional centroid coords
        # pad_mask:      (B, max_G)    — True = padding

        # Stage 6: Main network — reason over compressed tokens
        # padded_coords is (B, max_G, 2): apply_rope_2d handles the per-image case.
        # pad_mask prevents attention to padding tokens.
        for block in self.main_net:
            padded_tokens = block(padded_tokens, padded_coords, pad_mask)

        padded_tokens = self.norm(padded_tokens)

        # Stage 7: Masked global average pool — exclude padding tokens
        real_mask   = (~pad_mask).float()                                      # (B, max_G)
        token_sum   = (padded_tokens * real_mask.unsqueeze(-1)).sum(dim=1)    # (B, D)
        token_count = real_mask.sum(dim=1, keepdim=True).clamp(min=1)         # (B, 1)
        pooled      = token_sum / token_count                                  # (B, D)

        logits = self.classifier(pooled)   # (B, num_classes)
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
