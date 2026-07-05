import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Patch Embedding
# ---------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    """
    Splits a 224×224 image into non-overlapping patches and linearly
    projects each to d_model dimensions.

    Unlike ASFNet's PatchEmbed, this does not need to return grid
    coordinates — position is handled by a learnable 1D positional
    embedding added to tokens after the CLS token is prepended.
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)                    # (B, D, grid, grid)
        x = x.flatten(2).transpose(1, 2)   # (B, N, D)
        return self.norm(x)                 # (B, N, D)


# ---------------------------------------------------------------------------
# Multi-Head Self-Attention
# ---------------------------------------------------------------------------

class Attention(nn.Module):
    """
    Standard multi-head self-attention (Multi-Head Self-Attention — each token
    attends to every other token, with multiple parallel attention heads each
    working in a lower-dimensional subspace).

    No masking, no RoPE — position comes entirely from the learnable positional
    embedding on the token values. Uses F.scaled_dot_product_attention which
    dispatches to FlashAttention (a memory-efficient fused attention kernel)
    when available.
    """

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim  = d_model // num_heads

        self.qkv  = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape

        qkv = self.qkv(x)                                                     # (B, N, 3D)
        q, k, v = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4).unbind(0)
        # q, k, v: each (B, num_heads, N, head_dim)

        out = F.scaled_dot_product_attention(q, k, v)                         # (B, H, N, head_dim)
        out = out.transpose(1, 2).reshape(B, N, D)                            # (B, N, D)
        return self.proj(out)


# ---------------------------------------------------------------------------
# Transformer Block (pre-norm)
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """
    Standard Vision Transformer block with pre-norm (LayerNorm — a technique
    that normalizes the activations within each token to zero mean and unit
    variance — applied before each sub-layer rather than after).

    Structure: LN → Attention → residual → LN → FFN → residual

    No position-awareness at the block level — position is baked into
    the token values via the learnable positional embedding once before
    the first block, not recomputed per block.
    """

    def __init__(self, d_model: int, num_heads: int, mlp_ratio: float = 4.0):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# ViT
# ---------------------------------------------------------------------------

class ViT(nn.Module):
    """
    Standard Vision Transformer (ViT) — plain baseline for ablation.

    Deliberately minimal: no routing, no grouping, no RoPE, no saccades.
    The point is a clean, well-understood reference point at the same
    parameter budget (~6M) as ASFNet and SaccadeNet.

    Architecture:
      PatchEmbed                     → N patch tokens, each d_model-dim
      Prepend CLS token              → N+1 tokens
      + learnable positional embed   → N+1 tokens (position injected once)
      → depth × TransformerBlock     → contextualised token sequence
      → LayerNorm on CLS token
      → Linear classifier

    The CLS token (Classification token — a learnable vector prepended to
    the patch sequence that aggregates global image information through
    attention and serves as the classification representation) is the only
    output used for classification.

    Positional embedding: learnable 1D (one float vector per position in
    the N+1 sequence). Simpler and equally competitive with sinusoidal or
    2D variants at this scale.
    """

    def __init__(
        self,
        image_size:  int   = 224,
        patch_size:  int   = 16,
        in_channels: int   = 3,
        d_model:     int   = 256,
        num_heads:   int   = 8,
        depth:       int   = 7,
        mlp_ratio:   float = 4.0,
        num_classes: int   = 100,
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(image_size, patch_size, in_channels, d_model)
        n_patches = (image_size // patch_size) ** 2

        # CLS token: one learnable vector per batch, expanded at forward time
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        # Learnable positional embedding: one vector per sequence position
        # (n_patches patch positions + 1 CLS position)
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, d_model))

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, mlp_ratio)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

        self._init_weights()

    def _init_weights(self):
        # Standard ViT initialisation from "An Image is Worth 16x16 Words"
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)
        Returns: logits (B, num_classes)
        """
        B = x.shape[0]

        tokens = self.patch_embed(x)                                   # (B, N, D)
        cls    = self.cls_token.expand(B, -1, -1)                     # (B, 1, D)
        tokens = torch.cat([cls, tokens], dim=1)                       # (B, N+1, D)
        tokens = tokens + self.pos_embed                               # (B, N+1, D)

        for block in self.blocks:
            tokens = block(tokens)

        cls_out = self.norm(tokens[:, 0])                              # (B, D) — CLS only
        return self.head(cls_out)                                      # (B, num_classes)

    def count_parameters(self) -> dict:
        def n(m): return sum(p.numel() for p in m.parameters())
        return {
            "patch_embed":   n(self.patch_embed),
            "cls+pos_embed": self.cls_token.numel() + self.pos_embed.numel(),
            "blocks":        n(self.blocks),
            "norm+head":     n(self.norm) + n(self.head),
            "total":         n(self),
        }
