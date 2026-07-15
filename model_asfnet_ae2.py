"""
model_asfnet_ae2.py — Self-supervised autoencoder on the TWO-STAGE
border-retention backbone (ASFNetBR2).

WHY: classification pressure made stage 1 chunk too aggressively /
degeneratively when a second stage sat on top of it (observed in the
classifier experiments). This model asks how the same two-stage routing
behaves when the grading signal is dense reconstruction instead of a
~6.6-bit label.

MECHANICS — the compression is hierarchical:
  Stage 1 drops interior tokens (border retention, as in ASFNetAE).
  Stage 2 groups the survivors over true grid adjacency and POOLS each
  group to a single token (GroupMerge, border-weighted).

The decoder never sees per-token content — only per-GROUP content:
each retained grid position is painted with its group's pooled token
(the chunk layout is given, matching stage 1's "retained tokens re-enter
at exact coords" philosophy), dropped positions get the learned mask
token, and the loss is computed on ALL positions. Nothing is copied
through verbatim (a kept position holds its group mean, not itself), so
grading everything is safe — and loss-on-dropped-only would re-open the
keep-all/singleton-groups degeneracy, where the graded set shrinks to
nothing. Watch mean_groups2: the known degenerate corner is "cut every
stage-2 edge", which makes every survivor a singleton group and turns
painting into identity for kept positions. The ratio losses are the only
guard against it — that is part of what this experiment observes.
"""

import torch
import torch.nn as nn

from model_asfnet import TransformerBlock
from model_asfnet_br import ASFNetBR2


class ASFNetAE2(nn.Module):
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
        target_group_size_1: float = 3.0,
        target_group_size_2: float = 3.0,
        router_proj_dim:     int   = 64,
        decoder_d_model:     int   = 128,
        decoder_blocks:      int   = 4,
        decoder_heads:       int   = 4,
        norm_pix_loss:       bool  = True,
    ):
        super().__init__()
        assert (decoder_d_model // decoder_heads) % 4 == 0, \
            "decoder head_dim must be divisible by 4 for 2D RoPE"

        self.patch_size    = patch_size
        self.in_channels   = in_channels
        self.grid_size     = image_size // patch_size
        self.n_patches     = self.grid_size ** 2
        self.norm_pix_loss = norm_pix_loss

        self.backbone = ASFNetBR2(
            image_size          = image_size,
            patch_size          = patch_size,
            in_channels         = in_channels,
            d_model             = d_model,
            num_heads           = num_heads,
            encoder1_blocks     = encoder1_blocks,
            encoder2_blocks     = encoder2_blocks,
            main_blocks         = main_blocks,
            mlp_ratio           = mlp_ratio,
            num_classes         = 0,
            target_group_size_1 = target_group_size_1,
            target_group_size_2 = target_group_size_2,
            router_proj_dim     = router_proj_dim,
            weighted_merge2     = True,   # probs2's only gradient path
        )

        # ---- Decoder (identical shape to ASFNetAE's) ----
        self.decoder_embed = nn.Linear(d_model, decoder_d_model)
        self.mask_token    = nn.Parameter(torch.zeros(1, 1, decoder_d_model))
        nn.init.normal_(self.mask_token, std=0.02)

        self.decoder = nn.ModuleList([
            TransformerBlock(decoder_d_model, decoder_heads, mlp_ratio)
            for _ in range(decoder_blocks)
        ])
        self.decoder_norm = nn.LayerNorm(decoder_d_model)
        self.decoder_pred = nn.Linear(decoder_d_model, patch_size ** 2 * in_channels)

    # ------------------------------------------------------------------
    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) → (B, N, p*p*C), row-major over the grid."""
        B, C, H, W = imgs.shape
        p = self.patch_size
        g = H // p
        x = imgs.reshape(B, C, g, p, g, p)
        x = x.permute(0, 2, 4, 3, 5, 1).reshape(B, g * g, p * p * C)
        return x

    def unpatchify(self, pred: torch.Tensor) -> torch.Tensor:
        """(B, N, p*p*C) → (B, C, H, W). For visualisation."""
        B, N, _ = pred.shape
        p, g, C = self.patch_size, self.grid_size, self.in_channels
        x = pred.reshape(B, g, g, p, p, C)
        x = x.permute(0, 5, 1, 3, 2, 4).reshape(B, C, g * p, g * p)
        return x

    # ------------------------------------------------------------------
    def _decode(
        self,
        feats2:     torch.Tensor,   # (B, max_G, D) post-norm group tokens
        group_ids2: torch.Tensor,   # (B, N) stage-2 group id per position
        keep1:      torch.Tensor,   # (B, N) bool stage-1 retention
    ) -> torch.Tensor:
        """
        Paint each RETAINED grid position with its group's pooled token
        (per-group content, per-token position — the group layout is given,
        the interior detail is not), learned mask token at dropped positions.
        Dropped positions map to the trash group id; the `where` overwrites
        them with the mask token so the junk row is never read.
        Returns per-patch predictions (B, N, p*p*C).
        """
        B, G, _ = feats2.shape
        dd = self.mask_token.shape[-1]

        enc = self.decoder_embed(feats2)                                   # (B, G, dd)
        gid = group_ids2.clamp(min=0, max=G - 1)
        x   = enc.gather(1, gid.unsqueeze(-1).expand(-1, -1, dd))          # (B, N, dd)

        mask_tok = self.mask_token.to(x.dtype)
        x = torch.where(keep1.unsqueeze(-1), x, mask_tok.expand_as(x))

        coords = self.backbone.patch_embed.coords                          # (N, 2)
        for block in self.decoder:
            x = block(x, coords)

        return self.decoder_pred(self.decoder_norm(x))                     # (B, N, p*p*C)

    # ------------------------------------------------------------------
    def forward(self, imgs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """
        Same 6-tuple contract as ASFNetAE.forward so train_asfnet_ae.py's
        run_epoch needs no branching:
            loss_rec:    scalar — MSE on ALL patches (see module docstring)
            l_ratio:     scalar — l_ratio1 + l_ratio2 (one guard-rail weight)
            l_keep:      zero tensor (no token keep-rate loss in this model)
            mean_kept:   0-dim tensor — avg stage-1 survivors per image
            mean_groups: 0-dim tensor — avg REAL stage-2 groups per image
                         (the compression rate; watch for the singleton
                         collapse where this → mean_kept)
            drop_frac:   0-dim tensor — stage-1 dropped fraction
        """
        feats2, _, _, group_ids2, keep1, l_ratio1, l_ratio2, \
            mean_kept1, mean_groups2, _s1, _probs1 = \
            self.backbone.forward_features(imgs)

        pred = self._decode(feats2, group_ids2, keep1)      # (B, N, p*p*C)

        target = self.patchify(imgs)                        # (B, N, p*p*C)
        if self.norm_pix_loss:
            mu  = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mu) / (var + 1e-6) ** 0.5

        loss_rec = ((pred.float() - target.float()) ** 2).mean()

        drop_frac = (~keep1).float().mean().detach()
        return (loss_rec, l_ratio1 + l_ratio2, imgs.new_zeros(()),
                mean_kept1, mean_groups2, drop_frac)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def reconstruct(self, imgs: torch.Tensor):
        """Same contract as ASFNetAE.reconstruct: (pred_imgs, keep mask)."""
        feats2, _, _, group_ids2, keep1, *_ = \
            self.backbone.forward_features(imgs)
        pred = self._decode(feats2, group_ids2, keep1)
        return self.unpatchify(pred.float()), keep1

    def count_parameters(self) -> dict:
        def n(m): return sum(p.numel() for p in m.parameters())
        return {
            "backbone": n(self.backbone),
            "decoder":  n(self.decoder_embed) + n(self.decoder)
                        + n(self.decoder_norm) + n(self.decoder_pred)
                        + self.mask_token.numel(),
            "total":    n(self),
        }
