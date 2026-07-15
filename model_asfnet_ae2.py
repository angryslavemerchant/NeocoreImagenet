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
from model_asfnet_ae import SlotBottleneck
from model_asfnet_br import ASFNetBR2, ASFNetBR2R


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


class ASFNetAE2R(nn.Module):
    """
    Autoencoder on the DOUBLE-retention two-stage backbone (ASFNetBR2R).

    Decode is exactly the single-stage recipe: final survivors re-enter the
    decoder at their exact grid coordinates (retention preserved them at
    both stages), learned mask token everywhere else, loss on DROPPED
    positions only — the mask is what the two routers jointly chose to drop.

    Same keep-everything degenerate corner as the single-stage baseline;
    whether coarser stage-2 chunks develop real interiors (real drops) is
    the observation this run exists to make. Watch drop_frac and
    mean_kept2 vs mean_kept1.
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
        target_group_size_1: float = 3.0,
        target_group_size_2: float = 3.0,
        router_proj_dim:     int   = 64,
        decoder_d_model:     int   = 128,
        decoder_blocks:      int   = 4,
        decoder_heads:       int   = 4,
        norm_pix_loss:       bool  = True,
        xattn_slots:         int   = 0,
    ):
        super().__init__()
        assert (decoder_d_model // decoder_heads) % 4 == 0, \
            "decoder head_dim must be divisible by 4 for 2D RoPE"

        self.patch_size    = patch_size
        self.in_channels   = in_channels
        self.grid_size     = image_size // patch_size
        self.n_patches     = self.grid_size ** 2
        self.norm_pix_loss = norm_pix_loss
        # xattn_slots > 0: same slot bottleneck as the single-stage
        # ASFNetAE, but over the FINAL (post-stage-2) survivors — the
        # architectural rate limit the pure-retention path lacks. Loss
        # switches to ALL positions (nothing is copied through).
        self.xattn_slots   = xattn_slots

        self.backbone = ASFNetBR2R(
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
        )

        self.decoder_embed = nn.Linear(d_model, decoder_d_model)
        self.mask_token    = nn.Parameter(torch.zeros(1, 1, decoder_d_model))
        nn.init.normal_(self.mask_token, std=0.02)

        self.decoder = nn.ModuleList([
            TransformerBlock(decoder_d_model, decoder_heads, mlp_ratio)
            for _ in range(decoder_blocks)
        ])
        self.decoder_norm = nn.LayerNorm(decoder_d_model)
        self.decoder_pred = nn.Linear(decoder_d_model, patch_size ** 2 * in_channels)

        # ---- Slot-bottleneck extras (mirrors ASFNetAE; built only when
        # active so plain-retain checkpoints keep strict state dicts) ----
        if xattn_slots > 0:
            self.slot_bottleneck = SlotBottleneck(
                d_model, xattn_slots, num_heads, mlp_ratio)
            self.decoder_pos = nn.Parameter(
                torch.zeros(1, self.n_patches, decoder_d_model))
            nn.init.normal_(self.decoder_pos, std=0.02)
            self.slot_read_norm_q  = nn.LayerNorm(decoder_d_model)
            self.slot_read_norm_kv = nn.LayerNorm(decoder_d_model)
            self.slot_read = nn.MultiheadAttention(
                decoder_d_model, decoder_heads, batch_first=True)

    patchify   = ASFNetAE2.patchify
    unpatchify = ASFNetAE2.unpatchify

    # ------------------------------------------------------------------
    def _decode(
        self,
        feats:    torch.Tensor,   # (B, K, D)  post-norm survivor tokens
        pad_mask: torch.Tensor,   # (B, K)     True = padding slot
        sel:      torch.Tensor,   # (B, K)     original grid index per slot
    ) -> torch.Tensor:
        """Identical to ASFNetAE._decode: survivors scattered at their true
        grid positions, mask token everywhere else."""
        B  = feats.shape[0]
        dd = self.mask_token.shape[-1]

        enc      = self.decoder_embed(feats)
        mask_tok = self.mask_token.to(enc.dtype)
        enc      = torch.where(pad_mask.unsqueeze(-1), mask_tok.expand_as(enc), enc)

        base = mask_tok.expand(B, self.n_patches, dd)
        x    = base.scatter(1, sel.unsqueeze(-1).expand(-1, -1, dd), enc)

        coords = self.backbone.patch_embed.coords
        for block in self.decoder:
            x = block(x, coords)

        return self.decoder_pred(self.decoder_norm(x))

    # ------------------------------------------------------------------
    def _decode_slots(self, slot_feats: torch.Tensor) -> torch.Tensor:
        """Perceiver-IO decode, identical to ASFNetAE._decode_slots."""
        B = slot_feats.shape[0]

        slots = self.decoder_embed(slot_feats)
        x = (self.mask_token + self.decoder_pos).expand(B, -1, -1)

        kv = self.slot_read_norm_kv(slots)
        r, _ = self.slot_read(self.slot_read_norm_q(x), kv, kv,
                              need_weights=False)
        x = x + r

        coords = self.backbone.patch_embed.coords
        for block in self.decoder:
            x = block(x, coords)

        return self.decoder_pred(self.decoder_norm(x))

    # ------------------------------------------------------------------
    def forward(self, imgs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """
        Same 6-tuple contract as ASFNetAE / ASFNetAE2. Slot semantics here:
            mean_kept:   FINAL survivors (post stage-2)
            mean_groups: stage-1 survivors (so both retention levels log)
            drop_frac:   fraction dropped across both stages
                         (the loss support only when xattn_slots == 0)
        """
        feats, _, pad_mask, sel, keep2, l_ratio1, l_ratio2, \
            mean_kept1, mean_kept2, _s1, _s2, _probs1 = \
            self.backbone.forward_features(imgs)

        if self.xattn_slots > 0:
            pred = self._decode_slots(self.slot_bottleneck(feats, pad_mask))
        else:
            pred = self._decode(feats, pad_mask, sel)       # (B, N, p*p*C)

        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mu  = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mu) / (var + 1e-6) ** 0.5

        loss_patch = ((pred.float() - target.float()) ** 2).mean(dim=-1)

        if self.xattn_slots > 0:
            # Slot bottleneck: nothing is copied through — grade everything
            # (same reasoning as the single-stage xattn variant).
            loss_rec = loss_patch.mean()
        else:
            m = (~keep2).float()                            # dropped positions only
            loss_rec = (loss_patch * m).sum() / m.sum().clamp(min=1.0)

        drop_frac = (~keep2).float().mean().detach()
        return (loss_rec, l_ratio1 + l_ratio2, imgs.new_zeros(()),
                mean_kept2, mean_kept1, drop_frac)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def reconstruct(self, imgs: torch.Tensor):
        """Same contract as ASFNetAE.reconstruct: (pred_imgs, keep mask)."""
        feats, _, pad_mask, sel, keep2, *_ = self.backbone.forward_features(imgs)
        if self.xattn_slots > 0:
            pred = self._decode_slots(self.slot_bottleneck(feats, pad_mask))
        else:
            pred = self._decode(feats, pad_mask, sel)
        return self.unpatchify(pred.float()), keep2

    def count_parameters(self) -> dict:
        def n(m): return sum(p.numel() for p in m.parameters())
        return {
            "backbone": n(self.backbone),
            "decoder":  n(self.decoder_embed) + n(self.decoder)
                        + n(self.decoder_norm) + n(self.decoder_pred)
                        + self.mask_token.numel(),
            "total":    n(self),
        }
