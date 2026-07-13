"""
model_asfnet_ae.py — Self-supervised autoencoder on the border-retention
backbone (ASFNetBR).

WHY: classification is a ~6.6-bit distortion measure — it stops paying the
router the moment task-sufficient information is safe, so chunk placement
is never graded on retaining everything. Reconstruction is a dense
distortion measure: the chunking is graded on whether what it kept
suffices to predict what it dropped, everywhere, every image.

MECHANICS — this is MAE, with one twist:
  MAE masks a random 75% of patches and reconstructs them from the rest.
  Here the "mask" is the set of INTERIOR tokens the router chose to drop —
  a LEARNED mask covering exactly the content the model judged redundant.
  The reconstruction loss is therefore a direct dense audit of the
  router's compression judgements.

Standard MAE choices kept as-is:
  - lightweight decoder (narrower + shallower than the encoder)
  - learned mask token at dropped positions
  - per-patch normalised pixel targets (norm_pix_loss)
  - loss computed on dropped positions only
Retained tokens re-enter the decoder at their EXACT grid coordinates
(retention preserved them), so the decoder's 2D RoPE is true per token.

The backbone is a plain ASFNetBR (num_classes=0). After pretraining, load
its state dict into a classifier ASFNetBR for linear probing / fine-tuning
— keys match by construction.
"""

import torch
import torch.nn as nn

from model_asfnet import TransformerBlock
from model_asfnet_br import ASFNetBR


class ASFNetAE(nn.Module):
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
        target_group_size: float = 3.0,
        router_proj_dim:   int   = 64,
        decoder_d_model:   int   = 128,
        decoder_blocks:    int   = 4,
        decoder_heads:     int   = 4,
        norm_pix_loss:     bool  = True,
    ):
        super().__init__()
        assert (decoder_d_model // decoder_heads) % 4 == 0, \
            "decoder head_dim must be divisible by 4 for 2D RoPE"

        self.patch_size    = patch_size
        self.in_channels   = in_channels
        self.grid_size     = image_size // patch_size
        self.n_patches     = self.grid_size ** 2
        self.norm_pix_loss = norm_pix_loss

        # ---- Encoder: the full retention backbone, no classifier head ----
        self.backbone = ASFNetBR(
            image_size        = image_size,
            patch_size        = patch_size,
            in_channels       = in_channels,
            d_model           = d_model,
            num_heads         = num_heads,
            encoder_blocks    = encoder_blocks,
            main_blocks       = main_blocks,
            mlp_ratio         = mlp_ratio,
            num_classes       = 0,
            target_group_size = target_group_size,
            router_proj_dim   = router_proj_dim,
        )

        # ---- Decoder (MAE-style, lightweight) ----
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
        """
        (B, C, H, W) → (B, N, p*p*C), token order matching PatchEmbed
        (row-major over the grid).
        """
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
        feats:    torch.Tensor,   # (B, K, D)  post-norm survivor tokens
        pad_mask: torch.Tensor,   # (B, K)     True = padding slot
        sel:      torch.Tensor,   # (B, K)     original grid index per slot
    ) -> torch.Tensor:
        """
        Rebuild the full N-token grid: survivor embeddings at their true
        positions, learned mask token everywhere else. Pad slots in `sel`
        point at dropped positions — writing the mask token there (via the
        `where` below) is exactly what those positions should hold, so no
        special-casing is needed.
        Returns per-patch predictions (B, N, p*p*C).
        """
        B  = feats.shape[0]
        dd = self.mask_token.shape[-1]

        enc      = self.decoder_embed(feats)                              # (B, K, dd)
        mask_tok = self.mask_token.to(enc.dtype)
        enc      = torch.where(pad_mask.unsqueeze(-1), mask_tok.expand_as(enc), enc)

        base = mask_tok.expand(B, self.n_patches, dd)
        x    = base.scatter(1, sel.unsqueeze(-1).expand(-1, -1, dd), enc)  # (B, N, dd)

        coords = self.backbone.patch_embed.coords                          # (N, 2)
        for block in self.decoder:
            x = block(x, coords)

        return self.decoder_pred(self.decoder_norm(x))                     # (B, N, p*p*C)

    # ------------------------------------------------------------------
    def forward(
        self,
        imgs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, float, float, float]:
        """
        Returns:
            loss_rec:    scalar — MSE on DROPPED patches only
                         (per-patch-normalised targets if norm_pix_loss)
            l_ratio:     scalar — the backbone's ratio loss (guard rail;
                         see train script notes on scheduling it down)
            mean_kept:   float  — avg retained tokens per image
            mean_groups: float  — avg stage-1 group count
            drop_frac:   float  — avg fraction of patches reconstructed
        """
        feats, _, pad_mask, sel, keep, l_ratio, mean_kept, mean_groups = \
            self.backbone.forward_features(imgs)

        pred = self._decode(feats, pad_mask, sel)          # (B, N, p*p*C)

        target = self.patchify(imgs)                        # (B, N, p*p*C)
        if self.norm_pix_loss:
            mu  = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mu) / (var + 1e-6) ** 0.5

        loss_patch = ((pred.float() - target.float()) ** 2).mean(dim=-1)   # (B, N)

        m = (~keep).float()                                 # dropped positions only
        loss_rec = (loss_patch * m).sum() / m.sum().clamp(min=1.0)

        drop_frac = float(m.mean().item())
        return loss_rec, l_ratio, mean_kept, mean_groups, drop_frac

    # ------------------------------------------------------------------
    @torch.no_grad()
    def reconstruct(self, imgs: torch.Tensor):
        """
        Visualisation helper: returns (pred_imgs, keep) where pred_imgs is
        the decoder output unpatchified to image space (in normalised-pixel
        space if norm_pix_loss — visualise per-patch structure, not exact
        colours) and keep is the (B, N) retention mask.
        """
        feats, _, pad_mask, sel, keep, _, _, _ = self.backbone.forward_features(imgs)
        pred = self._decode(feats, pad_mask, sel)
        return self.unpatchify(pred.float()), keep

    def count_parameters(self) -> dict:
        def n(m): return sum(p.numel() for p in m.parameters())
        return {
            "backbone": n(self.backbone),
            "decoder":  n(self.decoder_embed) + n(self.decoder)
                        + n(self.decoder_norm) + n(self.decoder_pred)
                        + self.mask_token.numel(),
            "total":    n(self),
        }
