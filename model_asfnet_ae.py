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

from model_asfnet import TransformerBlock, load_balancing_loss
from model_asfnet_br import ASFNetBR


class SlotBottleneck(nn.Module):
    """
    Perceiver-style resampler: S learned query slots cross-attend over the
    router's survivor tokens, compressing the image to a FIXED S x D code.

    Contrast with the top-k budget: top-k is selection (attention restricted
    to one-hot — unselected survivors are discarded outright), slots are
    superposition (every survivor can contribute to every slot). The rate
    limit is architectural — S x D numbers reach the decoder no matter how
    many tokens attend in — so no loss term has to enforce compression and
    there is no soft/hard threshold seam for SGD to park under.
    """

    def __init__(self, d_model: int, n_slots: int, num_heads: int,
                 mlp_ratio: float):
        super().__init__()
        self.slots = nn.Parameter(torch.zeros(1, n_slots, d_model))
        nn.init.normal_(self.slots, std=0.02)
        self.norm_kv = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        hidden = int(d_model * mlp_ratio)
        self.norm_mlp = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden), nn.GELU(), nn.Linear(hidden, d_model),
        )

    def forward(
        self,
        feats:    torch.Tensor,   # (B, K, D) post-norm survivor tokens
        pad_mask: torch.Tensor,   # (B, K)    True = padding slot
    ) -> torch.Tensor:            # (B, S, D)
        kv = self.norm_kv(feats)
        q = self.slots.expand(feats.shape[0], -1, -1)
        x, _ = self.attn(q, kv, kv, key_padding_mask=pad_mask,
                         need_weights=False)
        x = q + x
        return x + self.mlp(self.norm_mlp(x))


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
        keep_budget:       float = 0.0,
        keep_ratio_target: float = 0.0,
        xattn_slots:       int   = 0,
    ):
        super().__init__()
        assert (decoder_d_model // decoder_heads) % 4 == 0, \
            "decoder head_dim must be divisible by 4 for 2D RoPE"
        assert not (keep_budget > 0 and xattn_slots > 0), \
            "keep_budget (top-k selection) and xattn_slots (slot " \
            "compression) are alternative bottlenecks — pick one"

        self.patch_size    = patch_size
        self.in_channels   = in_channels
        self.grid_size     = image_size // patch_size
        self.n_patches     = self.grid_size ** 2
        self.norm_pix_loss = norm_pix_loss

        # Compression enforcement (see forward): the router alone has no
        # pressure to compress — with the edge-level ratio target satisfied,
        # near-zero tokens have interiors, so ~everything is border and kept.
        #   keep_budget > 0:       HARD cap — at most round(N * keep_budget)
        #                          survivors (ranked by boundary evidence)
        #                          enter the decoder; the rest are masked and
        #                          count as dropped in the loss.
        #   keep_ratio_target > 0: SOFT pressure — H-Net load-balancing loss
        #                          on the token keep RATE (returned as l_keep;
        #                          the train script weights it).
        #   xattn_slots > 0:       SLOT bottleneck — S learned queries
        #                          cross-attend over the survivors; only the
        #                          S x D code reaches the decoder, which
        #                          reconstructs ALL patches from it (loss on
        #                          every position — nothing is copied through,
        #                          so nothing is trivial).
        self.keep_budget       = keep_budget
        self.keep_ratio_target = keep_ratio_target
        self.xattn_slots       = xattn_slots

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

        # ---- Slot-bottleneck extras (only built when active, so topk /
        # baseline checkpoints keep loading with strict state dicts) ----
        if xattn_slots > 0:
            self.slot_bottleneck = SlotBottleneck(
                d_model, xattn_slots, num_heads, mlp_ratio)
            # With no survivor tokens placed in the grid, every decoder query
            # starts as the same mask token — RoPE alone cannot break that
            # symmetry (identical values pool to identical outputs), so the
            # queries need an explicit learned position embedding before they
            # read the slots.
            self.decoder_pos = nn.Parameter(
                torch.zeros(1, self.n_patches, decoder_d_model))
            nn.init.normal_(self.decoder_pos, std=0.02)
            self.slot_read_norm_q  = nn.LayerNorm(decoder_d_model)
            self.slot_read_norm_kv = nn.LayerNorm(decoder_d_model)
            self.slot_read = nn.MultiheadAttention(
                decoder_d_model, decoder_heads, batch_first=True)

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
    def _decode_slots(
        self,
        slot_feats: torch.Tensor,   # (B, S, D) compressed slot code
    ) -> torch.Tensor:
        """
        Perceiver-IO-style decode: N position-embedded mask-token queries
        cross-attend ONCE to the slot code, then the standard self-attention
        decoder refines. Slots carry no grid coordinates (they are mixtures,
        not tokens), so they enter through cross-attention rather than being
        scattered into the grid.
        Returns per-patch predictions (B, N, p*p*C).
        """
        B = slot_feats.shape[0]

        slots = self.decoder_embed(slot_feats)                # (B, S, dd)
        x = (self.mask_token + self.decoder_pos).expand(B, -1, -1)

        kv = self.slot_read_norm_kv(slots)
        r, _ = self.slot_read(self.slot_read_norm_q(x), kv, kv,
                              need_weights=False)
        x = x + r

        coords = self.backbone.patch_embed.coords             # (N, 2)
        for block in self.decoder:
            x = block(x, coords)

        return self.decoder_pred(self.decoder_norm(x))        # (B, N, p*p*C)

    # ------------------------------------------------------------------
    def _apply_budget(
        self,
        pad_mask: torch.Tensor,   # (B, K) True = padding slot
        sel:      torch.Tensor,   # (B, K) original grid index per slot
        keep:     torch.Tensor,   # (B, N) bool retention mask
        s:        torch.Tensor,   # (B, N) boundary evidence (ranking key)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Enforce the keep budget: only the round(N * keep_budget) survivors
        with the strongest boundary evidence stay; the rest become padding
        (-> mask token in _decode) and flip to dropped in `keep`, so the
        reconstruction loss grades them. Selection is hard/detached — same
        contract as the retention mask itself; placement learns through the
        confidence residual and the ratio/keep losses, not through top-k.
        """
        K_budget = max(1, round(self.n_patches * self.keep_budget))
        if pad_mask.shape[1] <= K_budget:
            return pad_mask, keep    # every survivor already fits

        s_slot = s.detach().gather(1, sel)
        s_slot = s_slot.masked_fill(pad_mask, float("-inf"))
        top    = s_slot.topk(K_budget, dim=1).indices        # (B, K_budget)

        in_budget = torch.zeros_like(pad_mask)
        in_budget.scatter_(1, top, True)
        new_pad = pad_mask | ~in_budget      # pads stay pads even if topk'd

        # sel is a permutation prefix (unique indices), so scatter is exact:
        # slots surviving the budget mark their grid position kept.
        new_keep = torch.zeros_like(keep)
        new_keep.scatter_(1, sel, ~new_pad)
        return new_pad, new_keep

    def _keep_rate_loss(
        self,
        probs: torch.Tensor,   # (B, E) soft edge boundary probabilities
        keep:  torch.Tensor,   # (B, N) hard retention mask (unused; see below)
    ) -> torch.Tensor:
        """
        H-Net load-balancing loss on the token keep RATE (not the edge cut
        rate — that one is blind to chunk geometry and is satisfied by
        keep-everything). Soft keep probability per token is
        P(>=1 incident cut edge) under edge independence:
            p_keep = 1 - prod_e(1 - p_e)
        which matches border_keep_mask exactly in the hard limit.

        F must be the THRESHOLDED version of the same p_keep that G averages
        — not the actual retention rate. The first ablation used the real
        keep rate and the router found the loss's degenerate corner: collapse
        every edge prob to 0, so the zero-border guard keeps all 196 tokens
        (F=1) while G→0, and (1-F)(1-G) + F·G·(N-1) is exactly 0 there.
        With F = (p_keep > 0.5) the corners are unreachable/expensive
        (all-probs-0 costs N/(N-1); keep-all-soft costs ~N) and the only
        minimum (value 1.0) is at keep rate == keep_ratio_target.
        """
        ei = self.backbone.router.edge_indices
        idx_i, idx_j = ei[:, 0], ei[:, 1]

        log_q = torch.log1p(-probs.float().clamp(max=1.0 - 1e-6))   # log(1-p)
        acc = log_q.new_zeros(probs.shape[0], self.n_patches)
        acc = acc.index_add(1, idx_i, log_q).index_add(1, idx_j, log_q)
        p_keep = 1.0 - acc.exp()                                    # (B, N)

        N_tok  = 1.0 / self.keep_ratio_target
        F_rate = (p_keep > 0.5).float().mean().detach()
        G_rate = p_keep.mean()
        return load_balancing_loss(F_rate, G_rate,
                                   torch.as_tensor(N_tok, device=probs.device))

    # ------------------------------------------------------------------
    def forward(
        self,
        imgs: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """
        Returns:
            loss_rec:    scalar — MSE on DROPPED patches only
                         (per-patch-normalised targets if norm_pix_loss)
            l_ratio:     scalar — the backbone's edge-level ratio loss
            l_keep:      scalar — token-level keep-rate loss (zero tensor
                         when keep_ratio_target == 0)
            mean_kept:   0-dim tensor — avg retained tokens per image
                         (post-budget when keep_budget > 0)
            mean_groups: 0-dim tensor — avg stage-1 group count
            drop_frac:   0-dim tensor — avg fraction of patches reconstructed
                         (stats stay on GPU; .item() them only when logging,
                         otherwise every step pays a CPU/GPU sync)
        """
        feats, _, pad_mask, sel, keep, l_ratio, mean_kept, mean_groups, s, probs = \
            self.backbone.forward_features(imgs)

        # Keep-rate loss grades the router's own keep rate (pre-budget).
        l_keep = self._keep_rate_loss(probs, keep) if self.keep_ratio_target > 0 \
            else imgs.new_zeros(())

        if self.keep_budget > 0:
            pad_mask, keep = self._apply_budget(pad_mask, sel, keep, s)
            mean_kept = keep.float().sum(dim=1).mean().detach()

        if self.xattn_slots > 0:
            pred = self._decode_slots(self.slot_bottleneck(feats, pad_mask))
        else:
            pred = self._decode(feats, pad_mask, sel)      # (B, N, p*p*C)

        target = self.patchify(imgs)                        # (B, N, p*p*C)
        if self.norm_pix_loss:
            mu  = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mu) / (var + 1e-6) ** 0.5

        loss_patch = ((pred.float() - target.float()) ** 2).mean(dim=-1)   # (B, N)

        if self.xattn_slots > 0:
            # Slot bottleneck: no position is copied through the decoder, so
            # EVERY position is graded. (Loss-on-dropped-only here would
            # re-open the keep-all degeneracy: the router could shrink the
            # graded set to nothing. The rate limit is architectural, so
            # grading everything costs the model nothing it can dodge.)
            loss_rec = loss_patch.mean()
        else:
            m = (~keep).float()                             # dropped positions only
            loss_rec = (loss_patch * m).sum() / m.sum().clamp(min=1.0)

        # Router's drop fraction in every mode (in xattn mode it is a
        # monitoring stat — what the slots could not see — not the loss mask).
        drop_frac = (~keep).float().mean().detach()
        return loss_rec, l_ratio, l_keep, mean_kept, mean_groups, drop_frac

    # ------------------------------------------------------------------
    @torch.no_grad()
    def reconstruct(self, imgs: torch.Tensor):
        """
        Visualisation helper: returns (pred_imgs, keep) where pred_imgs is
        the decoder output unpatchified to image space (in normalised-pixel
        space if norm_pix_loss — visualise per-patch structure, not exact
        colours) and keep is the (B, N) retention mask.
        """
        feats, _, pad_mask, sel, keep, _, _, _, s, _ = \
            self.backbone.forward_features(imgs)
        if self.keep_budget > 0:
            pad_mask, keep = self._apply_budget(pad_mask, sel, keep, s)
        if self.xattn_slots > 0:
            pred = self._decode_slots(self.slot_bottleneck(feats, pad_mask))
        else:
            pred = self._decode(feats, pad_mask, sel)
        return self.unpatchify(pred.float()), keep

    def count_parameters(self) -> dict:
        def n(m): return sum(p.numel() for p in m.parameters())
        counts = {
            "backbone": n(self.backbone),
            "decoder":  n(self.decoder_embed) + n(self.decoder)
                        + n(self.decoder_norm) + n(self.decoder_pred)
                        + self.mask_token.numel(),
        }
        if self.xattn_slots > 0:
            counts["slot_bneck"] = (n(self.slot_bottleneck) + n(self.slot_read)
                                    + n(self.slot_read_norm_q)
                                    + n(self.slot_read_norm_kv)
                                    + self.decoder_pos.numel())
        counts["total"] = n(self)
        return counts
