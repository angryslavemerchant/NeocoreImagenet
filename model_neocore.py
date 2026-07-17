"""
model_neocore.py — Neocore: a recursive-admission autoencoder.

WHY (the ladder era's verdict, 2026-07-16): chunking is a postcondition of
recognition — a one-shot knowledge-free router cannot parse an image, no
matter how mechanically healthy (the ghost-grid run removed every excuse
and the cut fields still never coarsened). Every reconstruction gain of
that era came from evidence-ranked admission under an architecturally
fixed rate; every failure came from letting the model modulate how many
tokens survive. Neocore keeps only what worked and makes the knowledge
explicit: ADMISSION IS RECURRENT.

MECHANICS: one weight-shared encoder ("core") is applied R rounds over the
full patch grid. Each round it scores the not-yet-admitted tokens and
admits exactly K/R more into working memory; admitted tokens are stamped
once with a learned in-memory marker, so every later round scores the
image KNOWING what memory already holds. After round R the decoder
reconstructs the image from the K admitted tokens alone (MAE-style: mask
token elsewhere, loss on dropped positions, norm_pix targets).

The law, obeyed: nothing anywhere modulates how many tokens survive —
K and R are architectural constants; the only learnable degrees of
freedom are WHICH tokens enter and in WHAT ORDER. R=1 collapses to the
one-shot budget model (AE_budget25, val rec 0.101), the built-in control.

The pre-registered failure mode is NULLITY, not collapse: the score head
could ignore the markers and reproduce the one-shot ranking in R slices.
forward() therefore returns two alarms computed every step:
  overlap_r1: |top-K by round-1 scores ∩ final memory| / K — pinned at
              1.0 means the loop is a sorted one-shot;
  admit_corr: Pearson(round-1 score, admission round) over admitted
              tokens — ~-1.0 means later rounds just walk down the
              round-1 ranking.

Gradient paths (top-k itself is hard/detached, as always):
  - the confidence residual at admission, token += sigmoid(score)·token —
    the same mechanism that trained every ASFNet router, minus the edges;
  - features carry across rounds (round r+1 encodes round r's output),
    so early rounds shape everything downstream;
  - the marker parameter rides the residual stream into the decoder.

Compute/memory: R full-grid passes = R× the one-shot encoder. At B=1024,
N=196, D=256, bf16, ~8 blocks store ≈9 GB activations per round — 7
rounds would need ~60 GB, so per-round gradient checkpointing is ON by
default (~9 GB total for ~30% recompute).
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from model_asfnet import PatchEmbed, TransformerBlock


class NeocoreAE(nn.Module):
    def __init__(
        self,
        image_size:       int   = 224,
        patch_size:       int   = 16,
        in_channels:      int   = 3,
        d_model:          int   = 256,
        num_heads:        int   = 8,
        core_blocks:      int   = 8,
        mlp_ratio:        float = 3.0,
        rounds:           int   = 7,
        memory_tokens:    int   = 49,
        decoder_d_model:  int   = 128,
        decoder_blocks:   int   = 4,
        decoder_heads:    int   = 4,
        norm_pix_loss:    bool  = True,
        round_checkpoint: bool  = True,
    ):
        super().__init__()
        assert (d_model // num_heads) % 4 == 0, \
            "head_dim must be divisible by 4 for 2D RoPE"
        assert (decoder_d_model // decoder_heads) % 4 == 0, \
            "decoder head_dim must be divisible by 4 for 2D RoPE"

        self.patch_size    = patch_size
        self.in_channels   = in_channels
        self.grid_size     = image_size // patch_size
        self.n_patches     = self.grid_size ** 2
        self.rounds        = rounds
        self.memory_tokens = memory_tokens
        self.norm_pix_loss = norm_pix_loss
        self.round_checkpoint = round_checkpoint

        assert 1 <= rounds <= memory_tokens <= self.n_patches

        # Exact-K per round — the only regime that has ever held. Any
        # remainder of K/R goes to the earliest rounds.
        base, rem = divmod(memory_tokens, rounds)
        self.admit_schedule = [base + (1 if r < rem else 0)
                               for r in range(rounds)]

        # ---- Core: weight-shared encoder applied every round ----
        self.patch_embed = PatchEmbed(image_size, patch_size,
                                      in_channels, d_model)
        self.core = nn.ModuleList([
            TransformerBlock(d_model, num_heads, mlp_ratio)
            for _ in range(core_blocks)
        ])
        self.norm = nn.LayerNorm(d_model)   # scores, decoder input, probes

        # ---- Admission machinery ----
        self.score_head = nn.Linear(d_model, 1)
        # In-memory marker: stamped ONCE, at admission — strictly binary
        # information ("you are in memory"), no round index. It persists
        # through later rounds via the residual stream.
        self.marker = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.marker, std=0.02)

        # ---- Decoder (MAE-style, lightweight) ----
        self.decoder_embed = nn.Linear(d_model, decoder_d_model)
        self.mask_token    = nn.Parameter(torch.zeros(1, 1, decoder_d_model))
        nn.init.normal_(self.mask_token, std=0.02)
        self.decoder = nn.ModuleList([
            TransformerBlock(decoder_d_model, decoder_heads, mlp_ratio)
            for _ in range(decoder_blocks)
        ])
        self.decoder_norm = nn.LayerNorm(decoder_d_model)
        self.decoder_pred = nn.Linear(decoder_d_model,
                                      patch_size ** 2 * in_channels)

    # ------------------------------------------------------------------
    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) → (B, N, p*p*C), row-major to match PatchEmbed."""
        B, C, H, W = imgs.shape
        p = self.patch_size
        g = H // p
        x = imgs.reshape(B, C, g, p, g, p)
        return x.permute(0, 2, 4, 3, 5, 1).reshape(B, g * g, p * p * C)

    def unpatchify(self, pred: torch.Tensor) -> torch.Tensor:
        """(B, N, p*p*C) → (B, C, H, W). For visualisation."""
        B, N, _ = pred.shape
        p, g, C = self.patch_size, self.grid_size, self.in_channels
        x = pred.reshape(B, g, g, p, p, C)
        return x.permute(0, 5, 1, 3, 2, 4).reshape(B, C, g * p, g * p)

    # ------------------------------------------------------------------
    def _core_pass(self, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        for block in self.core:
            x = block(x, coords)
        return x

    def forward_rounds(
        self,
        imgs: torch.Tensor,
        collect_trace: bool = False,
    ):
        """
        The loop. Returns:
            tok:          (B, N, D) final features (pre-norm)
            admitted:     (B, N) bool — exactly K True per image
            admit_round:  (B, N) long — round each token entered (-1 = never)
            first_scores: (B, N) detached round-1 scores (nullity alarms)
            trace:        list of (tok_after_round, admitted_after_round)
                          detached clones, one per round — only when
                          collect_trace (eval instrument), else None
        """
        tok, coords = self.patch_embed(imgs)          # (B, N, D)
        B, N, _ = tok.shape

        admitted    = torch.zeros(B, N, dtype=torch.bool, device=tok.device)
        admit_round = torch.full((B, N), -1, dtype=torch.long,
                                 device=tok.device)
        first_scores = None
        trace = [] if collect_trace else None

        for r in range(self.rounds):
            if self.round_checkpoint and self.training \
                    and torch.is_grad_enabled():
                tok = checkpoint(self._core_pass, tok, coords,
                                 use_reentrant=False)
            else:
                tok = self._core_pass(tok, coords)

            scores = self.score_head(self.norm(tok)).squeeze(-1)   # (B, N)
            if r == 0:
                first_scores = scores.detach()

            # Hard exact-K admission among the not-yet-admitted. Detached —
            # placement learns through the confidence residual, not top-k.
            cand = scores.detach().masked_fill(admitted, float("-inf"))
            top  = cand.topk(self.admit_schedule[r], dim=1).indices
            new  = torch.zeros_like(admitted)
            new.scatter_(1, top, True)

            # Admission stamp: amplify by confidence (the score head's
            # gradient path) and add the in-memory marker — once per token.
            gate = torch.sigmoid(scores).unsqueeze(-1)
            tok  = tok + new.unsqueeze(-1) * (gate * tok + self.marker)

            admitted    = admitted | new
            admit_round = torch.where(new, torch.full_like(admit_round, r),
                                      admit_round)
            if collect_trace:
                trace.append((tok.detach().clone(), admitted.clone()))

        return tok, admitted, admit_round, first_scores, trace

    # ------------------------------------------------------------------
    def _decode(
        self,
        tok:      torch.Tensor,   # (B, N, D) final features (pre-norm)
        admitted: torch.Tensor,   # (B, N) bool
    ) -> torch.Tensor:
        """
        Everything stayed on the grid, so no compaction/scatter machinery:
        admitted positions carry their features, the rest the mask token.
        Exactly K×D numbers reach the decoder — the rate is architectural.
        """
        enc      = self.decoder_embed(self.norm(tok))              # (B, N, dd)
        mask_tok = self.mask_token.to(enc.dtype)
        x = torch.where(admitted.unsqueeze(-1), enc, mask_tok.expand_as(enc))

        coords = self.patch_embed.coords                           # (N, 2)
        for block in self.decoder:
            x = block(x, coords)
        return self.decoder_pred(self.decoder_norm(x))             # (B, N, p*p*C)

    # ------------------------------------------------------------------
    @staticmethod
    def _nullity_alarms(first_scores, admitted, admit_round, K):
        """
        overlap_r1: fraction of the final memory that plain one-shot top-K
                    on round-1 scores would have picked anyway.
        admit_corr: Pearson(round-1 score, admission round) over admitted
                    tokens (masked-weight form — static shapes for compile).
        Both detached; ~1.0 / ~-1.0 together = the loop is a sorted one-shot.
        """
        one_shot = torch.zeros_like(admitted)
        one_shot.scatter_(1, first_scores.topk(K, dim=1).indices, True)
        overlap = (one_shot & admitted).float().sum(dim=1).mean() / K

        w = admitted.float()
        n = w.sum().clamp(min=1.0)
        x = first_scores.float()
        y = admit_round.float()
        mx = (x * w).sum() / n
        my = (y * w).sum() / n
        cov = ((x - mx) * (y - my) * w).sum()
        vx  = (((x - mx) ** 2) * w).sum()
        vy  = (((y - my) ** 2) * w).sum()
        corr = cov / (vx.sqrt() * vy.sqrt()).clamp(min=1e-6)
        return overlap.detach(), corr.detach()

    # ------------------------------------------------------------------
    def forward(self, imgs: torch.Tensor):
        """
        Returns:
            loss_rec:   scalar — MSE on the N-K dropped patches
                        (per-patch-normalised targets if norm_pix_loss)
            overlap_r1: 0-dim detached tensor — see _nullity_alarms
            admit_corr: 0-dim detached tensor — see _nullity_alarms
        No auxiliary losses: rates are architectural constants.
        """
        tok, admitted, admit_round, first_scores, _ = self.forward_rounds(imgs)
        pred = self._decode(tok, admitted)                  # (B, N, p*p*C)

        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mu  = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mu) / (var + 1e-6) ** 0.5

        loss_patch = ((pred.float() - target.float()) ** 2).mean(dim=-1)
        m = (~admitted).float()
        loss_rec = (loss_patch * m).sum() / m.sum().clamp(min=1.0)

        overlap, corr = self._nullity_alarms(
            first_scores, admitted, admit_round, self.memory_tokens)
        return loss_rec, overlap, corr

    # ------------------------------------------------------------------
    def forward_features(self, imgs: torch.Tensor):
        """
        Probe interface: (feats, admitted, admit_round) with feats normed.
        The attentive probe decides what to pool over (memory only vs all).
        """
        tok, admitted, admit_round, _, _ = self.forward_rounds(imgs)
        return self.norm(tok), admitted, admit_round

    @torch.no_grad()
    def reconstruct(self, imgs: torch.Tensor):
        """
        Viz helper: (pred_imgs, admitted, admit_round). pred_imgs is in
        normalised-pixel space when norm_pix_loss (structure, not colour).
        """
        tok, admitted, admit_round, _, _ = self.forward_rounds(imgs)
        pred = self._decode(tok, admitted)
        return self.unpatchify(pred.float()), admitted, admit_round

    @torch.no_grad()
    def round_trace(self, imgs: torch.Tensor):
        """
        Eval instrument for the anti-correlation prediction: per-round
        partial decodes. Returns (preds, admits, admit_round) where
        preds[r] (B, N, p*p*C) is decoded from memory as of round r and
        admits[r] is that round's (B, N) admitted mask.
        NOTE the decoder only ever trains on K-token memories, so early-
        round decodes are off-distribution — use the error maps' spatial
        RANKING, not their absolute values.
        """
        tok, admitted, admit_round, _, trace = self.forward_rounds(
            imgs, collect_trace=True)
        preds  = [self._decode(t, a) for t, a in trace]
        admits = [a for _, a in trace]
        return preds, admits, admit_round

    def count_parameters(self) -> dict:
        def n(m): return sum(p.numel() for p in m.parameters())
        counts = {
            "patch_embed": n(self.patch_embed),
            "core":        n(self.core) + n(self.norm),
            "admission":   n(self.score_head) + self.marker.numel(),
            "decoder":     n(self.decoder_embed) + n(self.decoder)
                           + n(self.decoder_norm) + n(self.decoder_pred)
                           + self.mask_token.numel(),
        }
        counts["total"] = n(self)
        return counts
