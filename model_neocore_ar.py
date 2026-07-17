"""
model_neocore_ar.py — Neocore-AR: bidirectional encode once, then generate
the memory autoregressively, one token at a time.

WHY (2026-07-17, user-directed): the recursive loop model re-encodes the
full 196-token grid every round — R× the one-shot compute — and the R-sweep
showed the returns grow with R. This sibling keeps the recursion where it is
cheap and drops it where it is expensive:

  PHASE A  the core encodes the full image ONCE, bidirectionally. Its
           per-layer K/V are cached — the image is never re-encoded.
  PHASE B  K memory tokens are generated strictly one at a time. Step t:
           a pointer head scores every not-yet-admitted patch against the
           current controller state (the previous memory token's output;
           a learned start query at t=0), the argmax patch is admitted,
           its encoded features (+ a learned marker) seed a NEW sequence
           position that attends over the cached image and all previous
           memory tokens (causal). Its K/V join the cache.
  PHASE C  after all K tokens exist, the MAE decoder reconstructs the
           grid from them (mask token elsewhere, dropped-only loss).

The law, still obeyed: K is an architectural constant; the model chooses
WHICH tokens enter and in WHAT ORDER, never how many. Admission is
argmax-detached; the score head learns through a confidence gate applied
to each memory token's contribution to the decoder.

Versus the loop model: the grid never re-contextualizes (image features
are frozen after phase A); all accumulated knowledge lives in the memory
tokens themselves. Admission granularity is 1 token (49 "rounds" of 1)
instead of 7×7. Compute is ~2.5× the R=1 one-shot instead of R=7's 7×.

TRAINING = free-run + teacher-forced replay. Selection is hard/detached
(as always), so gradients never flow through the CHOICE — which means the
admission ORDER can be determined in a cheap no-grad sequential pass with
a real KV cache (pass 1), and the whole computation then REPLAYED in one
parallel masked pass with gradients (pass 2): sequence [196 image tokens |
K memory seeds], mask = image↔image bidirectional, memory→image full,
memory→memory causal, image→memory blocked. Pass 2 is mathematically
identical to pass 1 (the smoke test asserts it) but backpropagates like a
standard transformer. At eval, pass 1 IS the deployment path: pure
incremental generation against a frozen cache.

Interface-compatible with the loop model where it matters: forward()
returns (loss, overlap_r1, admit_corr, stability=1.0); forward_features /
reconstruct / round_trace / patchify / unpatchify match. For the round
instruments, the K admission steps are binned into `rounds` (=7 for K=49)
pseudo-rounds so admission maps and error-percentile stats stay directly
comparable with loop-era panels; admit_corr uses the raw step index.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_asfnet import PatchEmbed, TransformerBlock, apply_rope_2d
from model_neocore import NeocoreAE


class ARAttention(nn.Module):
    """Self-attention with 2D RoPE supporting a full masked pass (training
    replay) and an incremental cached step (generation). Same parameter
    shapes as model_asfnet.Attention."""

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = d_model // num_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def _heads(self, t: torch.Tensor, B: int, N: int) -> torch.Tensor:
        return t.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

    def forward_full(self, x, coords, attn_mask=None, return_kv=False):
        """x (B,S,D); coords (S,2) or (B,S,2); attn_mask bool (1,1,S,S),
        True = may attend. Keys are stored ROTATED, so cached K from this
        pass are directly reusable by forward_step."""
        B, S, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = self._heads(q, B, S), self._heads(k, B, S), self._heads(v, B, S)
        q, k = apply_rope_2d(q, k, coords, self.head_dim)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        o = self.out(o.transpose(1, 2).reshape(B, S, D))
        if return_kv:
            return o, k, v
        return o

    def forward_step(self, x_new, coords_new, k_cache, v_cache, cache_len):
        """One new token against the cache. x_new (B,1,D); coords_new
        (B,1,2); caches (B,H,S_max,hd) preallocated, filled to cache_len.
        Writes the new K/V at cache_len and attends over [0, cache_len]."""
        B = x_new.shape[0]
        q, k, v = self.qkv(x_new).chunk(3, dim=-1)
        q, k, v = self._heads(q, B, 1), self._heads(k, B, 1), self._heads(v, B, 1)
        q, k = apply_rope_2d(q, k, coords_new, self.head_dim)
        k_cache[:, :, cache_len:cache_len + 1] = k
        v_cache[:, :, cache_len:cache_len + 1] = v
        o = F.scaled_dot_product_attention(
            q, k_cache[:, :, :cache_len + 1], v_cache[:, :, :cache_len + 1])
        return self.out(o.transpose(1, 2).reshape(B, 1, -1))


class ARBlock(nn.Module):
    """Pre-norm transformer block wrapping ARAttention. Same parameter
    shapes as model_asfnet.TransformerBlock."""

    def __init__(self, d_model: int, num_heads: int, mlp_ratio: float = 3.0):
        super().__init__()
        ffn_dim = int(d_model * mlp_ratio)
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = ARAttention(d_model, num_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, d_model),
        )

    def forward_full(self, x, coords, attn_mask=None, return_kv=False):
        if return_kv:
            a, k, v = self.attn.forward_full(self.norm1(x), coords,
                                             attn_mask, return_kv=True)
            x = x + a
            x = x + self.ffn(self.norm2(x))
            return x, k, v
        x = x + self.attn.forward_full(self.norm1(x), coords, attn_mask)
        x = x + self.ffn(self.norm2(x))
        return x

    def forward_step(self, x_new, coords_new, k_cache, v_cache, cache_len):
        x_new = x_new + self.attn.forward_step(self.norm1(x_new), coords_new,
                                               k_cache, v_cache, cache_len)
        x_new = x_new + self.ffn(self.norm2(x_new))
        return x_new


class NeocoreARAE(nn.Module):
    def __init__(
        self,
        image_size:      int   = 224,
        patch_size:      int   = 16,
        in_channels:     int   = 3,
        d_model:         int   = 256,
        num_heads:       int   = 8,
        core_blocks:     int   = 8,
        mlp_ratio:       float = 3.0,
        memory_tokens:   int   = 49,
        decoder_d_model: int   = 128,
        decoder_blocks:  int   = 4,
        decoder_heads:   int   = 4,
        norm_pix_loss:   bool  = True,
        trace_bins:      int   = 7,
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
        self.memory_tokens = memory_tokens
        self.norm_pix_loss = norm_pix_loss
        assert 1 <= memory_tokens <= self.n_patches

        # Pseudo-round binning of the K admission steps, purely for the
        # eval instruments (admission maps / round decodes / percentile
        # stats stay comparable with loop-era panels).
        self.trace_bins = min(trace_bins, memory_tokens)
        self.bin_size   = math.ceil(memory_tokens / self.trace_bins)
        self.rounds     = math.ceil(memory_tokens / self.bin_size)

        # ---- Core: one stack for both phases (weight-shared) ----
        self.patch_embed = PatchEmbed(image_size, patch_size,
                                      in_channels, d_model)
        self.core = nn.ModuleList([
            ARBlock(d_model, num_heads, mlp_ratio)
            for _ in range(core_blocks)
        ])
        self.norm = nn.LayerNorm(d_model)   # scores, controller, decoder in

        # ---- Admission machinery (pointer-style) ----
        # score(step t, patch p) = q_proj(controller_t) · normed_feats_p /√D
        # where controller_t is the previous memory token's normed output
        # (a learned start query at t=0). The controller CHANGES as memory
        # grows — that is the "look again knowing what you have", now
        # carried by the memory tokens instead of grid re-encoding.
        self.score_q = nn.Linear(d_model, d_model, bias=False)
        self.start_query = nn.Parameter(torch.zeros(d_model))
        nn.init.normal_(self.start_query, std=0.02)
        # Seed marker: added to the admitted patch's features when they
        # become a memory-token input ("you are becoming memory").
        self.marker = nn.Parameter(torch.zeros(d_model))
        nn.init.normal_(self.marker, std=0.02)

        # ---- Decoder (identical to the loop model) ----
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

        # Replay mask: True = may attend. Image rows are bidirectional
        # among themselves and blind to memory (phase A must be exactly
        # reproduced); memory row i sees the whole image + memory ≤ i.
        N, K, S = self.n_patches, memory_tokens, self.n_patches + memory_tokens
        allowed = torch.zeros(S, S, dtype=torch.bool)
        allowed[:N, :N] = True
        allowed[N:, :N] = True
        allowed[N:, N:] = ~torch.triu(torch.ones(K, K, dtype=torch.bool),
                                      diagonal=1)
        self.register_buffer("replay_allowed", allowed, persistent=False)

    # ------------------------------------------------------------------
    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        B, C, H, W = imgs.shape
        p = self.patch_size
        g = H // p
        x = imgs.reshape(B, C, g, p, g, p)
        return x.permute(0, 2, 4, 3, 5, 1).reshape(B, g * g, p * p * C)

    def unpatchify(self, pred: torch.Tensor) -> torch.Tensor:
        B, N, _ = pred.shape
        p, g, C = self.patch_size, self.grid_size, self.in_channels
        x = pred.reshape(B, g, g, p, p, C)
        return x.permute(0, 5, 1, 3, 2, 4).reshape(B, C, g * p, g * p)

    # ------------------------------------------------------------------
    @torch.compiler.disable
    def _select_order(self, x_img: torch.Tensor, coords: torch.Tensor):
        """
        Pass 1 (no grad, sequential, KV-cached): determine the admission
        order by actually generating. Returns:
            order:        (B, K) long — patch index admitted at each step
            admitted:     (B, N) bool
            first_scores: (B, N) step-0 scores (nullity alarms)
            mem_seq:      (B, K, D) top-layer memory outputs (pre-norm) —
                          used by the smoke test to assert replay parity;
                          at eval this IS the deployment compute path.
        """
        with torch.no_grad():
            B, N, D = x_img.shape
            K = self.memory_tokens
            H = self.core[0].attn.num_heads
            hd = self.core[0].attn.head_dim

            ks, vs = [], []
            h = x_img
            for blk in self.core:
                h, k, v = blk.forward_full(h, coords, return_kv=True)
                ks.append(k)
                vs.append(v)
            h_img = h
            keys  = self.norm(h_img)

            k_caches = [torch.cat([k, k.new_zeros(B, H, K, hd)], dim=2)
                        for k in ks]
            v_caches = [torch.cat([v, v.new_zeros(B, H, K, hd)], dim=2)
                        for v in vs]

            admitted = torch.zeros(B, N, dtype=torch.bool, device=x_img.device)
            order    = torch.zeros(B, K, dtype=torch.long, device=x_img.device)
            mem_seq  = []
            first_scores = None

            ctrl = self.start_query.view(1, D).expand(B, D)
            for t in range(K):
                s = torch.einsum("bd,bnd->bn", self.score_q(ctrl), keys) \
                    / (D ** 0.5)
                if t == 0:
                    first_scores = s.clone()
                s = s.masked_fill(admitted, float("-inf"))
                sel = s.argmax(dim=1)                          # (B,)
                order[:, t] = sel
                admitted.scatter_(1, sel.unsqueeze(1), True)

                seed = h_img.gather(
                    1, sel.view(B, 1, 1).expand(-1, -1, D)) + self.marker
                c_new = coords[sel].unsqueeze(1)               # (B,1,2)
                x_new = seed
                for l, blk in enumerate(self.core):
                    x_new = blk.forward_step(x_new, c_new,
                                             k_caches[l], v_caches[l], N + t)
                mem_seq.append(x_new[:, 0])
                ctrl = self.norm(x_new[:, 0])

            return order, admitted, first_scores, torch.stack(mem_seq, dim=1)

    # ------------------------------------------------------------------
    def _encode(self, imgs: torch.Tensor):
        """Pass 1 (order) + pass 2 (parallel replay with gradients)."""
        x_img, coords = self.patch_embed(imgs)                 # (B,N,D), (N,2)
        order, admitted, first_scores, _ = self._select_order(x_img, coords)

        B, N, D = x_img.shape
        # Image pass with grad — identical math to phase A of pass 1.
        h = x_img
        for blk in self.core:
            h = blk.forward_full(h, coords)
        h_img = h
        keys  = self.norm(h_img)

        idx   = order.unsqueeze(-1).expand(-1, -1, D)          # (B,K,D)
        seeds = h_img.gather(1, idx) + self.marker
        seq        = torch.cat([x_img, seeds], dim=1)          # (B,S,D)
        seq_coords = torch.cat([coords.unsqueeze(0).expand(B, -1, -1),
                                coords[order]], dim=1)         # (B,S,2)
        mask = self.replay_allowed[None, None]                 # (1,1,S,S)
        for blk in self.core:
            seq = blk.forward_full(seq, seq_coords, attn_mask=mask)
        mem_out = seq[:, N:]                                   # (B,K,D)

        # Controllers & scores, recomputed in parallel WITH gradients —
        # the score head learns through the confidence gate below.
        ctrl = torch.cat([self.start_query.view(1, 1, D).expand(B, 1, D),
                          self.norm(mem_out[:, :-1])], dim=1)  # (B,K,D)
        scores = torch.einsum("bkd,bnd->bkn", self.score_q(ctrl), keys) \
            / (D ** 0.5)                                       # (B,K,N)
        gate = torch.sigmoid(
            scores.gather(2, order.unsqueeze(-1)).squeeze(-1)) # (B,K)

        return h_img, mem_out, order, admitted, first_scores, gate, coords

    def _mem_grid(self, mem_vals, order):
        """Scatter per-step memory values (B,K,D) back onto the grid."""
        B, K, D = mem_vals.shape
        grid = mem_vals.new_zeros(B, self.n_patches, D)
        return grid.scatter(1, order.unsqueeze(-1).expand(-1, -1, D),
                            mem_vals)

    def _decode(self, mem_grid, admitted, coords):
        enc = self.decoder_embed(mem_grid)
        mask_tok = self.mask_token.to(enc.dtype)
        x = torch.where(admitted.unsqueeze(-1), enc, mask_tok.expand_as(enc))
        for blk in self.decoder:
            x = blk(x, coords)
        return self.decoder_pred(self.decoder_norm(x))

    def _step_grids(self, order):
        """(B,N) long grids: raw admission step and eval bin (-1 = never)."""
        B, K = order.shape
        steps = torch.arange(K, device=order.device).unsqueeze(0).expand(B, K)
        step_grid = torch.full((B, self.n_patches), -1, dtype=torch.long,
                               device=order.device)
        step_grid = step_grid.scatter(1, order, steps)
        bin_grid = torch.where(step_grid >= 0, step_grid // self.bin_size,
                               step_grid)
        return step_grid, bin_grid

    # ------------------------------------------------------------------
    def forward(self, imgs: torch.Tensor):
        """Same contract as NeocoreAE.forward: (loss_rec, overlap_r1,
        admit_corr, stability). admit_corr uses the raw admission STEP
        (0..K-1); stability is constant 1.0 (nothing to evict)."""
        h_img, mem_out, order, admitted, first_scores, gate, coords = \
            self._encode(imgs)

        mem_dec = self.norm(mem_out) * (1.0 + gate).unsqueeze(-1)
        pred = self._decode(self._mem_grid(mem_dec, order), admitted, coords)

        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mu  = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mu) / (var + 1e-6) ** 0.5

        loss_patch = ((pred.float() - target.float()) ** 2).mean(dim=-1)
        m = (~admitted).float()
        loss_rec = (loss_patch * m).sum() / m.sum().clamp(min=1.0)

        step_grid, _ = self._step_grids(order)
        overlap, corr = NeocoreAE._nullity_alarms(
            first_scores, admitted, step_grid, self.memory_tokens)
        stability = torch.ones((), device=imgs.device)
        return loss_rec, overlap, corr, stability

    # ------------------------------------------------------------------
    def forward_features(self, imgs: torch.Tensor):
        """Probe interface: (feats, admitted, admit_round). Grid features
        with each admitted position carrying its MEMORY token's output
        (that is where the accumulated representation lives here) and
        unadmitted positions their frozen image features."""
        h_img, mem_out, order, admitted, _, _, _ = self._encode(imgs)
        feats = self.norm(h_img).scatter(
            1, order.unsqueeze(-1).expand(-1, -1, h_img.shape[-1]),
            self.norm(mem_out))
        _, bin_grid = self._step_grids(order)
        return feats, admitted, bin_grid

    @torch.no_grad()
    def reconstruct(self, imgs: torch.Tensor):
        h_img, mem_out, order, admitted, _, gate, coords = self._encode(imgs)
        mem_dec = self.norm(mem_out) * (1.0 + gate).unsqueeze(-1)
        pred = self._decode(self._mem_grid(mem_dec, order), admitted, coords)
        _, bin_grid = self._step_grids(order)
        return self.unpatchify(pred.float()), admitted, bin_grid

    @torch.no_grad()
    def round_trace(self, imgs: torch.Tensor):
        """Per-bin partial decodes (first t·bin_size memory tokens) —
        same shape contract as the loop model's round_trace."""
        h_img, mem_out, order, admitted, _, gate, coords = self._encode(imgs)
        mem_dec = self.norm(mem_out) * (1.0 + gate).unsqueeze(-1)
        step_grid, bin_grid = self._step_grids(order)

        preds, admits = [], []
        for t in range(self.rounds):
            upto = min((t + 1) * self.bin_size, self.memory_tokens)
            adm_t  = (step_grid >= 0) & (step_grid < upto)
            grid_t = self._mem_grid(mem_dec[:, :upto], order[:, :upto])
            preds.append(self._decode(grid_t, adm_t, coords))
            admits.append(adm_t)
        return preds, admits, bin_grid

    def count_parameters(self) -> dict:
        def n(m): return sum(p.numel() for p in m.parameters())
        counts = {
            "patch_embed": n(self.patch_embed),
            "core":        n(self.core) + n(self.norm),
            "admission":   n(self.score_q) + self.start_query.numel()
                           + self.marker.numel(),
            "decoder":     n(self.decoder_embed) + n(self.decoder)
                           + n(self.decoder_norm) + n(self.decoder_pred)
                           + self.mask_token.numel(),
        }
        counts["total"] = n(self)
        return counts
