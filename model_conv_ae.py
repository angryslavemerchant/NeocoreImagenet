"""
model_conv_ae.py — ConvAE: the no-tricks convolutional control.

WHY (2026-07-17, user-requested benchmark): every Neocore variant selects
WHICH 49 patch tokens survive; this model has no selection, no attention,
no recursion — just a plain strided conv encoder to a 7x7x256 feature map
(= 49 "tokens" of d=256, the exact Neocore bottleneck rate: 12,544
numbers) and a mirrored conv decoder back to pixels. If plain convolution
at the same rate reconstructs comparably, the token machinery isn't
earning its complexity for RECONSTRUCTION (probes are a separate
question); if it falls short, the gap measures what attention+selection
buy at equal rate and ~equal params (~6M).

LOSS CONVENTION (do not misread the number): all-position per-patch
norm-pix MSE over the 16x16 patch grid — there are no dropped tokens, so
dropped-only is undefined. Directly comparable to AE_xattn49 (all-pos
0.211/0.227), NOT to budget25/Neocore dropped-only numbers.

Interface-compatible with the Neocore harness: forward() returns
(loss, overlap_r1=0, admit_corr=0, stability=1); forward_features gives
the 49 bottleneck tokens (LayerNormed) with an all-True admitted mask for
the attentive probe; reconstruct/round_trace satisfy evaluate_neocore
(rounds=1, so admission instruments render trivially).
"""

import torch
import torch.nn as nn


def _enc_stage(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride=2, padding=1),
        nn.GroupNorm(8, cout),
        nn.GELU(),
        nn.Conv2d(cout, cout, 3, padding=1),
        nn.GroupNorm(8, cout),
        nn.GELU(),
    )


def _dec_stage(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(cin, cout, 3, padding=1),
        nn.GroupNorm(8, cout),
        nn.GELU(),
        nn.Conv2d(cout, cout, 3, padding=1),
        nn.GroupNorm(8, cout),
        nn.GELU(),
    )


class ConvAE(nn.Module):
    def __init__(
        self,
        image_size:    int  = 224,
        in_channels:   int  = 3,
        patch_size:    int  = 16,           # for loss patchify + eval only
        bottleneck_dim: int = 256,
        widths: tuple = (48, 96, 192, 256),
        norm_pix_loss: bool = True,
    ):
        super().__init__()
        self.in_channels   = in_channels
        self.patch_size    = patch_size
        self.grid_size     = image_size // patch_size      # 14 (eval compat)
        self.n_patches     = self.grid_size ** 2           # 196
        self.norm_pix_loss = norm_pix_loss
        # eval/probe-compat constants: 5 stride-2 stages -> 7x7 map
        self.bottleneck_hw = image_size // 32              # 7
        self.memory_tokens = self.bottleneck_hw ** 2       # 49
        self.rounds        = 1

        chs = [in_channels, *widths, bottleneck_dim]
        self.encoder = nn.Sequential(*[
            _enc_stage(chs[i], chs[i + 1]) for i in range(len(chs) - 1)
        ])
        dhs = [bottleneck_dim, *reversed(widths)]
        self.decoder = nn.Sequential(*[
            _dec_stage(dhs[i], dhs[i + 1]) for i in range(len(dhs) - 1)
        ])
        self.head = nn.Sequential(                          # 112 -> 224
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(dhs[-1], dhs[-1], 3, padding=1),
            nn.GELU(),
            nn.Conv2d(dhs[-1], in_channels, 3, padding=1),
        )
        self.token_norm = nn.LayerNorm(bottleneck_dim)      # probe features

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
    def _autoencode(self, imgs: torch.Tensor):
        z = self.encoder(imgs)                       # (B, 256, 7, 7)
        out = self.head(self.decoder(z))             # (B, 3, 224, 224)
        return z, out

    def forward(self, imgs: torch.Tensor):
        """(loss, overlap_r1, admit_corr, stability) — alarm slots are
        constants; there is nothing to select and nothing to collapse."""
        _, out = self._autoencode(imgs)

        pred   = self.patchify(out)
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mu  = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mu) / (var + 1e-6) ** 0.5

        # ALL-position loss — see module docstring before comparing.
        loss_rec = ((pred.float() - target.float()) ** 2).mean()

        zero = torch.zeros((), device=imgs.device)
        one  = torch.ones((), device=imgs.device)
        return loss_rec, zero, zero, one

    # ------------------------------------------------------------------
    def forward_features(self, imgs: torch.Tensor):
        """Probe interface: the 49 bottleneck positions as tokens."""
        z = self.encoder(imgs)                                  # (B,256,7,7)
        tok = z.flatten(2).transpose(1, 2)                      # (B,49,256)
        tok = self.token_norm(tok)
        B = tok.shape[0]
        admitted = torch.ones(B, self.memory_tokens, dtype=torch.bool,
                              device=imgs.device)
        admit_round = torch.zeros(B, self.memory_tokens, dtype=torch.long,
                                  device=imgs.device)
        return tok, admitted, admit_round

    @torch.no_grad()
    def reconstruct(self, imgs: torch.Tensor):
        _, out = self._autoencode(imgs)
        pred = self.patchify(out)
        if self.norm_pix_loss:
            mu  = pred.mean(dim=-1, keepdim=True)
            var = pred.var(dim=-1, keepdim=True)
            pred = (pred - mu) / (var + 1e-6) ** 0.5   # match Neocore viz space
        B = imgs.shape[0]
        admitted    = torch.ones(B, self.n_patches, dtype=torch.bool,
                                 device=imgs.device)
        admit_round = torch.zeros(B, self.n_patches, dtype=torch.long,
                                  device=imgs.device)
        return self.unpatchify(pred.float()), admitted, admit_round

    @torch.no_grad()
    def round_trace(self, imgs: torch.Tensor):
        _, out = self._autoencode(imgs)
        pred = self.patchify(out).float()
        B = imgs.shape[0]
        admitted    = torch.ones(B, self.n_patches, dtype=torch.bool,
                                 device=imgs.device)
        admit_round = torch.zeros(B, self.n_patches, dtype=torch.long,
                                  device=imgs.device)
        return [pred], [admitted], admit_round

    def count_parameters(self) -> dict:
        def n(m): return sum(p.numel() for p in m.parameters())
        counts = {
            "encoder": n(self.encoder),
            "decoder": n(self.decoder) + n(self.head),
            "probe_norm": n(self.token_norm),
        }
        counts["total"] = n(self)
        return counts
