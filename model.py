import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

from config import Config
from utils import extract_patch, clamp_pos


# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------

class MobileNetBackbone(nn.Module):
    """
    Frozen MobileNetV2 feature extractor + trainable linear projection.

    The backbone is permanently frozen — no gradients flow into it.
    torch.no_grad() on the forward pass avoids storing activations for
    backward, which saves significant memory across 16 loop steps.

    Only the projection layer (1280 -> d_feat) is trained, giving
    OutputNet a learned adapter onto the pretrained feature space.

    Input:  (B, C, patch_size, patch_size)  — works fine for 64px patches
    Output: (B, d_feat)
    """

    def __init__(self, cfg: Config):
        super().__init__()
        backbone = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        self.features = backbone.features   # (B, 1280, H', W') out
        for p in self.features.parameters():
            p.requires_grad = False

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(1280, cfg.d_feat)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            feat = self.features(x)             # (B, 1280, H', W')
        feat = self.pool(feat).flatten(1)       # (B, 1280)
        return self.proj(feat)                  # (B, d_feat)


# ---------------------------------------------------------------------------
# Subnetworks
# ---------------------------------------------------------------------------

class LocTracker(nn.Module):
    """
    Learned dead-reckoning: tracks relative position from movement history alone.

    Input per step:  actual_delta (B, 2)
    Output per step: loc_t (B, d_loc)
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.gru = nn.GRU(
            input_size=2,
            hidden_size=cfg.d_loc,
            num_layers=1,
            batch_first=False,
        )

    def forward(
        self,
        actual_delta: torch.Tensor,  # (B, 2)
        h: torch.Tensor,             # (1, B, d_loc)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out, h = self.gru(actual_delta.unsqueeze(0), h)
        return out.squeeze(0), h     # loc_t: (B, d_loc)


class AuxLocHead(nn.Module):
    """
    Auxiliary linear head: loc_t -> predicted 2D displacement from start.
    Training only — dropped at inference.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.head = nn.Linear(cfg.d_loc, 2)

    def forward(self, loc_t: torch.Tensor) -> torch.Tensor:
        return self.head(loc_t)  # (B, 2)


class OutputNet(nn.Module):
    """
    Unified network: given patch features and spatial context, produces
    both the content contribution AND the movement decision.

    Replaces the separate MoveNet + OutputNet pair. A shared trunk
    extracts a joint semantic representation; two lightweight heads
    branch from it — one for content accumulation, one for navigation.

    The key insight: where to look next IS a function of what you just
    understood. A separate MoveNet reading a classification-optimized vec
    creates a goal mismatch. Here the same representation drives both.

    Input:  f_t   (B, d_feat)  — patch features (frozen backbone)
            loc   (B, d_loc)   — spatial context from PREVIOUS step
                                 (zeros at step 0)
    Output: v_t   (B, d_vec)   — content contribution, added to vec
            delta (B, 2)       — movement, tanh-bounded by move_scale
    """

    def __init__(self, cfg: Config):
        super().__init__()
        in_dim = cfg.d_feat + cfg.d_loc
        hidden = 512
        self.move_scale = cfg.move_scale

        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.content_head = nn.Linear(hidden, cfg.d_vec)
        self.move_head    = nn.Linear(hidden, 2)

    def forward(
        self, f_t: torch.Tensor, loc: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x     = self.trunk(torch.cat([f_t, loc], dim=-1))
        v_t   = self.content_head(x)
        delta = torch.tanh(self.move_head(x)) * self.move_scale
        return v_t, delta  # (B, d_vec), (B, 2)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class SaccadeNet(nn.Module):
    """
    Looped saccadic vision model — unified OutputNet edition.

    MoveNet is removed. OutputNet now owns both content accumulation
    and movement decisions via a shared trunk + two heads. This means
    the same semantic reasoning that decides "what did I see" also
    decides "where should I look next" — no goal mismatch.

    State at each step t:
        pos  (B, 2)        — current patch center in normalized [-1, 1] coords
        vec  (B, d_vec)    — accumulated content vector
        loc  (B, d_loc)    — spatial context (previous step's LocTracker output)
        h    (1, B, d_loc) — LocTracker GRU hidden state

    Per loop step:
        1. Extract patch at pos
        2. f_t          = Backbone(patch)
        3. v_t, delta   = OutputNet(f_t, loc)   ← loc from previous step
        4. vec          = vec + v_t
        5. pos          = clamp(pos + delta)
        6. loc, h       = LocTracker(actual_delta, h)

    Output: task_head(vec_T) -> logits
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.backbone     = MobileNetBackbone(cfg)
        self.loc_tracker  = LocTracker(cfg)
        self.aux_loc_head = AuxLocHead(cfg)
        self.output_net   = OutputNet(cfg)
        self.task_head    = nn.Linear(cfg.d_vec, cfg.num_classes)

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Args:
            x: (B, C, H, W) input image batch

        Returns:
            logits:       (B, num_classes)
            aux_preds:    list[num_loops] of (B, 2)      — predicted cumulative displacement
            pos_history:  list[num_loops] of (B, 2)      — actual patch centers
            pos_0:        (B, 2)                          — initial position
            feat_history: list[num_loops] of (B, d_feat) — patch features for novelty loss
        """
        B = x.size(0)
        device = x.device

        if self.cfg.random_start and self.training:
            pixel_size = 2.0 / (self.cfg.image_size - 1)
            half_patch = (self.cfg.patch_size - 1) / 2.0 * pixel_size
            lo, hi = -1.0 + half_patch, 1.0 - half_patch
            pos = torch.zeros(B, 2, device=device).uniform_(lo, hi)
        else:
            pos = torch.zeros(B, 2, device=device)  # center at val time

        vec   = torch.zeros(B, self.cfg.d_vec, device=device)
        loc   = torch.zeros(B, self.cfg.d_loc, device=device)  # previous step's loc
        h     = torch.zeros(1, B, self.cfg.d_loc, device=device)
        pos_0 = pos.clone()

        aux_preds    = []
        pos_history  = []
        feat_history = []  # raw patch features — used for EMA novelty loss

        for _ in range(self.cfg.num_loops):
            patch = extract_patch(x, pos, self.cfg.patch_size, self.cfg.image_size)
            f_t   = self.backbone(patch)
            feat_history.append(f_t)

            # OutputNet sees loc from previous step — drives both content and movement
            v_t, delta = self.output_net(f_t, loc)
            vec = vec + v_t

            new_pos      = clamp_pos(pos + delta, self.cfg.patch_size, self.cfg.image_size)
            actual_delta = new_pos - pos
            pos          = new_pos

            # Update loc for the next step
            loc, h = self.loc_tracker(actual_delta, h)

            pos_history.append(pos)
            aux_preds.append(self.aux_loc_head(loc))

        logits = self.task_head(vec)
        return logits, aux_preds, pos_history, pos_0, feat_history

    def count_parameters(self) -> dict:
        """Parameter count breakdown. Backbone frozen params listed separately."""
        def n(module):
            return sum(p.numel() for p in module.parameters())

        def n_trainable(module):
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        return {
            "backbone (frozen)":  n(self.backbone.features),
            "backbone_proj":      n(self.backbone.proj),
            "loc_tracker":        n(self.loc_tracker),
            "aux_loc_head":       n(self.aux_loc_head),
            "output_net (trunk)": n(self.output_net.trunk),
            "output_net (heads)": n(self.output_net.content_head) + n(self.output_net.move_head),
            "task_head":          n(self.task_head),
            "total_trainable":    n_trainable(self),
            "total_all":          n(self),
        }