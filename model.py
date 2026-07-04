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

    Only the projection layer (1280 -> d_feat) is trained, giving MoveNet
    and OutputNet a learned adapter onto the pretrained feature space.

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
# Subnetworks  (unchanged from original)
# ---------------------------------------------------------------------------

class MoveNet(nn.Module):
    """
    Predicts where to look next given current features and accumulated context.

    Input:  f_t (B, d_feat) concat vec_t (B, d_vec) -> (B, d_feat + d_vec)
    Output: delta (B, 2) — movement in normalized image coords, tanh-bounded

    The tanh ensures the delta is bounded; move_scale controls max step size.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        in_dim = cfg.d_feat + cfg.d_vec
        hidden = 512
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2),
        )
        self.move_scale = cfg.move_scale

    def forward(self, f_t: torch.Tensor, vec_t: torch.Tensor) -> torch.Tensor:
        x = torch.cat([f_t, vec_t], dim=-1)
        return torch.tanh(self.net(x)) * self.move_scale  # (B, 2)


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
    Combines patch features with spatial context to produce each step's
    contribution to the accumulated output vector.

    Input:  f_t (B, d_feat) concat loc_t (B, d_loc)
    Output: v_t (B, d_vec)
    """

    def __init__(self, cfg: Config):
        super().__init__()
        in_dim = cfg.d_feat + cfg.d_loc
        hidden = 512
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, cfg.d_vec),
        )

    def forward(self, f_t: torch.Tensor, loc_t: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([f_t, loc_t], dim=-1))  # (B, d_vec)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class SaccadeNet(nn.Module):
    """
    Looped saccadic vision model — MobileNetV2 backbone edition.

    Backbone is fully frozen; only MoveNet, LocTracker, OutputNet,
    AuxLocHead, TaskHead, and the projection layer are trained.
    This lets the movement policy learn against stable, discriminative
    features from step 1 rather than fighting a randomly-initialized CNN.

    State at each step t:
        pos  (B, 2)        — current patch center in normalized [-1, 1] coords
        vec  (B, d_vec)    — accumulated content vector
        h    (1, B, d_loc) — LocTracker GRU hidden state

    Per loop step:
        1. Extract patch at pos (differentiable via grid_sample)
        2. f_t      = Backbone(patch)         — frozen features + trained proj
        3. delta    = MoveNet(f_t, vec)        — where to go
        4. pos      = clamp(pos + delta)
        5. loc_t, h = LocTracker(actual_delta, h)
        6. v_t      = OutputNet(f_t, loc_t)
        7. vec      = vec + v_t

    Output: task_head(vec_T) -> logits
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.backbone = MobileNetBackbone(cfg)
        self.move_net = MoveNet(cfg)
        self.loc_tracker = LocTracker(cfg)
        self.aux_loc_head = AuxLocHead(cfg)
        self.output_net = OutputNet(cfg)
        self.task_head = nn.Linear(cfg.d_vec, cfg.num_classes)

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Args:
            x: (B, C, H, W) input image batch

        Returns:
            logits:        (B, num_classes)
            aux_preds:     list[num_loops] of (B, 2) — predicted cumulative displacement
            pos_history:   list[num_loops] of (B, 2) — actual patch centers
            pos_0:         (B, 2) initial position
            delta_history: list[num_loops] of (B, 2) — raw pre-clamp MoveNet deltas
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
        h     = torch.zeros(1, B, self.cfg.d_loc, device=device)
        pos_0 = pos.clone()

        aux_preds     = []
        pos_history   = []
        delta_history = []

        for _ in range(self.cfg.num_loops):
            patch = extract_patch(x, pos, self.cfg.patch_size, self.cfg.image_size)
            f_t   = self.backbone(patch)

            delta = self.move_net(f_t, vec)
            delta_history.append(delta)

            new_pos      = clamp_pos(pos + delta, self.cfg.patch_size, self.cfg.image_size)
            actual_delta = new_pos - pos
            pos          = new_pos

            loc_t, h = self.loc_tracker(actual_delta, h)
            v_t      = self.output_net(f_t, loc_t)
            vec      = vec + v_t

            pos_history.append(pos)
            aux_preds.append(self.aux_loc_head(loc_t))

        logits = self.task_head(vec)
        return logits, aux_preds, pos_history, pos_0, delta_history

    def count_parameters(self) -> dict:
        """Parameter count breakdown. Backbone frozen params listed separately."""
        def n(module):
            return sum(p.numel() for p in module.parameters())

        def n_trainable(module):
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        return {
            "backbone (frozen)":  n(self.backbone.features),
            "backbone_proj":      n(self.backbone.proj),
            "move_net":           n(self.move_net),
            "loc_tracker":        n(self.loc_tracker),
            "aux_loc_head":       n(self.aux_loc_head),
            "output_net":         n(self.output_net),
            "task_head":          n(self.task_head),
            "total_trainable":    n_trainable(self),
            "total_all":          n(self),
        }