import torch
import torch.nn as nn

from config import Config
from utils import extract_patch, clamp_pos


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise separable convolution: depthwise -> BN -> ReLU -> pointwise -> BN -> ReLU.
    Significantly fewer parameters than a standard conv at similar representational power.
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            # Depthwise: one filter per input channel
            nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            # Pointwise: mix channels
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ---------------------------------------------------------------------------
# Subnetworks
# ---------------------------------------------------------------------------

class PatchCNN(nn.Module):
    """
    Extracts features from a single patch.

    Input:  (B, C, patch_size, patch_size)
    Output: (B, d_feat)

    Architecture: standard conv stem -> depthwise separable stack -> global avg pool.
    Weight-shared across all loop steps — the same parameters see every patch.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.net = nn.Sequential(
            # Standard conv stem for initial channel expansion
            nn.Conv2d(cfg.in_channels, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # Depthwise separable stack
            DepthwiseSeparableConv(32, 64),                      # 32x32
            DepthwiseSeparableConv(64, 128, stride=2),           # -> 16x16
            DepthwiseSeparableConv(128, 256, stride=2),          # -> 8x8
            DepthwiseSeparableConv(256, 384, stride=2),          # -> 4x4
            DepthwiseSeparableConv(384, cfg.d_feat),             # 4x4
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.net(x)).flatten(1)  # (B, d_feat)


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

    This is intentionally a blind tracker — it never sees absolute image coordinates,
    only the sequence of actual (post-clamp) deltas. Whatever spatial representation
    emerges in its hidden state is driven purely by task loss and the auxiliary
    position supervision signal.

    Input per step:  actual_delta (B, 2) — movement that actually happened after clamping
    Output per step: loc_t (B, d_loc) — relative position encoding for this step
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
        # GRU expects (seq_len, B, input_size); we process one step at a time
        out, h = self.gru(actual_delta.unsqueeze(0), h)  # out: (1, B, d_loc)
        return out.squeeze(0), h                          # loc_t: (B, d_loc)


class AuxLocHead(nn.Module):
    """
    Auxiliary linear head: loc_t -> predicted 2D displacement from start.

    Used only during training. Provides a direct supervision signal to LocTracker
    so its hidden state is anchored to real spatial meaning. Dropped at inference.
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

    Input:  f_t (B, d_feat) concat loc_t (B, d_loc) -> (B, d_feat + d_loc)
    Output: v_t (B, d_vec) — added to vec at each loop step
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
    Looped saccadic vision model.

    State at each step t:
        pos  (B, 2)       — current patch center in normalized [-1, 1] coords
        vec  (B, d_vec)   — accumulated content vector (what has been seen)
        h    (1, B, d_loc)— LocTracker GRU hidden state (where relative to start)

    Per loop step:
        1. Extract patch at pos (differentiable via grid_sample)
        2. f_t     = CNN(patch)
        3. delta   = MoveNet(f_t, vec)        — where to go
        4. pos     = clamp(pos + delta)       — apply + stay in bounds
        5. actual_delta = new_pos - old_pos   — what actually happened
        6. loc_t, h = LocTracker(actual_delta, h)
        7. v_t     = OutputNet(f_t, loc_t)
        8. vec     = vec + v_t

    Output: task_head(vec_T) -> logits
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.cnn = PatchCNN(cfg)
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
            logits:      (B, num_classes)
            aux_preds:   list of num_loops tensors, each (B, 2)
                         predicted cumulative displacement from start at each step
            pos_history: list of num_loops tensors, each (B, 2)
                         actual patch center positions at each step
            pos_0:       (B, 2) initial position (all zeros = image center)
        """
        B = x.size(0)
        device = x.device

        # Initialize state
        pos = torch.zeros(B, 2, device=device)             # image center
        vec = torch.zeros(B, self.cfg.d_vec, device=device)
        h   = torch.zeros(1, B, self.cfg.d_loc, device=device)
        pos_0 = pos.clone()

        aux_preds   = []
        pos_history = []

        for _ in range(self.cfg.num_loops):
            # 1. Extract patch at current position
            patch = extract_patch(x, pos, self.cfg.patch_size, self.cfg.image_size)

            # 2. Features from patch
            f_t = self.cnn(patch)

            # 3. Predict movement
            delta = self.move_net(f_t, vec)

            # 4. Apply and clamp
            new_pos = clamp_pos(pos + delta, self.cfg.patch_size, self.cfg.image_size)
            actual_delta = new_pos - pos  # what movement actually happened post-clamp
            pos = new_pos

            # 5. Update location tracker with actual (not requested) delta
            loc_t, h = self.loc_tracker(actual_delta, h)

            # 6. Output contribution
            v_t = self.output_net(f_t, loc_t)
            vec = vec + v_t

            # Record for loss and visualization
            pos_history.append(pos)
            aux_preds.append(self.aux_loc_head(loc_t))

        logits = self.task_head(vec)
        return logits, aux_preds, pos_history, pos_0

    def count_parameters(self) -> dict:
        """Parameter count breakdown by subnetwork."""
        def n(module):
            return sum(p.numel() for p in module.parameters())

        return {
            "cnn":          n(self.cnn),
            "move_net":     n(self.move_net),
            "loc_tracker":  n(self.loc_tracker),
            "aux_loc_head": n(self.aux_loc_head),
            "output_net":   n(self.output_net),
            "task_head":    n(self.task_head),
            "total":        n(self),
        }
