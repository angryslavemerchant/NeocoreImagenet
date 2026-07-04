from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    # --- Image / patch ---
    image_size: int = 224
    patch_size: int = 64
    in_channels: int = 3

    # --- Model dimensions ---
    d_feat: int = 384       # backbone projection output dim
    d_vec: int = 384        # accumulated output vector dim
    d_loc: int = 128        # LocTracker hidden / output dim
    num_loops: int = 16
    num_classes: int = 100
    move_scale: float = 0.15

    # --- Loss ---
    loc_loss_weight: float = 0.000001
    coverage_loss_weight: float = 0.05
    min_step: float = 0.05  # hinge threshold — steps smaller than this are penalized
                            # 0.05 in normalized coords ≈ 5px on a 224px image

    # --- Movement ---
    random_start: bool = True

    # --- Training ---
    batch_size: int = 4096*2
    num_epochs: int = 90
    lr: float = 8e-3
    weight_decay: float = 0.05
    grad_clip: float = 1.0
    warmup_epochs: int = 10
    num_workers: int = 12

    # --- Data ---
    dataset_name: str = "clane9/imagenet-100"
    dataset_cache_dir: str = "./data"
    jpeg_cache_dir: str = "./jpeg_cache"

    # --- Checkpointing ---
    checkpoint_dir: str = "./checkpoints"
    resume: Optional[str] = None

    # --- Logging ---
    wandb_project: str = "saccade-net-mobilnetv2"
    wandb_entity: Optional[str] = None
    log_interval: int = 50

    # --- Misc ---
    seed: int = 42
    device: str = "cuda"