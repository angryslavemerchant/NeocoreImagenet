from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    # --- Image / patch ---
    image_size: int = 224
    patch_size: int = 32
    in_channels: int = 3

    # --- Model dimensions ---
    d_feat: int = 384       # CNN output / feature dim
    d_vec: int = 384        # accumulated output vector dim
    d_loc: int = 128        # LocTracker hidden / output dim
    num_loops: int = 16
    num_classes: int = 100
    move_scale: float = 0.5  # max delta magnitude in normalized coords (tanh * this)

    # --- Loss ---
    loc_loss_weight: float = 0.1  # weight of auxiliary location loss

    # --- Training ---
    batch_size: int = 2048
    num_epochs: int = 180
    lr: float = 4e-3
    weight_decay: float = 0.05
    grad_clip: float = 1.0
    warmup_epochs: int = 10
    num_workers: int = 8

    # --- Data ---
    dataset_name: str = "clane9/imagenet-100"
    dataset_cache_dir: str = "./data"         # HuggingFace arrow cache
    jpeg_cache_dir: str = "./jpeg_cache"    # JPEGs on disk for DALI to read

    # --- Checkpointing ---
    checkpoint_dir: str = "./checkpoints"
    resume: Optional[str] = None  # path to checkpoint to resume from

    # --- Logging ---
    wandb_project: str = "saccade-net"
    wandb_entity: Optional[str] = None
    log_interval: int = 50  # steps between wandb logs

    # --- Misc ---
    seed: int = 42
    device: str = "cuda"