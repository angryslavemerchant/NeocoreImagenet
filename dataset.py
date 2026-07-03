import os
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TF
from datasets import load_dataset
from tqdm import tqdm

from config import Config


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Sentinel filename — presence means the cache is complete and valid
_CACHE_DONE_FLAG = ".cache_complete"


# ---------------------------------------------------------------------------
# Cache building
# ---------------------------------------------------------------------------

def _build_split_cache(hf_split, split_dir: Path, resize: int = 256):
    """
    Decode every image in hf_split, resize the shorter side to `resize`,
    store as uint8 tensor to disk, and save all labels in one file.

    Stores:
        split_dir/000000.pt  ... (3, H, W) uint8 tensor per image
        split_dir/labels.pt  — list[int] of all labels
        split_dir/.cache_complete — empty sentinel, written last
    """
    split_dir.mkdir(parents=True, exist_ok=True)
    resizer = transforms.Resize(resize)
    labels = []

    for idx, sample in enumerate(tqdm(hf_split, desc=f"  Caching {split_dir.name}")):
        img = resizer(sample["image"].convert("RGB"))
        # to_tensor gives float32 in [0,1]; convert to uint8 to save 4x disk space
        t = (TF.to_tensor(img) * 255).byte()          # (3, H, W) uint8
        torch.save(t, split_dir / f"{idx:06d}.pt")
        labels.append(int(sample["label"]))

    torch.save(labels, split_dir / "labels.pt")
    (split_dir / _CACHE_DONE_FLAG).touch()             # mark complete


def _ensure_cache(cfg: Config):
    """
    Download the HuggingFace dataset and build the tensor cache if it does
    not already exist. Safe to call multiple times — a no-op if already done.
    """
    cache_root = Path(cfg.tensor_cache_dir)
    train_dir  = cache_root / "train"
    val_dir    = cache_root / "validation"

    train_done = (train_dir / _CACHE_DONE_FLAG).exists()
    val_done   = (val_dir   / _CACHE_DONE_FLAG).exists()

    if train_done and val_done:
        return  # nothing to do

    print(f"Tensor cache not found at {cache_root}. Building now (one-time cost)...")
    print(f"Downloading {cfg.dataset_name} ...")
    raw = load_dataset(cfg.dataset_name, cache_dir=cfg.dataset_cache_dir)

    if not train_done:
        _build_split_cache(raw["train"], train_dir)
    if not val_done:
        _build_split_cache(raw["validation"], val_dir)

    print(f"Cache complete. ~{_approx_cache_size_gb(cache_root):.1f} GB on disk.\n")


def _approx_cache_size_gb(root: Path) -> float:
    total = sum(f.stat().st_size for f in root.rglob("*.pt"))
    return total / 1e9


# ---------------------------------------------------------------------------
# Dataset that reads from the tensor cache
# ---------------------------------------------------------------------------

class CachedImageNet100(Dataset):
    """
    Reads pre-decoded uint8 tensors from disk.

    Workers load raw bytes from NVMe (fast) instead of decoding JPEGs (slow).
    mmap=True lets the OS page in only what's needed, keeping RAM pressure low.

    Augmentation pipeline:
        Train — RandomCrop(224) + RandomHorizontalFlip + ColorJitter + Normalize
        Val   — CenterCrop(224) + Normalize
    (RandomResizedCrop is replaced by RandomCrop on already-256px images;
    augmentation diversity is marginally reduced but negligible in practice.)
    """

    def __init__(self, split_dir: Path, train: bool, cfg: Config):
        self.split_dir = split_dir
        self.labels    = torch.load(split_dir / "labels.pt")
        self.n         = len(self.labels)
        self.train     = train

        if train:
            self.spatial = transforms.Compose([
                transforms.RandomCrop(cfg.image_size),
                transforms.RandomHorizontalFlip(),
            ])
            self.color = transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
            )
        else:
            self.spatial = transforms.CenterCrop(cfg.image_size)
            self.color   = None

        self.normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> tuple:
        # mmap=True: OS pages data in on demand rather than loading the full file
        t = torch.load(
            self.split_dir / f"{idx:06d}.pt",
            mmap=True,
            weights_only=True,
        )
        # uint8 -> float32 in [0, 1]
        img = t.float() / 255.0

        # Spatial augmentation (operates on float tensors)
        img = self.spatial(img)

        # Color jitter (train only) — convert to PIL, jitter, back to tensor
        if self.color is not None:
            img = TF.to_pil_image(img)
            img = self.color(img)
            img = TF.to_tensor(img)

        img = self.normalize(img)
        return img, self.labels[idx]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def get_dataloaders(cfg: Config) -> tuple[DataLoader, DataLoader]:
    """
    Ensures the tensor cache exists (building it on first run if needed),
    then returns train and validation DataLoaders backed by the cache.
    """
    _ensure_cache(cfg)

    cache_root = Path(cfg.tensor_cache_dir)
    train_ds = CachedImageNet100(cache_root / "train",      train=True,  cfg=cfg)
    val_ds   = CachedImageNet100(cache_root / "validation", train=False, cfg=cfg)

    print(f"Dataset ready — Train: {len(train_ds):,}  |  Val: {len(val_ds):,}")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
        prefetch_factor=4,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )

    return train_loader, val_loader