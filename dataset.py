import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from datasets import load_dataset

from config import Config


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_transforms(cfg: Config, train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(cfg.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(cfg.image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])


class ImageNet100Dataset(Dataset):
    def __init__(self, hf_split, transform=None):
        self.data = hf_split
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple:
        sample = self.data[idx]
        image = sample["image"].convert("RGB")
        label = int(sample["label"])
        if self.transform:
            image = self.transform(image)
        return image, label


def get_dataloaders(cfg: Config) -> tuple[DataLoader, DataLoader]:
    print(f"Loading {cfg.dataset_name} ...")
    raw = load_dataset(cfg.dataset_name, cache_dir=cfg.dataset_cache_dir)

    # Sanity check: labels should be plain integers in [0, num_classes)
    sample = raw["train"][0]
    assert "image" in sample, f"Expected 'image' field, got keys: {list(sample.keys())}"
    assert "label" in sample, f"Expected 'label' field, got keys: {list(sample.keys())}"
    label_val = sample["label"]
    assert isinstance(label_val, int), f"Expected int label, got {type(label_val)}: {label_val}"
    assert 0 <= label_val < cfg.num_classes, (
        f"Label {label_val} out of range [0, {cfg.num_classes})"
    )

    train_ds = ImageNet100Dataset(raw["train"],      get_transforms(cfg, train=True))
    val_ds   = ImageNet100Dataset(raw["validation"], get_transforms(cfg, train=False))

    print(f"  Train: {len(train_ds):,}  |  Val: {len(val_ds):,}")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,  # keeps batch size stable for BN
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
