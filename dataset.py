import json
import time
from pathlib import Path

import torch
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.pipeline import pipeline_def
from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy
from datasets import load_dataset
from tqdm import tqdm

from config import Config


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
JPEG_QUALITY  = 95
_DONE_FLAG    = ".cache_complete"


# ---------------------------------------------------------------------------
# One-time JPEG cache builder
# ---------------------------------------------------------------------------

def _build_split_cache(hf_split, split_dir: Path) -> int:
    """
    Save every image from a HuggingFace split to disk as JPEG.

    Structure: split_dir/{label:03d}/{idx:06d}.jpg
    Three-digit zero-padded label dirs guarantee that lexicographic sort
    (which DALI uses for label assignment) matches numeric order for all
    100 classes.

    Returns the number of images written.
    """
    split_dir.mkdir(parents=True, exist_ok=True)

    for idx, sample in enumerate(tqdm(hf_split, desc=f"  Caching {split_dir.name}")):
        label     = int(sample["label"])
        class_dir = split_dir / f"{label:03d}"
        class_dir.mkdir(exist_ok=True)
        sample["image"].convert("RGB").save(
            class_dir / f"{idx:06d}.jpg",
            quality=JPEG_QUALITY,
            optimize=True,
        )

    count = idx + 1
    (split_dir / "metadata.json").write_text(json.dumps({"count": count}))
    (split_dir / _DONE_FLAG).touch()  # written last — incomplete builds won't be trusted
    return count


def _ensure_cache(cfg: Config) -> tuple[int, int]:
    """
    Build the JPEG cache if not already present, then return
    (n_train, n_val) sample counts.
    """
    root      = Path(cfg.jpeg_cache_dir)
    train_dir = root / "train"
    val_dir   = root / "validation"

    train_done = (train_dir / _DONE_FLAG).exists()
    val_done   = (val_dir   / _DONE_FLAG).exists()

    if not (train_done and val_done):
        print(f"JPEG cache not found at {root}. Building (one-time cost) ...")
        # HF Hub throws transient 5xx (502 killed a cloud run at boot,
        # 2026-07-15) — retry with backoff before giving up.
        for attempt in range(6):
            try:
                raw = load_dataset(cfg.dataset_name, cache_dir=cfg.dataset_cache_dir)
                break
            except Exception as e:
                if attempt == 5:
                    raise
                wait = 60 * (attempt + 1)
                print(f"load_dataset failed ({e!r}) — "
                      f"retry {attempt + 1}/5 in {wait}s")
                time.sleep(wait)
        if not train_done:
            _build_split_cache(raw["train"], train_dir)
        if not val_done:
            _build_split_cache(raw["validation"], val_dir)
        total_gb = sum(f.stat().st_size for f in root.rglob("*.jpg")) / 1e9
        print(f"Cache complete. {total_gb:.1f} GB on disk.\n")

    n_train = json.loads((train_dir / "metadata.json").read_text())["count"]
    n_val   = json.loads((val_dir   / "metadata.json").read_text())["count"]
    return n_train, n_val


# ---------------------------------------------------------------------------
# DALI pipeline
# ---------------------------------------------------------------------------

@pipeline_def
def _imagenet_pipeline(data_dir: str, crop_size: int, is_training: bool):
    """
    Full augmentation pipeline running on GPU via DALI.

    JPEG decode happens in hardware on Blackwell (nvjpeg2k / nvJPEG).
    All subsequent ops — resize, crop, flip, color jitter, normalize —
    run as CUDA kernels. The CPU is only involved in disk I/O and
    scheduling, not pixel processing.

    Train:  hardware decode + random crop → resize → color jitter → normalize + flip
    Val:    hardware decode → resize shorter=256 → center crop → normalize
    """
    images, labels = fn.readers.file(
        file_root=data_dir,
        random_shuffle=is_training,
        name="Reader",
    )

    if is_training:
        # Fused GPU JPEG decode + random crop in one op — fastest path on Blackwell
        images = fn.decoders.image_random_crop(
            images,
            device="mixed",            # CPU reads bytes, GPU decodes
            output_type=types.RGB,
            random_aspect_ratio=[0.75, 4.0 / 3.0],
            random_area=[0.08, 1.0],
            num_attempts=100,
        )
        images = fn.resize(images, device="gpu", resize_x=crop_size, resize_y=crop_size)

        # Color jitter — all on GPU
        images = fn.brightness_contrast(
            images, device="gpu",
            brightness=fn.random.uniform(range=[0.6, 1.4]),
            contrast=fn.random.uniform(range=[0.6, 1.4]),
        )
        images = fn.hsv(
            images, device="gpu",
            saturation=fn.random.uniform(range=[0.6, 1.4]),
            hue=fn.random.uniform(range=[-36.0, 36.0]),  # ±0.1 in torchvision = ±36°
        )

        # Normalize + random horizontal flip (fused in one kernel)
        images = fn.crop_mirror_normalize(
            images, device="gpu",
            dtype=types.FLOAT,
            output_layout="CHW",
            mean=[m * 255 for m in IMAGENET_MEAN],
            std=[s * 255 for s in IMAGENET_STD],
            mirror=fn.random.coin_flip(probability=0.5),
        )
    else:
        images = fn.decoders.image(images, device="mixed", output_type=types.RGB)
        images = fn.resize(images, device="gpu", resize_shorter=256)
        images = fn.crop_mirror_normalize(
            images, device="gpu",
            dtype=types.FLOAT,
            output_layout="CHW",
            crop=[crop_size, crop_size],   # center crop
            mean=[m * 255 for m in IMAGENET_MEAN],
            std=[s * 255 for s in IMAGENET_STD],
        )

    return images, labels.gpu()


# ---------------------------------------------------------------------------
# Thin wrapper — same (images, labels) interface as a standard DataLoader
# ---------------------------------------------------------------------------

class DALILoader:
    """
    Wraps DALIClassificationIterator to look exactly like a PyTorch DataLoader.

    DALI outputs tensors already on GPU, so .to(device) calls in train.py
    and evaluate.py are no-ops. No code changes needed there.
    """

    def __init__(self, dali_iter, n_batches: int):
        self._iter    = dali_iter
        self._n_batches = n_batches

    def __len__(self) -> int:
        return self._n_batches

    def __iter__(self):
        for batch in self._iter:
            images = batch[0]["data"]
            labels = batch[0]["label"].squeeze(-1).long()
            yield images, labels


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def get_dataloaders(cfg: Config) -> tuple[DALILoader, DALILoader]:
    n_train, n_val = _ensure_cache(cfg)
    print(f"Dataset ready — Train: {n_train:,}  |  Val: {n_val:,}")

    root = Path(cfg.jpeg_cache_dir)

    # prefetch_queue_depth 4 (DALI default 2): once the model outruns the
    # loader (Neocore ck3 hit data_wait 85% at depth 2), a deeper queue
    # lets decode run further ahead. Costs a little extra GPU memory only.
    train_pipe = _imagenet_pipeline(
        data_dir=str(root / "train"),
        crop_size=cfg.image_size,
        is_training=True,
        batch_size=cfg.batch_size,
        num_threads=cfg.num_workers,
        device_id=0,
        seed=cfg.seed,
        prefetch_queue_depth=4,
    )
    val_pipe = _imagenet_pipeline(
        data_dir=str(root / "validation"),
        crop_size=cfg.image_size,
        is_training=False,
        batch_size=cfg.batch_size,
        num_threads=cfg.num_workers,
        device_id=0,
        seed=cfg.seed,
        prefetch_queue_depth=4,
    )

    train_pipe.build()
    val_pipe.build()

    train_loader = DALILoader(
        DALIClassificationIterator(
            train_pipe,
            size=n_train,
            last_batch_policy=LastBatchPolicy.DROP,
            auto_reset=True,
        ),
        n_batches=n_train // cfg.batch_size,  # DROP policy so floor div
    )
    val_loader = DALILoader(
        DALIClassificationIterator(
            val_pipe,
            size=n_val,
            last_batch_policy=LastBatchPolicy.PARTIAL,
            auto_reset=True,
        ),
        n_batches=(n_val + cfg.batch_size - 1) // cfg.batch_size,
    )

    return train_loader, val_loader