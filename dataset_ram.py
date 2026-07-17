"""
dataset_ram.py — RAM-resident raw dataset (the FFCV move, sized for
ImageNet-100).

WHY: the DALI pipeline decodes ORIGINAL-resolution JPEGs every epoch and
caps at ~1.6-2k img/s; the optimized Neocore spends 85% of each epoch
waiting on it (measured 2026-07-17). At 126,689 train images, raw uint8
256x256 is only ~25 GB — small enough to decode ONCE and keep resident.

  build (once, instance-side, needs DALI): jpeg_cache --GPU decode-->
      resize shorter side to 256 --> center crop 256x256 --> uint8 CHW
      --> one .pt blob per split in <jpeg_cache_dir>/ram256/.
      The blob is also the unit of the dataset "bank": push/pull one
      file instead of the HF download -> jpeg cache dance.

  load (every run, torch-only, no DALI import): blob in CPU RAM (or
      --data_device cuda: fully VRAM-resident, zero PCIe per step),
      background prefetch thread does gather + H2D on a side stream,
      augmentation runs as batched CUDA ops.

Augmentation parity with dataset.py (a "pipeline epoch" — close, not
bit-identical; do not mix pipelines within a comparison series):
  train: RandomResizedCrop(224, area [0.08,1], ratio U[3/4,4/3]) + flip
         via one fused affine grid_sample; brightness/contrast U[0.6,1.4],
         saturation U[0.6,1.4], hue U[-36,36]deg (YIQ rotation); normalize.
         Crops are taken from the cached center-256 square rather than the
         full original — the one real distribution difference.
  val:   center crop 224 + normalize. EXACTLY equivalent to the DALI val
         protocol (center-224-of-center-256 == center-224 of the
         shorter-256 resize), so val numbers stay comparable.
"""

import json
import math
import os
import queue
import shutil
import subprocess
import threading
import time
from pathlib import Path

import torch
import torch.nn.functional as F

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
SIDE = 256          # cached square side
CROP = 224          # model input


# ---------------------------------------------------------------------------
# One-time build (instance-side; imports DALI lazily)
# ---------------------------------------------------------------------------

def _build_split_blob(jpeg_split_dir: Path, out_file: Path, batch: int = 512):
    """GPU-decode every JPEG in the split into one uint8 (N,3,256,256) blob."""
    import nvidia.dali.fn as fn
    import nvidia.dali.types as types
    from nvidia.dali.pipeline import pipeline_def
    from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy

    n = json.loads((jpeg_split_dir / "metadata.json").read_text())["count"]

    @pipeline_def
    def build_pipe():
        imgs, labels = fn.readers.file(
            file_root=str(jpeg_split_dir), random_shuffle=False, name="Reader")
        imgs = fn.decoders.image(imgs, device="mixed", output_type=types.RGB)
        imgs = fn.resize(imgs, device="gpu", resize_shorter=SIDE)
        imgs = fn.crop_mirror_normalize(          # center crop, NO normalize
            imgs, device="gpu", dtype=types.UINT8, output_layout="CHW",
            crop=[SIDE, SIDE], mean=[0.0] * 3, std=[1.0] * 3)
        return imgs, labels.gpu()

    pipe = build_pipe(batch_size=batch, num_threads=8, device_id=0, seed=0)
    pipe.build()
    it = DALIGenericIterator(pipe, ["data", "label"], size=n,
                             last_batch_policy=LastBatchPolicy.PARTIAL,
                             auto_reset=False)

    images = torch.empty(n, 3, SIDE, SIDE, dtype=torch.uint8)
    labels = torch.empty(n, dtype=torch.int16)
    i = 0
    for out in it:
        d = out[0]["data"]                       # (b,3,256,256) uint8 cuda
        l = out[0]["label"].squeeze(-1)
        b = min(d.shape[0], n - i)               # PARTIAL pads the tail
        images[i:i + b] = d[:b].cpu()
        labels[i:i + b] = l[:b].to(torch.int16).cpu()
        i += b
    assert i == n, f"blob build wrote {i}/{n} images"

    out_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_file.with_suffix(".tmp")
    torch.save({"images": images, "labels": labels}, tmp)
    tmp.rename(out_file)                          # atomic-ish: no torn blobs
    print(f"[ram256] built {out_file.name}: {n} images, "
          f"{images.numel() / 1e9:.1f} GB")


# ---------------------------------------------------------------------------
# Dataset bank: pull prebuilt blobs at boot instead of the HF -> jpeg cache
# -> blob-build dance (~15 min and two external dependencies saved per boot).
# Backend chosen by RAM_BANK env: "wandb" (default; creds already on every
# instance), "gdrive" (rclone; dormant until RCLONE_DRIVE_TOKEN is set in
# secrets.env), "none" (skip — always build locally).
# Upload side: vast/upload_blobs.py, run once on an instance holding blobs.
# ---------------------------------------------------------------------------

BANK_REF_DEFAULT = "luckymushy-individual/neocore/imagenet100-ram256:latest"


def _bank_pull_wandb(blob_dir: Path) -> bool:
    import wandb
    ref = os.environ.get("RAM_BANK_REF", BANK_REF_DEFAULT)
    for attempt in range(8):    # patient, but not build-blockingly so
        try:
            art = wandb.Api().artifact(ref, type="dataset")
            art.download(root=str(blob_dir))
            print(f"[bank] pulled {ref} -> {blob_dir}")
            return True
        except Exception as e:
            msg = str(e).lower()
            if "not found" in msg or "does not exist" in msg:
                print(f"[bank] artifact {ref} not published yet — will build")
                return False
            wait = min(300, 60 * (attempt + 1))
            print(f"[bank] wandb pull failed ({e!r}) — "
                  f"retry {attempt + 1}/7 in {wait}s")
            time.sleep(wait)
    return False


def _bank_pull_gdrive(blob_dir: Path) -> bool:
    """rclone pull from Drive. Requires RCLONE_DRIVE_TOKEN (the JSON from
    `rclone authorize "drive"`) in the environment; remote path override
    via RAM_BANK_DRIVE_PATH. UNTESTED until the token exists."""
    token = os.environ.get("RCLONE_DRIVE_TOKEN")
    if not token:
        print("[bank] RCLONE_DRIVE_TOKEN not set — skipping gdrive")
        return False
    if shutil.which("rclone") is None:
        subprocess.run("curl -fsSL https://rclone.org/install.sh | bash",
                       shell=True, check=False)
        if shutil.which("rclone") is None:
            print("[bank] rclone install failed")
            return False
    conf = Path.home() / ".config" / "rclone" / "rclone.conf"
    conf.parent.mkdir(parents=True, exist_ok=True)
    conf.write_text(f"[gdrive]\ntype = drive\nscope = drive\n"
                    f"token = {token}\n")
    src = os.environ.get("RAM_BANK_DRIVE_PATH", "gdrive:NeocoreBank/ram256")
    r = subprocess.run(["rclone", "copy", src, str(blob_dir),
                        "--transfers", "4", "--retries", "8",
                        "--low-level-retries", "20"], check=False)
    return r.returncode == 0


def _bank_pull(blob_dir: Path) -> bool:
    kind = os.environ.get("RAM_BANK", "wandb")
    if kind == "none":
        return False
    blob_dir.mkdir(parents=True, exist_ok=True)
    return _bank_pull_gdrive(blob_dir) if kind == "gdrive" \
        else _bank_pull_wandb(blob_dir)


def ensure_ram_cache(cfg) -> tuple[Path, Path]:
    """Return (train_blob, val_blob): local hit -> bank pull -> full build."""
    root = Path(cfg.jpeg_cache_dir)
    blob_dir = root / "ram256"
    train_blob = blob_dir / "train.pt"
    val_blob   = blob_dir / "validation.pt"

    if not (train_blob.exists() and val_blob.exists()):
        _bank_pull(blob_dir)

    if not (train_blob.exists() and val_blob.exists()):
        # jpeg cache is the build input — create it first if needed
        # (lazy import: dataset.py pulls DALI at module level)
        from dataset import _ensure_cache
        _ensure_cache(cfg)
        if not train_blob.exists():
            _build_split_blob(root / "train", train_blob)
        if not val_blob.exists():
            _build_split_blob(root / "validation", val_blob)
    return train_blob, val_blob


# ---------------------------------------------------------------------------
# GPU augmentation (batched; all shapes static)
# ---------------------------------------------------------------------------

def _rrc_flip_grid(B: int, device) -> torch.Tensor:
    """Fused RandomResizedCrop + horizontal flip as one affine grid.

    DALI-matched sampling: area U[0.08, 1], aspect U[3/4, 4/3], both of
    the (cached 256^2) source; boxes clamped to the source instead of
    DALI's 100-attempt rejection loop (differs only at extreme aspects).
    """
    area  = torch.empty(B, device=device).uniform_(0.08, 1.0)
    ratio = torch.empty(B, device=device).uniform_(0.75, 4.0 / 3.0)
    w = (area * ratio).sqrt().clamp(max=1.0)      # fractions of SIDE
    h = (area / ratio).sqrt().clamp(max=1.0)
    x0 = torch.rand(B, device=device) * (1 - w)   # top-left, fraction
    y0 = torch.rand(B, device=device) * (1 - h)
    flip = torch.where(torch.rand(B, device=device) < 0.5,
                       -torch.ones(B, device=device),
                       torch.ones(B, device=device))

    theta = torch.zeros(B, 2, 3, device=device)
    theta[:, 0, 0] = w * flip
    theta[:, 0, 2] = 2 * x0 + w - 1
    theta[:, 1, 1] = h
    theta[:, 1, 2] = 2 * y0 + h - 1
    return F.affine_grid(theta, (B, 3, CROP, CROP), align_corners=False)


# YIQ hue rotation (batched): RGB -> YIQ, rotate IQ by theta, back.
_RGB2YIQ = torch.tensor([[0.299, 0.587, 0.114],
                         [0.596, -0.274, -0.322],
                         [0.211, -0.523, 0.312]])
_YIQ2RGB = torch.tensor([[1.0, 0.956, 0.621],
                         [1.0, -0.272, -0.647],
                         [1.0, -1.106, 1.703]])


def _color_jitter(x: torch.Tensor) -> torch.Tensor:
    """x: (B,3,H,W) float in [0,255]. Brightness/contrast/saturation/hue,
    ranges matching dataset.py's DALI ops."""
    B, device = x.shape[0], x.device
    u = lambda lo, hi: torch.empty(B, 1, 1, 1, device=device).uniform_(lo, hi)

    x = x * u(0.6, 1.4)                                        # brightness
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    x = (x - mean) * u(0.6, 1.4) + mean                        # contrast
    gray = (x * torch.tensor([0.299, 0.587, 0.114], device=device)
            .view(1, 3, 1, 1)).sum(1, keepdim=True)
    x = torch.lerp(gray.expand_as(x), x, u(0.6, 1.4))          # saturation

    theta = torch.empty(B, device=device).uniform_(-36.0, 36.0) \
        * (math.pi / 180.0)                                    # hue
    c, s = torch.cos(theta), torch.sin(theta)
    rot = torch.zeros(B, 3, 3, device=device)
    rot[:, 0, 0] = 1
    rot[:, 1, 1] = c
    rot[:, 1, 2] = -s
    rot[:, 2, 1] = s
    rot[:, 2, 2] = c
    m = (_YIQ2RGB.to(device) @ rot @ _RGB2YIQ.to(device))      # (B,3,3)
    x = torch.einsum("bij,bjhw->bihw", m, x)

    return x.clamp_(0, 255)


def _normalize(x: torch.Tensor) -> torch.Tensor:
    """(B,3,H,W) float [0,255] -> normalized float32."""
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1) * 255
    std  = torch.tensor(IMAGENET_STD,  device=x.device).view(1, 3, 1, 1) * 255
    return (x - mean) / std


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

class RAMLoader:
    """
    Iterates (images, labels) with images already augmented + normalized on
    GPU — same contract as DALILoader. A background thread overlaps the
    gather + H2D copy of the NEXT batch (side CUDA stream) with the model's
    compute on the current one; augmentation runs on the main stream.
    """

    def __init__(self, images: torch.Tensor, labels: torch.Tensor,
                 batch_size: int, train: bool, device: torch.device):
        self.images = images          # uint8 (N,3,256,256), cpu or cuda
        self.labels = labels
        self.batch_size = batch_size
        self.train = train
        self.device = device
        n = images.shape[0]
        self._n_batches = n // batch_size if train \
            else (n + batch_size - 1) // batch_size

    def __len__(self):
        return self._n_batches

    def _batch_indices(self):
        n = self.images.shape[0]
        order = torch.randperm(n) if self.train else torch.arange(n)
        for b in range(self._n_batches):
            yield order[b * self.batch_size:(b + 1) * self.batch_size]

    def _produce(self, q: queue.Queue):
        stream = torch.cuda.Stream(self.device) \
            if self.device.type == "cuda" and self.images.device.type == "cpu" \
            else None
        try:
            for idx in self._batch_indices():
                if self.images.device.type == "cpu":
                    raw = self.images[idx]                     # CPU gather
                    lab = self.labels[idx]
                    if stream is not None:
                        with torch.cuda.stream(stream):
                            raw = raw.to(self.device, non_blocking=False)
                            lab = lab.to(self.device)
                            ev = torch.cuda.Event()
                            ev.record(stream)
                    else:
                        raw = raw.to(self.device)
                        lab = lab.to(self.device)
                        ev = None
                else:                                          # VRAM-resident
                    dev_idx = idx.to(self.images.device)
                    raw = self.images.index_select(0, dev_idx)
                    lab = self.labels.index_select(0, dev_idx)
                    ev = None
                q.put((raw, lab, ev))
        finally:
            q.put(None)

    def __iter__(self):
        q: queue.Queue = queue.Queue(maxsize=2)
        t = threading.Thread(target=self._produce, args=(q,), daemon=True)
        t.start()
        while True:
            item = q.get()
            if item is None:
                break
            raw, lab, ev = item
            if ev is not None:
                torch.cuda.current_stream(self.device).wait_event(ev)
            x = raw.float()
            if self.train:
                grid = _rrc_flip_grid(x.shape[0], x.device)
                x = F.grid_sample(x, grid, mode="bilinear",
                                  padding_mode="zeros", align_corners=False)
                x = _color_jitter(x)
            else:
                off = (SIDE - CROP) // 2
                x = x[:, :, off:off + CROP, off:off + CROP]
            yield _normalize(x), lab.long()
        t.join()


def get_ram_dataloaders(cfg) -> tuple[RAMLoader, RAMLoader]:
    train_blob, val_blob = ensure_ram_cache(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_device = torch.device(getattr(cfg, "data_device", "cpu"))

    loaders = []
    for blob, train in ((train_blob, True), (val_blob, False)):
        d = torch.load(blob, map_location="cpu", mmap=True)
        images, labels = d["images"], d["labels"]
        if data_device.type == "cuda":
            images = images.to(device)            # whole split into VRAM
            labels = labels.to(device)
        else:
            images = images.contiguous()          # materialize from mmap
        loaders.append(RAMLoader(images, labels, cfg.batch_size,
                                 train=train, device=device))
        print(f"[ram256] {blob.name}: {images.shape[0]:,} images on "
              f"{images.device} ({images.numel() / 1e9:.1f} GB)")
    return loaders[0], loaders[1]
