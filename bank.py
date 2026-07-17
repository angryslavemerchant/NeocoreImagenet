"""
bank.py — Google Drive dataset bank.

The bank unit is the JPEG cache as ONE tar (~2.2 GB: train/ + validation/
including metadata.json and .done flags) at gdrive:NeocoreBank/jpeg_cache.tar.
The 25 GB ram256 tensor blob is NOT banked — it rebuilds deterministically
from the jpegs on-GPU in minutes (dataset_ram) and persists across restarts
via the /workspace symlink. clane9/imagenet-100 ships shorter-side-160
images, which is why the whole information content is 2.2 GB.

Auth: an rclone OAuth token for the DEDICATED storage Google account (never
the personal one), in RCLONE_DRIVE_TOKEN (raw JSON, secrets.env) or
RCLONE_DRIVE_TOKEN_B64 (base64 — how launch.py ships it through the
docker --env string, which would mangle raw JSON). No token → bank
disabled, callers fall back to the HF build.

Boot flow (dataset._ensure_cache): local cache → bank pull → HF build.
Publish once from wherever a cache exists: python vast/upload_bank.py
"""

import base64
import os
import shutil
import subprocess
import tarfile
from pathlib import Path

BANK_TAR    = "jpeg_cache.tar"
BANK_REMOTE = os.environ.get("RAM_BANK_DRIVE_PATH", "gdrive:NeocoreBank")


def _token():
    b64 = os.environ.get("RCLONE_DRIVE_TOKEN_B64")
    if b64:
        return base64.b64decode(b64).decode()
    return os.environ.get("RCLONE_DRIVE_TOKEN")


def _rclone_ready() -> bool:
    """Token present + rclone installed + remote configured."""
    tok = _token()
    if not tok:
        print("[bank] no RCLONE_DRIVE_TOKEN(_B64) — bank disabled")
        return False
    if shutil.which("rclone") is None:
        subprocess.run("curl -fsSL https://rclone.org/install.sh | bash",
                       shell=True, check=False)
        if shutil.which("rclone") is None:
            print("[bank] rclone install failed")
            return False
    conf = Path.home() / ".config" / "rclone" / "rclone.conf"
    conf.parent.mkdir(parents=True, exist_ok=True)
    conf.write_text(f"[gdrive]\ntype = drive\nscope = drive\ntoken = {tok}\n")
    return True


_RCLONE_RETRIES = ["--retries", "8", "--low-level-retries", "20"]


def pull_jpeg_cache(root: Path) -> bool:
    """Fetch + extract the bank tar into root (-> root/train, root/validation).
    False (never raises) when the bank is unconfigured/unpublished — the
    caller's HF build path handles it."""
    if not _rclone_ready():
        return False
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    tar_path = root / BANK_TAR
    r = subprocess.run(["rclone", "copy", f"{BANK_REMOTE}/{BANK_TAR}",
                        str(root), *_RCLONE_RETRIES], check=False)
    if r.returncode != 0 or not tar_path.exists():
        print("[bank] pull failed or tar not published — will build locally")
        return False
    print(f"[bank] extracting {tar_path.stat().st_size / 2**30:.2f} GiB ...")
    with tarfile.open(tar_path) as tf:
        tf.extractall(root)
    tar_path.unlink()
    ok = (root / "train").is_dir() and (root / "validation").is_dir()
    print(f"[bank] jpeg cache pulled -> {root} (ok={ok})")
    return ok


def upload_jpeg_cache(root: Path) -> None:
    """Tar root/{train,validation} and push to the bank, size-verified."""
    if not _rclone_ready():
        raise SystemExit("bank not configured (RCLONE_DRIVE_TOKEN missing?)")
    root = Path(root)
    tar_path = root / BANK_TAR
    print(f"[bank] tarring {root} ...")
    with tarfile.open(tar_path, "w") as tf:
        tf.add(root / "train", arcname="train")
        tf.add(root / "validation", arcname="validation")
    size = tar_path.stat().st_size
    print(f"[bank] uploading {size / 2**30:.2f} GiB to {BANK_REMOTE} ...")
    # 256M chunks: Drive uploads with rclone's default 8 MiB chunks crawl
    # (measured 2026-07-17: 2.35 GiB not done after 15 min on a 32 MiB/s
    # uplink); large chunks fix it. Periodic one-line stats keep the tee'd
    # log honest about progress.
    subprocess.run(["rclone", "copy", str(tar_path), BANK_REMOTE,
                    "--drive-chunk-size", "256M",
                    "--stats", "15s", "--stats-one-line", "-v",
                    *_RCLONE_RETRIES], check=True)
    out = subprocess.run(["rclone", "lsl", f"{BANK_REMOTE}/{BANK_TAR}"],
                         capture_output=True, text=True, check=True)
    remote_size = int(out.stdout.split()[0])
    if remote_size != size:
        raise SystemExit(f"BANK_UPLOAD_MISMATCH remote={remote_size} local={size}")
    tar_path.unlink()
    print(f"BANK_UPLOAD_OK {BANK_REMOTE}/{BANK_TAR} bytes={size}")
