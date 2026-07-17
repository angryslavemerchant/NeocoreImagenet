"""
vast/upload_blobs.py — publish the ram256 blobs as the dataset-bank
artifact. Run ONCE on an instance that has built (or pulled) the blobs:

    python vast/upload_blobs.py                       # defaults
    python vast/upload_blobs.py --blob_dir ./jpeg_cache/ram256

The artifact is the bank unit dataset_ram._bank_pull_wandb() fetches at
boot (ref: <entity>/neocore/imagenet100-ram256:latest). Upload is
VERIFIED committed (art.wait()) before this script reports success —
the 2026-07-15 wandb-outage lesson.
"""

import os
import argparse

import wandb


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--blob_dir", type=str, default="./jpeg_cache/ram256")
    p.add_argument("--project",  type=str, default="neocore")
    p.add_argument("--name",     type=str, default="imagenet100-ram256")
    args = p.parse_args()

    files = [os.path.join(args.blob_dir, f)
             for f in ("train.pt", "validation.pt")]
    for f in files:
        if not os.path.exists(f):
            raise SystemExit(f"missing blob: {f}")
        print(f"  {f}: {os.path.getsize(f) / 2**30:.2f} GiB")

    run = wandb.init(project=args.project, name="blob-upload",
                     job_type="dataset-upload")
    art = wandb.Artifact(
        args.name, type="dataset",
        metadata={"side": 256, "layout": "uint8 CHW torch blob",
                  "splits": ["train", "validation"],
                  "source": "clane9/imagenet-100 via dataset_ram build"})
    for f in files:
        art.add_file(f)
    run.log_artifact(art)
    art.wait()   # verified commit
    print(f"BANK_UPLOAD_OK {art.qualified_name}")
    wandb.finish()


if __name__ == "__main__":
    main()
