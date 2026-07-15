"""
vast/upload_results.py — attach eval visualisations and final checkpoints to
the training run on wandb.

Resumes the run identified by WANDB_RUN_ID (exported by run_training.sh so
training and this step share one run), logs every PNG in --viz_dir as an
image panel, and uploads best.pt / latest.pt (+ any --extra files, e.g. the
benchmark JSON) as a model artifact. Download later with:

    wandb artifact get <entity>/<project>/asfnet-ae-<run_id>:latest
"""

import argparse
import glob
import os

import wandb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--viz_dir",  type=str, default="viz_ae")
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints_asfnet_ae")
    parser.add_argument("--extra",    type=str, nargs="*", default=[])
    args = parser.parse_args()

    run = wandb.init(
        project=os.environ.get("WANDB_PROJECT", "asfnet"),
        id=os.environ.get("WANDB_RUN_ID"),
        resume="allow",
    )

    pngs = sorted(glob.glob(os.path.join(args.viz_dir, "*.png")))
    if pngs:
        run.log({f"eval/{os.path.basename(p)}": wandb.Image(p) for p in pngs})
        print(f"Logged {len(pngs)} eval image(s)")

    artifact = wandb.Artifact(f"asfnet-ae-{run.id}", type="model")
    added = False
    for name in ("best.pt", "latest.pt"):
        path = os.path.join(args.ckpt_dir, name)
        if os.path.exists(path):
            artifact.add_file(path)
            added = True
    for path in args.extra:
        if os.path.exists(path):
            artifact.add_file(path)
    if added:
        run.log_artifact(artifact, aliases=["final"])
        print("Checkpoint artifact uploaded")

    run.finish()


if __name__ == "__main__":
    main()
