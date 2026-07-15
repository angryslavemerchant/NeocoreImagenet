# NeocoreImagenet

ASFNet experiments on ImageNet-100 (`clane9/imagenet-100`, auto-downloaded
from HuggingFace on first run; DALI dataloading via a one-time JPEG cache).

- `train_asfnet*.py` / `model_asfnet*.py` — classification variants
  (`_br` = border-retention, `2` = two-stage).
- `*_ae` — self-supervised MAE-style autoencoder on the ASFNetBR backbone.
  Current research direction. Eval/visualisation: `evaluate_asfnet_br.py --ae`
  (reconstruction + retention panels); non-AE modes produce chunk-map grids.
- `train_linear_probe.py` — frozen-backbone linear probe of an AE checkpoint
  on ImageNet-100 (loads the checkpoint from a wandb artifact).
- Training logs to wandb: project `asfnetAE` for AE runs, `asfnet` for probes.

## Research checkpoint (2026-07-15) — where the AE work stands

Full history in the wandb projects; the state of play:

1. **Baseline AE collapses to keep-everything.** 300-epoch run `cgyu2m9e`:
   final drop_frac 0.013, mean_kept 193.5/196, val rec 0.379 (on the ~1% it
   dropped). Mechanism: the edge-level ratio loss is *satisfied* at ~64%
   cuts (its value 1.0 IS the normalised minimum) in any spatial
   arrangement, and at target_group_size 3 chunks have no interiors, so
   border-retention keeps ~all tokens and reconstruction is trivial. There
   is no pressure toward compression in that objective.
2. **Variant A — hard bottleneck (`--keep_budget 0.25`) WORKS.** Run
   `wx8mlobe` (`AE_budget25`): top-49-by-evidence survivors enter the
   decoder, rest masked + counted as dropped. drop_frac pinned 0.75, val
   rec 0.101, legible reconstructions (user-confirmed). This is the
   working instrument going forward.
3. **Variant B — soft token keep-rate loss FAILED TWICE**, each time by
   finding a degenerate solution (both were exploits the user predicted):
   `s3vuij4s`: collapse all edge probs to 0 → zero-border guard keeps all →
   the loss's (F=1, G=0) corner is exactly 0 (fixed by thresholding soft
   p_keep for F). `0ie7me7e` (v2): park all edge probs in ~[0.16, 0.5) so
   accumulated soft evidence hits the 25% target while NO edge crosses the
   0.5 hard-cut threshold → zero cuts → guard keeps all, rec loss 0, stable
   fixed point (killed at epoch 15). **Lesson: any soft pressure + hard
   drop threshold + loss-only-on-dropped lets SGD park just under the
   threshold.** A third attempt would need retention itself redefined on
   accumulated evidence (keep ⇔ p_keep > 0.5) — an architecture change,
   user's call, not yet made.
4. **Linear probe of A** (`probe_AE_budget25`, project `asfnet`): top-1
   8.3% / top-5 23.5%. Known caveats: probe mean-pools ALL ~190
   router-kept tokens but only the top-49 got reconstruction gradient
   (pooling mismatch), and MAE-style features probe poorly in general.
   Candidate follow-ups: pool only top-49-evidence tokens, attentive
   probe, or full fine-tune (the honest ceiling).

Agreed next directions (not started): budget annealing for A (e.g. 0.9 →
0.25), rate-distortion learnable K, fine-tune script (small variant of
train_linear_probe.py with everything unfrozen + lower LR).

## Local environment (Windows)

- No `python` on PATH. The project env is the `ToastEnv` conda env:
  `& "C:\Users\JmgLi\anaconda3\envs\ToastEnv\python.exe"` and
  `& "C:\Users\JmgLi\anaconda3\envs\ToastEnv\Scripts\vastai.exe"`.
- Training does NOT run locally — it runs on rented Vast.ai GPUs.

## Vast.ai cloud training (see vast/README.md for full runbook)

Everything is driven by `vast/launch.py`. **Offer selection is a judgment
call by the operating agent**: run `search`, apply `vast/OFFER_JUDGEMENT.md`
(price near the middle of the range, consumer CPUs, known-bad machine list),
then `launch --offer <ID>`. Auto-pick only as a fallback.

```powershell
& "C:\Users\JmgLi\anaconda3\envs\ToastEnv\python.exe" vast\launch.py status
& "C:\Users\JmgLi\anaconda3\envs\ToastEnv\python.exe" vast\launch.py launch --smoke
& "C:\Users\JmgLi\anaconda3\envs\ToastEnv\python.exe" vast\launch.py launch   # 300-epoch AE run
& "C:\Users\JmgLi\anaconda3\envs\ToastEnv\python.exe" vast\launch.py logs
& "C:\Users\JmgLi\anaconda3\envs\ToastEnv\python.exe" vast\launch.py destroy  # "kill it"
```

- Secrets live in `vast/secrets.env` (gitignored — NEVER commit; the repo is
  public). Instance state in `.vast/instances.json`.
- Instances health-gate themselves at boot (`vast/benchmark.py` vs
  `vast/thresholds.json`) and self-destroy when unhealthy. Failed runs keep
  the instance alive for inspection.
- **Local-first persistence (since the 2026-07-15 wandb storage outage):**
  each run writes checkpoints + eval PNGs to `runs/<run_name>/` on the
  instance (gitignored; `runs/LATEST` points at the newest). wandb is
  metrics logging only; artifact uploads are opportunistic backups.
  Successful runs do NOT self-destroy anymore — they print `AWAITING_PULL`;
  fetch results with `launch.py pull --id <ID>`, verify, then `destroy`.
- The instance clones this repo from GitHub, so cloud-side changes
  (onstart.sh, thresholds, train args defaults) only take effect after push.
