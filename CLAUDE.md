# NeocoreImagenet

ASFNet experiments on ImageNet-100 (`clane9/imagenet-100`, auto-downloaded
from HuggingFace on first run; DALI dataloading via a one-time JPEG cache).

- `train_asfnet*.py` / `model_asfnet*.py` — classification variants
  (`_br` = border-retention, `2` = two-stage).
- `*_ae` — self-supervised MAE-style autoencoder on the ASFNetBR backbone.
  Current research direction. Eval/visualisation: `evaluate_asfnet_br.py --ae`
  (reconstruction + retention panels); non-AE modes produce chunk-map grids.
- Training logs to wandb (project `asfnetAE` for AE runs).

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
  `vast/thresholds.json`) and self-destroy when unhealthy or when a run
  completes successfully. Failed runs keep the instance alive for inspection.
- The instance clones this repo from GitHub, so cloud-side changes
  (onstart.sh, thresholds, train args defaults) only take effect after push.
