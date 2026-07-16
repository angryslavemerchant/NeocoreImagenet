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

## Research checkpoint (2026-07-16) — probes, slots, and the two-stage AEs

All checkpoints + eval panels now live in local `runs/<run_name>/`
(wandb = metrics only). Loss conventions: "dropped-only" vs "all-position"
rec numbers are NOT comparable across conventions.

1. **Probe grid on frozen AE_budget25** (180 epochs each): attentive+ALL
   survivors **32.0%** top-1 / 59.3% top-5; attentive+top49 26.5%;
   mean+top49 12.1% (original mean+all 50ep: 8.3%). Lessons: the attentive
   (MAP) head is the dominant factor (+14pt); restricting to the 49
   reconstruction-graded tokens HURTS (-5.5pt) — the other ~140 survivors
   carry real class signal; the old 8.3% was a measurement artifact.
2. **Single-stage slot bottleneck (AE_xattn49) works**: 49 learned queries
   cross-attend over survivors, Perceiver-IO decode, loss on ALL positions.
   val rec 0.211 (original) / 0.227 (resume from ep174 after the 2026-07-15
   wandb storage outage ate the original's final weights; resume weights
   are in runs/AE_xattn49_resume/). Router stayed non-degenerate (~26% drop).
3. **Two-stage AE head-to-head (all 300 ep, enc 3+3, main 6) — hierarchy
   is currently an optimization problem, not a capacity one:**
   - pool (AE2): router2 group count THRASHED 1.7↔91 all run; rec 0.62
     all-position. Hypothesis: group count is a percolation quantity and
     tgs=3 sits near criticality; two stacked near-critical routers drive
     each other.
   - double retention (AE2BR): keep-all basin, only 12.5% dropped,
     rec 0.116 dropped-only = trivial. Same corner as the single-stage
     baseline collapse.
   - double retention + 49 slots: total collapse — routers thrashed, slots
     output a constant, rec pinned at 1.00 all-position from epoch 26.
   - double retention + budget 49 (rank by s1+s2): the ONLY stable
     two-stage — drop pinned 0.75, kept2=49, smooth descent — but final
     rec 0.218 dropped-only vs single-stage budget25's 0.101 at the same
     rate/convention. **Hard budgets stabilize two-stage training; the
     hierarchy still costs ~2x reconstruction.**
4. Candidate next: per-stage rate ladder (e.g. stage1 budget 98 → stage2
   49); warm-start stage 1 + encoder from AE_budget25 and let stage 2 learn
   on a stationary substrate; dense 49-slot control with no routing (the
   "price of interpretability" baseline); distillation/JEPA target instead
   of pixels (probe evidence says pixel targets underorganize semantics).

Cloud lessons (2026-07-15/16): wandb GCS 403 outage (~1 h) and an HF 502
each killed runs at boot — artifact/dataset I/O now retries with backoff;
successful runs AWAIT PULL (`launch.py pull`, scp + account ssh key)
instead of self-destroying; known-bad machines list grew (m48680 GPU hang,
m140634 zombie boot).

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
- **Run persistence (settled 2026-07-16):** runs write checkpoints + eval
  PNGs to `runs/<run_name>/` on the instance (`runs/LATEST` points at the
  newest; `runs/` is gitignored locally too). On success the final
  checkpoint artifact is uploaded to wandb and **VERIFIED committed**
  (`art.wait()`), and only then does the instance self-destroy. If the
  verified upload fails (e.g. a wandb storage outage — 2026-07-15 one ate a
  run's weights under the old async upload), the instance stays alive
  printing `AWAITING_PULL`; fetch with `launch.py pull --id <ID>` (scp;
  account ssh key), then `destroy`.
- The instance clones this repo from GitHub, so cloud-side changes
  (onstart.sh, thresholds, train args defaults) only take effect after push.
