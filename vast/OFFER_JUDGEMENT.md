# Offer selection judgment guide

For the agent operating `vast/launch.py`: do NOT blindly auto-pick. Run
`launch.py search`, look at the full table, choose an offer with the criteria
below, then `launch.py launch --offer <ID>`.

## User briefing (2026-07-17) — authoritative, supersedes GPU-class rules below

1. **Default GPU class: single RTX 5090 (`RTX_5090`, num_gpus=1).** The
   user cut the burn rate: 96 GB cards were overkill for these ~6 M-param
   models. RTX PRO 6000 WS / B200 only when the user explicitly asks for a
   fast run. Market at briefing time: 45 offers, floor $0.296/hr, median
   $0.497/hr. Default cap $0.60/hr.
2. **Training recipe on a 5090 (32 GB)** — launch.py's default train-args:
   batch 256, lr 7.5e-4 (linearly rescaled from the 1024/3e-3 recipe),
   `--data ram` with the blob in **system RAM** (the 25 GB train blob does
   not fit in 32 GB VRAM — never pass `--data_device cuda` on this class),
   `--compile_mode default`, `--checkpoint_rounds 3` (loop archs; ~15 GB
   for R7 at batch 256). Expect ~2x the PRO-6000 epoch time (~160 s for an
   R7-class run) at roughly half the hourly cost.
3. **CPU rules relax on this class**: the 5090 market is nearly all EPYC
   hosts, and with the RAM-blob loader (GPU-side aug, torch-only) training
   is GPU-bound — EPYC is fine now. Still avoid the genuine toasters
   (6-core Ryzen 3600 / 7500F class). **cpu_ram ≥ 48 GB is a hard
   requirement** (25 GB blob lives in system RAM).

## User briefing (2026-07-14) — GPU-class rules superseded above

1. **GPU classes: RTX 6000 Blackwell (`RTX_PRO_6000_WS`) only**, plus the
   occasional **B200** rapid run when the user asks for one.
   (`RTX_6000Ada` is the older Ada card — NOT what is meant.)
2. **Hard price cap: $1.20/hr.** Market note at briefing time: RTX PRO 6000
   WS floors around $0.81/hr (22 offers); B200 floors at ~$6.25/hr, i.e.
   above the cap — a B200 run requires the user explicitly approving a
   temporary cap override.
3. **Reliability score: ignore it.** (Empirically 0.996 hosts still dropped
   contracts; the score doesn't predict the failures that matter.)
4. **CPU preference dominates everything, including location**: consumer
   CPUs (Ryzen, Intel Core) over server CPUs (EPYC, Xeon, Threadripper) —
   unless the consumer chip is an absolute toaster (ancient/low-end, e.g. an
   old 6-core Ryzen 3600-class part). Rule of thumb from the user: an 8-core
   modern consumer CPU beats a 32-core Threadripper more often than not.
5. **Location**: prefer North America over elsewhere; prefer Canada over the
   US. Subordinate to the CPU preference.
6. **Container/disk size: 80 GB.**
7. **Template (user-specified): the official "PyTorch (Vast)" template**,
   template_id `2ad6d615db5927a06fef0c9cd51d77c4` — replicated in launch.py
   from the CLI command the Vast console generates: image
   `vastai/pytorch:@vastai-automatic-tag` plus the template's port/env
   config (portal, jupyter, tensorboard), launched `--jupyter --ssh
   --direct`, with the template's `entrypoint.sh` kept in the foreground
   and our provisioning chained in the background. Its DockerHub
   description text is outdated boilerplate (mentions CUDA 10) — ignore;
   the image is auto-updated and Blackwell-capable. Deviation from the
   template: disk 80 GB (not 16) per the user's briefing. Do not
   substitute other images without asking the user.

## Standing criteria (from operational experience)

- **Never the cheapest offers in the list** — the bottom of the price range
  over-samples broken/flaky hosts (3 consecutive bottom-price hosts failed
  to boot on day one). Shop the middle of the band.
- **On retry after a failure, always switch physical machine** (`m<id>` in
  search output) — same machine = same problem.
- **PCIe**: gen4 x8+ preferred; gen3 x8 acceptable (DALI overlaps H2D with
  compute); below that, skip.
- **Internet down**: ≥500 Mbps required (dataset ~13 GB); ≥1 Gbps preferred.

## Known-good machines

- machine 32649 — Alberta, Ryzen 9 9950X3D + RTX PRO 6000 WS, ~$1.10/hr.
  Ran the full smoke pipeline flawlessly 2026-07-15: boot→gate in ~1 min,
  best fleet benchmark yet (416 bf16 TFLOPS, 36.9 GB/s H2D, 4.2 GB/s disk,
  15.8k jpeg/s), full smoke boot→train→eval→upload in ~10 min. First choice
  when its offers are listed.
- machine 36615 — Alberta, Core Ultra 7 265K + RTX PRO 6000 WS. Benchmarked
  superbly (384 TFLOPS bf16, 12.8k jpeg/s). Was briefly suspected of
  dropping a contract on 2026-07-15 — that instance was actually STOPPED
  MANUALLY BY THE USER; the machine is fine. Good second choice.

## Machines with a caution flag (usable, watch them)

- machine 137683 — Ryzen 9 9950X3D, Germany. Excellent performer (402
  TFLOPS, ran two clean speed tests 2026-07-17), but ONCE spontaneously
  stopped a fresh container ~2 min after boot (intended_status=stopped,
  no error; not user-initiated). `vastai start instance <id>` revived it
  and the run completed normally. If it stops twice, blacklist it.

## Known-bad machines (add as found; also `.vast/blacklist.json`)

- machine 12092 — Core Ultra 9 285K, Nevada. Instance never booted (no
  ONSTART after ~7 min, no logs) and vanished from the API; user confirmed
  it never loaded, 2026-07-15.

- machine 48680 — i7-12700KF, Washington. Boots and trains fine, then GPU
  hangs mid-run: DALI decode hit CUDA_ERROR_LAUNCH_TIMEOUT (702) in epoch 4
  of a probe run, process aborted, 2026-07-15
- machine 140634 — Ryzen 7 9700X, South Korea. Zombie boot: API reports
  running/success but zero onstart logs for 3 h and ssh refused on the
  advertised port. (Same machine booted fine earlier that day —
  intermittent.) 2026-07-16
- machine 14825 — docker daemon broken (OCI runtime create failed), 2026-07-14
- machine 9020  — instances wedge in created/stopped, never boot; ignores
  explicit start, 2026-07-14
- machine 34887 — Ryzen 9 9950X 4090, looks great on paper, boots fast, but
  silently DROPS contracts: accepted twice, instance vanished within ~90s
  both times (no status, no logs, no billing), 2026-07-14

## Post-boot judgment

The boot health gate (benchmark vs thresholds.json) handles measurable
sickness automatically. If an instance sits in `created`/null status >10 min
with no logs: check `vastai show instance <id> --raw` → `status_msg`;
destroy, record the machine above, pick a different machine. If the instance
disappears from `show instances` entirely, the host dropped the contract —
same response. Note: current `thresholds.json` GPU floor (90 bf16 TFLOPS) is
a broken-hardware floor, not a Blackwell performance bar — rerun
`launch.py scan` on the Blackwell fleet to calibrate real thresholds.
