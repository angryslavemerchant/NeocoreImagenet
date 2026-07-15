# Offer selection judgment guide

For the agent operating `vast/launch.py`: do NOT blindly auto-pick. Run
`launch.py search`, look at the full table, choose an offer with the criteria
below, then `launch.py launch --offer <ID>`.

## User briefing (2026-07-14) — authoritative

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
7. **Image (user-specified, verbatim): `vastai/pytorch:cuda-13.2.1-auto`.**
   The Vast.ai pytorch template from DockerHub — its description text is
   outdated boilerplate (mentions CUDA 10 / pytorch 1.0), ignore that; the
   image itself is auto-updated (CUDA 13.2, Blackwell-capable, refreshed
   ~monthly). Do not substitute other tags without asking the user.

## Standing criteria (from operational experience)

- **Never the cheapest offers in the list** — the bottom of the price range
  over-samples broken/flaky hosts (3 consecutive bottom-price hosts failed
  to boot on day one). Shop the middle of the band.
- **On retry after a failure, always switch physical machine** (`m<id>` in
  search output) — same machine = same problem.
- **PCIe**: gen4 x8+ preferred; gen3 x8 acceptable (DALI overlaps H2D with
  compute); below that, skip.
- **Internet down**: ≥500 Mbps required (dataset ~13 GB); ≥1 Gbps preferred.

## Known-bad machines (add as found; also `.vast/blacklist.json`)

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
