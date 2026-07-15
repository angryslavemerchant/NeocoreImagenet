# Offer selection judgment guide

For the agent operating `vast/launch.py`: do NOT blindly auto-pick. Run
`launch.py search`, look at the full table, choose an offer with the criteria
below, then `launch.py launch --offer <ID>`.

> STATUS: draft — seeded from the first day's experience (2026-07-14).
> The user will brief further judgment rules; record them here.

## Criteria (in rough priority order)

1. **Price position**: never the cheapest offers in the list — the bottom of
   the range over-samples broken/flaky hosts (empirically: 3 consecutive
   bottom-price hosts failed to boot). Aim near the middle of the price
   spread for the GPU class.
2. **CPU type**: prefer consumer processors (AMD Ryzen, Intel Core i7/i9) —
   generally much faster per core than server CPUs (EPYC, Xeon), which
   matters for dataloading. The `search` output shows `[cpu_name]` per offer.
3. **Known-bad machines** (add here as found; also `.vast/blacklist.json`):
   - machine 14825 — docker daemon broken (OCI runtime create failed), 2026-07-14
   - machine 9020  — instances wedge in created/stopped, never boot; ignores
     explicit start, 2026-07-14
4. **PCIe**: gen4 x8 or better preferred; gen3 x8 (~6.5 GB/s) is acceptable
   for this workload (DALI overlaps H2D with compute). Below that, skip.
5. **Internet down**: ≥500 Mbps required (dataset ~13 GB); ≥1 Gbps preferred.
6. **Reliability**: the search filter floors at 0.98; prefer higher.
7. **On retry after a failure, always switch physical machine** (`m<id>` in
   the search output) — same machine = same problem.

## Post-boot judgment

The boot health gate (benchmark vs thresholds.json) handles measurable
sickness automatically. If an instance sits in `created`/null status >10 min
with no logs: check `vastai show instance <id> --raw` → `status_msg`;
destroy, add machine to the list above, pick a different machine.

## User briefing notes

(to be filled in as the user provides judgment guidance)
