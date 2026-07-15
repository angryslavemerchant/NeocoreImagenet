# Vast.ai training automation

Rent a GPU, health-check it, train the ASFNet autoencoder, run the eval
visualisations, push everything to wandb, and destroy the instance — all
driven from `vast/launch.py` on the local machine.

## One-time setup (local)

1. `pip install vastai` (already done on this machine — see CLAUDE.md for paths)
2. Create `vast/secrets.env` (gitignored) with:
   ```
   VAST_API_KEY=...
   WANDB_API_KEY=...
   HF_TOKEN=...
   ```

## Commands

```bash
python vast/launch.py search                 # list candidate RTX 4090 offers
python vast/launch.py scan --n 3             # bench 3 machines, suggest thresholds
python vast/launch.py launch --smoke         # 1-epoch pipeline test (keep-alive)
python vast/launch.py launch                 # real run: 300 epochs, artifacts every 25
python vast/launch.py status                 # live instances
python vast/launch.py logs [--id ID]         # container logs
python vast/launch.py ssh  [--id ID]         # ssh command for the instance
python vast/launch.py destroy [--id ID|--all]
```

## Lifecycle of a `launch`

1. `launch.py` picks the cheapest offer passing filters (GPU, price, internet
   speed, reliability, disk) and creates the instance with a
   `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel` image.
2. The onstart command clones this repo (branch from `--branch`) and runs
   `vast/onstart.sh`, which:
   - installs deps,
   - runs `vast/benchmark.py --gate vast/thresholds.json` — a sick machine
     (slow download / disk / CPU / PCIe / GPU) **destroys itself** so you can
     immediately launch on another,
   - starts `vast/run_training.sh` inside tmux.
3. `run_training.sh` trains (`train_asfnet_ae.py`, args from `--train-args`),
   then runs `evaluate_asfnet_br.py --ae` for reconstruction/retention panels,
   uploads panels + `best.pt`/`latest.pt` + benchmark JSON to the wandb run
   (`vast/upload_results.py`), and **destroys the instance** on success.
   On failure the instance stays alive for inspection.
   `--keep-alive` disables auto-destroy.

## Monitoring

- wandb project `asfnet` — losses live, eval panels at the end, checkpoints
  under Artifacts (`asfnet-ae-<run_id>`). Download a checkpoint locally with
  `wandb artifact get <entity>/asfnet/asfnet-ae-<run_id>:final`.
- `python vast/launch.py logs` — provisioning + gate + training stdout.
  Markers: `ONSTART_BEGIN`, `BENCHMARK_JSON {...}`, `GATE_PASSED`/`GATE_FAILED`,
  `TRAIN_LAUNCHED`, `TRAIN_EXIT`, `RUN_COMPLETE`, `SELF_DESTROY`.

## Health gate tuning

`vast/thresholds.json` holds minimums checked at every boot. Run
`launch.py scan --n 3` occasionally: it rents N distinct machines in
bench-only mode (a few cents each — they self-destruct after ~10 min),
writes `vast/scan_results.json`, and prints suggested thresholds
(70% of the median per metric). Update thresholds.json and push — instances
read it from the cloned repo.

## Notes

- The repo is public, so the instance clones it without keys. Secrets travel
  only as container env vars, never through git.
- Anything committed here lands on public GitHub — keep keys out.
- Instance state is tracked in `.vast/instances.json` (gitignored); the
  ground truth is always `launch.py status`, which queries the Vast API.
