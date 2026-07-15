"""
vast/launch.py — local orchestrator for Vast.ai training runs.

Runs on the local machine (Windows: use the Anaconda python). Wraps the
`vastai` CLI; reads secrets from vast/secrets.env (gitignored). State about
live instances is kept in .vast/instances.json (gitignored).

Commands:
    search   [--gpu RTX_4090] [--max-dph 0.6]        list candidate offers
    launch   [--offer ID] [--train-args "..."]       rent + provision + train
    scan     [--n 3]                                 bench-only pass over N
                                                     instances, suggest gate
                                                     thresholds, self-destroy
    status                                           show live instances
    logs     [--id ID] [--tail 120]                  fetch instance logs
    ssh      [--id ID]                               print ssh command
    destroy  [--id ID | --all]                       kill instance(s)

Examples:
    python vast/launch.py scan --n 3
    python vast/launch.py launch --train-args "--num_epochs 300 --artifact_every 25"
    python vast/launch.py launch --smoke          # 1-epoch pipeline test, keep-alive
    python vast/launch.py destroy
"""

import argparse
import json
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT      = Path(__file__).resolve().parent.parent
SECRETS   = ROOT / "vast" / "secrets.env"
STATE     = ROOT / ".vast" / "instances.json"
BLACKLIST = ROOT / ".vast" / "blacklist.json"
SCAN_OUT  = ROOT / "vast" / "scan_results.json"

REPO_URL  = "https://github.com/angryslavemerchant/NeocoreImagenet.git"
# Vast's own template image — hosts pre-cache these, so boots take ~1 min
# instead of a 10-30 min cold DockerHub pull. cu128 build: required for
# Blackwell (sm_120 / sm_100); torch 2.9.1 = mature release, low API drift.
IMAGE     = "vastai/pytorch:2.9.1-cu128-cuda-12.9-mini-py312-2026-06-15"
DISK_GB   = 80

VASTAI = (shutil.which("vastai")
          or r"C:\Users\JmgLi\anaconda3\envs\ToastEnv\Scripts\vastai.exe")


# ---------------------------------------------------------------------------
# Plumbing
# ---------------------------------------------------------------------------

def load_secrets() -> dict:
    if not SECRETS.exists():
        sys.exit(f"Missing {SECRETS} — create it with VAST_API_KEY, "
                 "WANDB_API_KEY, HF_TOKEN lines.")
    out = {}
    for line in SECRETS.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def vast(*args, raw: bool = True, check: bool = True):
    """Run the vastai CLI; parse JSON when raw=True."""
    cmd = [VASTAI, *map(str, args)]
    if raw:
        cmd.append("--raw")
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=180)
    if check and proc.returncode != 0:
        raise RuntimeError(f"vastai {' '.join(map(str, args))} failed:\n"
                           f"{proc.stdout}\n{proc.stderr}")
    if raw:
        try:
            return json.loads(proc.stdout)
        except json.JSONDecodeError:
            return proc.stdout
    return proc.stdout


def load_state() -> list:
    if STATE.exists():
        return json.loads(STATE.read_text())
    return []


def save_state(records: list):
    STATE.parent.mkdir(exist_ok=True)
    STATE.write_text(json.dumps(records, indent=2))


def resolve_id(args) -> int:
    """--id if given, else the single live instance in local state."""
    if getattr(args, "id", None):
        return args.id
    live = load_state()
    if len(live) == 1:
        return live[0]["id"]
    if not live:
        sys.exit("No tracked instances — pass --id (see `status`).")
    ids = ", ".join(str(r["id"]) for r in live)
    sys.exit(f"Multiple tracked instances ({ids}) — pass --id.")


# ---------------------------------------------------------------------------
# Offers
# ---------------------------------------------------------------------------

def load_blacklist() -> set:
    if BLACKLIST.exists():
        return set(json.loads(BLACKLIST.read_text()))
    return set()


def add_to_blacklist(machine_id: int):
    bl = load_blacklist()
    bl.add(machine_id)
    BLACKLIST.parent.mkdir(exist_ok=True)
    BLACKLIST.write_text(json.dumps(sorted(bl)))


def search_offers(gpu: str, max_dph: float, inet: int = 500, limit: int = 40):
    # No reliability filter (user: doesn't matter). cuda>=12.8 for Blackwell.
    query = (f"gpu_name={gpu} num_gpus=1 rentable=true verified=true "
             f"inet_down>={inet} disk_space>={DISK_GB} "
             f"cpu_cores_effective>=8 cpu_ram>=32 "
             f"cuda_max_good>=12.8 dph<={max_dph}")
    offers = vast("search", "offers", query, "-o", "dph")
    if not isinstance(offers, list):
        return []
    bl = load_blacklist()
    return [o for o in offers if o.get("machine_id") not in bl][:limit]


def _is_server_cpu(o: dict) -> bool:
    name = (o.get("cpu_name") or "").lower()
    return "epyc" in name or "xeon" in name


def pick_offer(offers: list):
    """
    Not the cheapest — the bottom of the price range is where the lemons
    live (learned the hard way: 3 of the 4 cheapest hosts in a row failed
    to boot or run CUDA). Pick near the middle of the price distribution,
    preferring consumer CPUs (Ryzen/Core), which are generally faster per
    core than the EPYC/Xeon boxes for dataloading.
    """
    if not offers:
        return None
    median_dph = statistics.median(o["dph_total"] for o in offers)
    pool = [o for o in offers if not _is_server_cpu(o)] or offers
    return min(pool, key=lambda o: abs(o["dph_total"] - median_dph))


def fmt_offer(o: dict) -> str:
    return (f"  {o['id']:>10}  {o.get('gpu_name', '?'):<14} "
            f"${o.get('dph_total', 0):.3f}/hr  "
            f"{o.get('inet_down', 0):>5.0f} Mbps down  "
            f"{o.get('cpu_cores_effective', 0):>4.0f} cores  "
            f"{o.get('cpu_ram', 0) / 1024:>4.0f} GB RAM  "
            f"pcie x{o.get('pci_gen', '?')}g{o.get('gpu_lanes', '?')}  "
            f"rel {o.get('reliability2', 0):.3f}  "
            f"m{o.get('machine_id', '?')}  "
            f"[{o.get('cpu_name', '?')}]  "
            f"({o.get('geolocation', '?')})")


def cmd_search(args):
    offers = search_offers(args.gpu, args.max_dph, args.inet)
    if not offers:
        print("No offers matched — relax --max-dph or --inet.")
        return
    print(f"Top {len(offers)} offers for {args.gpu}:")
    for o in offers:
        print(fmt_offer(o))


# ---------------------------------------------------------------------------
# Instance creation
# ---------------------------------------------------------------------------

def build_onstart(branch: str, train_args: str, bench_only: bool,
                  keep_alive: bool) -> str:
    exports = [f"export TRAIN_ARGS='{train_args}'"]
    if bench_only:
        exports.append("export BENCH_ONLY=1")
    if keep_alive:
        exports.append("export KEEP_ALIVE=1")
    return (
        "cd /workspace && rm -rf NeocoreImagenet && "
        f"git clone -b {branch} {REPO_URL} && "
        "cd NeocoreImagenet && "
        + " && ".join(exports) + " && "
        "bash vast/onstart.sh 2>&1 | tee -a /workspace/onstart.log"
    )


def create_instance(offer_id: int, secrets: dict, branch: str,
                    train_args: str, bench_only: bool, keep_alive: bool,
                    purpose: str) -> int:
    env = (f"-e WANDB_API_KEY={secrets['WANDB_API_KEY']} "
           f"-e HF_TOKEN={secrets['HF_TOKEN']} "
           f"-e VAST_API_KEY={secrets['VAST_API_KEY']} "
           f"-e WANDB_PROJECT=asfnet")
    onstart = build_onstart(branch, train_args, bench_only, keep_alive)
    result = vast("create", "instance", offer_id,
                  "--image", IMAGE,
                  "--disk", DISK_GB,
                  "--env", env,
                  "--onstart-cmd", onstart,
                  "--ssh", "--direct")
    if not isinstance(result, dict) or not result.get("success"):
        raise RuntimeError(f"create instance failed: {result}")
    iid = result["new_contract"]

    records = load_state()
    records.append({
        "id": iid,
        "offer": offer_id,
        "purpose": purpose,
        "branch": branch,
        "train_args": train_args,
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    save_state(records)
    return iid


def cmd_launch(args):
    secrets = load_secrets()
    if args.smoke:
        args.train_args = ("--num_epochs 1 --batch_size 256 "
                           "--run_name smoke_test")
        args.keep_alive = True

    offer_id = args.offer
    if offer_id is None:
        offers = search_offers(args.gpu, args.max_dph, args.inet)
        offer = pick_offer(offers)
        if offer is None:
            sys.exit("No offers matched the filters.")
        offer_id = offer["id"]
        print("Selected offer (median-price, consumer-CPU preferred):")
        print(fmt_offer(offer))

    iid = create_instance(offer_id, secrets, args.branch, args.train_args,
                          bench_only=False, keep_alive=args.keep_alive,
                          purpose="smoke" if args.smoke else "train")
    print(f"\nInstance {iid} created.")
    print(f"  watch:   python vast/launch.py logs --id {iid}")
    print(f"  destroy: python vast/launch.py destroy --id {iid}")
    print("  wandb:   project 'asfnet' — run appears once training starts")


# ---------------------------------------------------------------------------
# Bench-only scan across multiple machines
# ---------------------------------------------------------------------------

def get_logs_text(iid: int, tail: int = 400) -> str:
    try:
        out = vast("logs", iid, "--tail", tail, raw=False, check=False)
        return out or ""
    except Exception:
        return ""


def extract_marker_json(logs: str, marker: str = "BENCHMARK_JSON "):
    for line in reversed(logs.splitlines()):
        if marker in line:
            try:
                return json.loads(line.split(marker, 1)[1])
            except json.JSONDecodeError:
                return None
    return None


def cmd_scan(args):
    secrets = load_secrets()
    offers = search_offers(args.gpu, args.max_dph, args.inet, limit=40)
    if len(offers) < args.n:
        sys.exit(f"Only {len(offers)} offers matched; need {args.n}.")

    # Prefer distinct physical machines for a meaningful sample, sampled
    # around the median price (cheapest offers over-sample lemons).
    offers.sort(key=lambda o: abs(
        o["dph_total"] - statistics.median(x["dph_total"] for x in offers)))
    picked, seen_machines = [], set()
    for o in offers:
        m = o.get("machine_id")
        if m not in seen_machines:
            picked.append(o)
            seen_machines.add(m)
        if len(picked) == args.n:
            break

    pending = {}
    for o in picked:
        try:
            iid = create_instance(o["id"], secrets, args.branch, "",
                                  bench_only=True, keep_alive=False,
                                  purpose="scan")
            pending[iid] = {"offer": o, "result": None}
            print(f"Scan instance {iid} created on machine {o.get('machine_id')} "
                  f"(${o.get('dph_total', 0):.3f}/hr)")
        except Exception as e:
            print(f"Failed to create on offer {o['id']}: {e}")

    if not pending:
        sys.exit("No scan instances created.")

    deadline = time.time() + args.timeout_min * 60
    print(f"\nPolling logs every 60s (timeout {args.timeout_min} min)...")
    while time.time() < deadline:
        done = all(v["result"] is not None for v in pending.values())
        if done:
            break
        time.sleep(60)
        for iid, slot in pending.items():
            if slot["result"] is not None:
                continue
            logs = get_logs_text(iid)
            bench = extract_marker_json(logs)
            if bench is not None:
                slot["result"] = bench
                print(f"  {iid}: benchmark received "
                      f"({bench.get('gpu_name', '?')}, "
                      f"{bench.get('download_mbps', '?')} Mbps)")
            elif "ONSTART_BEGIN" in logs:
                print(f"  {iid}: booted, benchmarking...")
            else:
                print(f"  {iid}: waiting for boot (image pull)...")

    # Clean up: instances self-destroy, but never trust that alone.
    for iid in pending:
        try:
            vast("destroy", "instance", iid, "-y", check=False)
        except Exception:
            pass
    remaining = [r for r in load_state() if r["id"] not in pending]
    save_state(remaining)

    results = []
    for iid, slot in pending.items():
        entry = {"instance": iid,
                 "machine_id": slot["offer"].get("machine_id"),
                 "dph": slot["offer"].get("dph_total"),
                 "geolocation": slot["offer"].get("geolocation")}
        if slot["result"]:
            entry.update(slot["result"])
        else:
            entry["error"] = "timeout — no benchmark received"
        results.append(entry)
    SCAN_OUT.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {SCAN_OUT}")

    metrics = ["download_mbps", "disk_write_mbps", "cpu_jpeg_per_sec",
               "pcie_h2d_gbps", "gpu_bf16_tflops"]
    ok = [r for r in results if "error" not in r]
    if ok:
        print("\nResults:")
        for r in ok:
            print("  " + json.dumps(r))
        print("\nSuggested gate thresholds (70% of median — update "
              "vast/thresholds.json and push):")
        for m in metrics:
            vals = [r[m] for r in ok if m in r]
            if vals:
                print(f'  "{m}": {round(0.7 * statistics.median(vals), 1)},')
    else:
        print("No successful benchmarks — inspect logs / retry.")


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

def cmd_status(args):
    instances = vast("show", "instances")
    if not instances:
        print("No live instances on this account.")
        return
    tracked = {r["id"]: r for r in load_state()}
    for inst in instances:
        iid = inst["id"]
        info = tracked.get(iid, {})
        print(f"  {iid}  {inst.get('actual_status') or 'creating':<10} "
              f"{inst.get('gpu_name') or '?':<14} "
              f"${inst.get('dph_total') or 0:.3f}/hr  "
              f"purpose={info.get('purpose', 'untracked')}  "
              f"label={inst.get('label') or ''}")


def cmd_logs(args):
    iid = resolve_id(args)
    print(vast("logs", iid, "--tail", args.tail, raw=False, check=False))


def cmd_ssh(args):
    iid = resolve_id(args)
    print(vast("ssh-url", iid, raw=False, check=False).strip())


def cmd_destroy(args):
    if args.all:
        instances = vast("show", "instances")
        ids = [i["id"] for i in instances] if instances else []
    else:
        ids = [resolve_id(args)]
    for iid in ids:
        result = vast("destroy", "instance", iid, "-y", check=False)
        print(f"destroy {iid}: {result}")
    save_state([r for r in load_state() if r["id"] not in set(ids)])


# ---------------------------------------------------------------------------

def main():
    # Windows consoles default to cp1252; instance logs contain UTF-8.
    # line_buffering so progress is visible when output is redirected.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace",
                               line_buffering=True)
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        # RTX 6000 Blackwell workstation cards; B200 for occasional rapid
        # runs via --gpu B200 (note: B200 market floor may exceed max-dph).
        sp.add_argument("--gpu",     type=str,   default="RTX_PRO_6000_WS")
        sp.add_argument("--max-dph", type=float, default=1.2, dest="max_dph")
        sp.add_argument("--inet",    type=int,   default=500)
        sp.add_argument("--branch",  type=str,   default="master")

    sp = sub.add_parser("search");  common(sp); sp.set_defaults(fn=cmd_search)

    sp = sub.add_parser("launch");  common(sp)
    sp.add_argument("--offer",      type=int, default=None)
    sp.add_argument("--train-args", type=str, dest="train_args",
                    default="--num_epochs 300 --artifact_every 25")
    sp.add_argument("--keep-alive", action="store_true", dest="keep_alive")
    sp.add_argument("--smoke",      action="store_true",
                    help="1-epoch pipeline test with keep-alive")
    sp.set_defaults(fn=cmd_launch)

    sp = sub.add_parser("scan");    common(sp)
    sp.add_argument("--n",           type=int, default=3)
    sp.add_argument("--timeout-min", type=int, default=30, dest="timeout_min")
    sp.set_defaults(fn=cmd_scan)

    sp = sub.add_parser("status");  sp.set_defaults(fn=cmd_status)

    sp = sub.add_parser("logs")
    sp.add_argument("--id",   type=int, default=None)
    sp.add_argument("--tail", type=int, default=120)
    sp.set_defaults(fn=cmd_logs)

    sp = sub.add_parser("ssh")
    sp.add_argument("--id", type=int, default=None)
    sp.set_defaults(fn=cmd_ssh)

    sp = sub.add_parser("destroy")
    sp.add_argument("--id",  type=int, default=None)
    sp.add_argument("--all", action="store_true")
    sp.set_defaults(fn=cmd_destroy)

    sp = sub.add_parser("blacklist",
                        help="add a machine_id to the do-not-rent list")
    sp.add_argument("machine_id", type=int)
    sp.set_defaults(fn=lambda a: (add_to_blacklist(a.machine_id),
                                  print(f"blacklisted: {sorted(load_blacklist())}")))

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
