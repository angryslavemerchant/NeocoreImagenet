#!/usr/bin/env bash
# vast/onstart.sh — runs once at instance boot, invoked by the onstart
# command that vast/launch.py builds (repo is already cloned, cwd is set,
# and TRAIN_ARGS / BENCH_ONLY / KEEP_ALIVE are exported there).
#
# Secrets (WANDB_API_KEY, HF_TOKEN, VAST_API_KEY) arrive as container env
# vars via `vastai create instance --env`.
#
# Log markers scraped by vast/launch.py:
#   ONSTART_BEGIN / BENCH_ONLY_DONE / GATE_FAILED / GATE_PASSED /
#   TRAIN_LAUNCHED / SELF_DESTROY  (+ BENCHMARK_JSON from benchmark.py)

set -u
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
export DEBIAN_FRONTEND=noninteractive

INSTANCE_ID="${VAST_CONTAINERLABEL#C.}"
echo "ONSTART_BEGIN instance=${INSTANCE_ID} $(date -u +%FT%TZ)"

self_destroy() {
    echo "SELF_DESTROY instance=${INSTANCE_ID}"
    sleep 20   # let the log collector catch the final lines
    vastai destroy instance "${INSTANCE_ID}" --api-key "${VAST_API_KEY}"
}

pip install -q vastai
command -v tmux >/dev/null 2>&1 || { apt-get update -qq && apt-get install -y -qq tmux; }

# --- Benchmark-only mode: measure, report, self-destroy -------------------
if [ -n "${BENCH_ONLY:-}" ]; then
    pip install -q numpy Pillow
    python vast/benchmark.py --out /workspace/benchmark.json
    echo "BENCH_ONLY_DONE"
    self_destroy
    exit 0
fi

# --- Full provisioning -----------------------------------------------------
echo "INSTALLING_DEPS"
pip install -q -r requirements.txt

# --- Health gate: refuse to train on a sick machine -------------------------
if ! python vast/benchmark.py --gate vast/thresholds.json --out /workspace/benchmark.json; then
    echo "GATE_FAILED — instance below thresholds, destroying"
    self_destroy
    exit 1
fi
echo "GATE_PASSED"

# --- Launch training detached so it survives everything ---------------------
tmux new-session -d -s train "bash vast/run_training.sh 2>&1 | tee /workspace/train.log"
echo "TRAIN_LAUNCHED tmux=train log=/workspace/train.log"
