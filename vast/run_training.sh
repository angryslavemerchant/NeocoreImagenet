#!/usr/bin/env bash
# vast/run_training.sh — train -> eval -> upload results -> self-destroy.
# Runs inside tmux (started by onstart.sh). On failure the instance is
# LEFT ALIVE for inspection; only a fully successful run destroys itself
# (set KEEP_ALIVE=1 at launch to disable auto-destroy entirely).

set -u
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1

INSTANCE_ID="${VAST_CONTAINERLABEL#C.}"
export WANDB_PROJECT="${WANDB_PROJECT:-asfnetAE}"

# PY / VAST_CLI are exported by onstart.sh; resolve again if run standalone.
PY="${PY:-$( [ -x /venv/main/bin/python ] && echo /venv/main/bin/python || echo python3 )}"
VAST_CLI="${VAST_CLI:-vastai}"

# Share one wandb run id between training and the post-eval upload step.
export WANDB_RUN_ID="${WANDB_RUN_ID:-$("$PY" -c 'import wandb.util,sys; sys.stdout.write(wandb.util.generate_id())')}"
export WANDB_RESUME=allow

TRAIN_ARGS="${TRAIN_ARGS:---num_epochs 300 --artifact_every 25}"
echo "TRAIN_START run_id=${WANDB_RUN_ID} args=${TRAIN_ARGS}"

"$PY" train_asfnet_ae.py ${TRAIN_ARGS}
STATUS=$?
echo "TRAIN_EXIT status=${STATUS}"

if [ "${STATUS}" -ne 0 ]; then
    echo "RUN_FAILED — leaving instance alive for inspection (destroy manually)"
    exit "${STATUS}"
fi

# --- Post-training eval: reconstruction + retention visualisations ----------
echo "EVAL_START"
"$PY" evaluate_asfnet_br.py --ae \
    --checkpoint checkpoints_asfnet_ae/best.pt \
    --output_dir viz_ae || echo "EVAL_FAILED (continuing to upload)"

# --- Push checkpoints + viz images to wandb ---------------------------------
"$PY" vast/upload_results.py \
    --viz_dir viz_ae \
    --ckpt_dir checkpoints_asfnet_ae \
    --extra /workspace/benchmark.json || echo "UPLOAD_FAILED"

echo "RUN_COMPLETE"

if [ -z "${KEEP_ALIVE:-}" ]; then
    echo "SELF_DESTROY instance=${INSTANCE_ID}"
    sleep 30
    "$VAST_CLI" destroy instance "${INSTANCE_ID}" --api-key "${VAST_API_KEY}" -y \
        || echo y | "$VAST_CLI" destroy instance "${INSTANCE_ID}" --api-key "${VAST_API_KEY}"
else
    echo "KEEP_ALIVE set — instance left running"
fi
