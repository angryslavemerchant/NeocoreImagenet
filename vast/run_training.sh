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

TRAIN_SCRIPT="${TRAIN_SCRIPT:-train_asfnet_ae.py}"
TRAIN_ARGS="${TRAIN_ARGS:---num_epochs 300 --artifact_every 25}"
echo "TRAIN_START script=${TRAIN_SCRIPT} run_id=${WANDB_RUN_ID} args=${TRAIN_ARGS}"

"$PY" "${TRAIN_SCRIPT}" ${TRAIN_ARGS}
STATUS=$?
echo "TRAIN_EXIT status=${STATUS}"

if [ "${STATUS}" -ne 0 ]; then
    echo "RUN_FAILED — leaving instance alive for inspection (destroy manually)"
    exit "${STATUS}"
fi

# --- AE-only post-processing: eval panels into the run folder ---------------
# Other train scripts (e.g. train_linear_probe.py) log metrics to wandb
# themselves and need no separate eval step. runs/LATEST is written by the
# train script and points at runs/<run_name>/.
UPLOAD_OK=1
if [ "${TRAIN_SCRIPT}" = "train_asfnet_ae.py" ]; then
    RUN_DIR="$(cat runs/LATEST 2>/dev/null || echo checkpoints_asfnet_ae)"
    echo "EVAL_START run_dir=${RUN_DIR}"
    "$PY" evaluate_asfnet_br.py --ae \
        --checkpoint "${RUN_DIR}/best.pt" \
        --output_dir "${RUN_DIR}/viz" || echo "EVAL_FAILED (continuing to upload)"

    # upload_results VERIFIES the final artifact committed (art.wait()) and
    # exits non-zero otherwise — self-destroy below is gated on that, so a
    # wandb storage outage can never again destroy the only copy of the
    # weights (2026-07-15 lesson).
    "$PY" vast/upload_results.py \
        --viz_dir "${RUN_DIR}/viz" \
        --ckpt_dir "${RUN_DIR}" \
        --extra /workspace/benchmark.json \
        || { echo "UPLOAD_FAILED (results remain on-instance)"; UPLOAD_OK=0; }
fi
# Non-AE scripts (e.g. train_linear_probe.py) log all results to wandb
# during training and keep no precious on-instance state.

echo "RUN_COMPLETE"

if [ -n "${KEEP_ALIVE:-}" ]; then
    echo "KEEP_ALIVE set — instance left running"
elif [ "${UPLOAD_OK}" -eq 1 ]; then
    echo "SELF_DESTROY instance=${INSTANCE_ID}"
    sleep 30
    "$VAST_CLI" destroy instance "${INSTANCE_ID}" --api-key "${VAST_API_KEY}" -y \
        || echo y | "$VAST_CLI" destroy instance "${INSTANCE_ID}" --api-key "${VAST_API_KEY}"
else
    # Fallback: results only exist here — hold for a manual/agent pull.
    #     python vast/launch.py pull --id ${INSTANCE_ID}
    #     python vast/launch.py destroy --id ${INSTANCE_ID}
    echo "AWAITING_PULL instance=${INSTANCE_ID} dir=runs/"
fi
