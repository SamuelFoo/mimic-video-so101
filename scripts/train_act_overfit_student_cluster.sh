#!/bin/bash
#
# Student-cluster ACT overfit runner.
#
# This keeps the original ACT launcher untouched while providing defaults that
# work on the ETH student cluster and the current local Ex1 merged dataset.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONDA_ENV="${CONDA_ENV:-lerobot}"
DEFAULT_DATASET_NAME="ex1_merged"
DEFAULT_DATASET_SRC="${REPO_ROOT}/data/${DEFAULT_DATASET_NAME}"
SCRATCH_DATASET_SRC="/work/scratch/${USER}/data/${DEFAULT_DATASET_NAME}"
if [[ -d "${SCRATCH_DATASET_SRC}" ]]; then
    DEFAULT_DATASET_SRC="${SCRATCH_DATASET_SRC}"
fi

DEFAULT_RUNS_ROOT="${REPO_ROOT}/runs"
SCRATCH_RUNS_ROOT="/work/scratch/${USER}/rl_runs"
if [[ -d "/work/scratch/${USER}" ]]; then
    DEFAULT_RUNS_ROOT="${SCRATCH_RUNS_ROOT}"
fi

DATASET_SRC="${DATASET_SRC:-${DEFAULT_DATASET_SRC}}"
DATASET_NAME="${DATASET_NAME:-$(basename "${DATASET_SRC}")}"
OVERFIT_EPISODES="${OVERFIT_EPISODES:-}"
OVERFIT_SUFFIX=""
if [[ -n "${OVERFIT_EPISODES}" ]]; then
    OVERFIT_SUFFIX="_first${OVERFIT_EPISODES}"
fi

RUN_NAME="${RUN_NAME:-act_${DATASET_NAME}${OVERFIT_SUFFIX}}"
RUNS_ROOT="${RUNS_ROOT:-${DEFAULT_RUNS_ROOT}}"
OUTPUT_DIR="${OUTPUT_DIR:-${RUNS_ROOT}/${RUN_NAME}_$(date +%Y%m%d_%H%M%S)}"
DATASET_DST="${DATASET_DST:-${TMPDIR:-/tmp}/${DATASET_NAME}}"
COPY_TO_TMP="${COPY_TO_TMP:-true}"

POLICY_DEVICE="${POLICY_DEVICE:-cuda}"
POLICY_REPO_ID="${POLICY_REPO_ID:-local/${RUN_NAME}}"
PUSH_TO_HUB="${PUSH_TO_HUB:-false}"

WANDB_ENABLE="${WANDB_ENABLE:-true}"
WANDB_MODE="${WANDB_MODE:-offline}"
WANDB_PROJECT="${WANDB_PROJECT:-robot-learning-act}"
WANDB_DISABLE_ARTIFACT="${WANDB_DISABLE_ARTIFACT:-true}"

VIDEO_BACKEND="${VIDEO_BACKEND:-pyav}"

STEPS="${STEPS:-10000}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LOG_FREQ="${LOG_FREQ:-10}"
SAVE_FREQ="${SAVE_FREQ:-500}"
NUM_WORKERS="${NUM_WORKERS:-4}"

RESUME="${RESUME:-false}"
CONFIG_PATH="${CONFIG_PATH:-}"

resolve_conda_base() {
    if [[ -n "${CONDA_BASE:-}" ]]; then
        echo "${CONDA_BASE}"
        return
    fi

    if [[ -n "${CONDA_EXE:-}" ]]; then
        local conda_exe_dir
        conda_exe_dir="$(cd "$(dirname "${CONDA_EXE}")/.." && pwd)"
        if [[ -f "${conda_exe_dir}/bin/activate" ]]; then
            echo "${conda_exe_dir}"
            return
        fi
    fi

    if command -v conda >/dev/null 2>&1; then
        conda info --base
        return
    fi

    for candidate in \
        "/work/scratch/${USER}/miniconda3" \
        "${HOME}/miniconda3" \
        "${HOME}/mambaforge" \
        "${HOME}/anaconda3"; do
        if [[ -f "${candidate}/bin/activate" ]]; then
            echo "${candidate}"
            return
        fi
    done
}

CONDA_BASE="$(resolve_conda_base)"
if [[ -z "${CONDA_BASE}" || ! -f "${CONDA_BASE}/bin/activate" ]]; then
    echo "Could not find a conda installation. Set CONDA_BASE=/path/to/conda." >&2
    exit 1
fi

source "${CONDA_BASE}/bin/activate"
conda activate "${CONDA_ENV}"

export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

if [[ ! -d "${DATASET_SRC}" ]]; then
    echo "Dataset directory not found: ${DATASET_SRC}" >&2
    exit 1
fi

if [[ ! -f "${DATASET_SRC}/meta/info.json" ]]; then
    echo "Expected a LeRobot dataset at ${DATASET_SRC} (missing meta/info.json)." >&2
    exit 1
fi

mkdir -p "${RUNS_ROOT}"

ACTIVE_DATASET_PATH="${DATASET_SRC}"
if [[ "${COPY_TO_TMP}" == "true" ]]; then
    echo "Copying dataset to ${DATASET_DST} ..."
    mkdir -p "${DATASET_DST}"
    rsync -a --delete "${DATASET_SRC}/" "${DATASET_DST}/"
    ACTIVE_DATASET_PATH="${DATASET_DST}"
    echo "Dataset copy done."
fi

DATASET_ROOT="${ACTIVE_DATASET_PATH}"
DATASET_REPO_ID="$(basename "${ACTIVE_DATASET_PATH}")"
DATASET_EPISODES_ARG=""

if [[ -n "${OVERFIT_EPISODES}" ]]; then
    if ! [[ "${OVERFIT_EPISODES}" =~ ^[0-9]+$ ]]; then
        echo "OVERFIT_EPISODES must be a positive integer." >&2
        exit 1
    fi

    TOTAL_EPISODES="$(python3 - <<'PY' "${ACTIVE_DATASET_PATH}/meta/info.json"
import json
import sys

with open(sys.argv[1]) as f:
    info = json.load(f)

print(info["total_episodes"])
PY
)"

    if (( OVERFIT_EPISODES < 1 || OVERFIT_EPISODES > TOTAL_EPISODES )); then
        echo "OVERFIT_EPISODES must be between 1 and ${TOTAL_EPISODES}." >&2
        exit 1
    fi

    if (( OVERFIT_EPISODES < TOTAL_EPISODES )); then
        DATASET_EPISODES_ARG="$(python3 - <<'PY' "${OVERFIT_EPISODES}"
import sys

keep = int(sys.argv[1])
print("[" + ",".join(str(i) for i in range(keep)) + "]")
PY
)"
        echo "Training on the first ${OVERFIT_EPISODES} / ${TOTAL_EPISODES} episodes."
    fi
fi

echo "=== ACT Training Job ==="
echo "Node:             $(hostname)"
echo "Dataset source:   ${DATASET_SRC}"
echo "Dataset active:   ${ACTIVE_DATASET_PATH}"
echo "Dataset root:     ${DATASET_ROOT}"
echo "Dataset repo_id:  ${DATASET_REPO_ID}"
echo "Runs root:        ${RUNS_ROOT}"
echo "Output dir:       ${OUTPUT_DIR}"
echo "Run name:         ${RUN_NAME}"
echo "Batch size:       ${BATCH_SIZE}"
echo "Steps:            ${STEPS}"
echo "WandB:            enable=${WANDB_ENABLE}, mode=${WANDB_MODE}, project=${WANDB_PROJECT}"
echo "WandB artifacts:  ${WANDB_DISABLE_ARTIFACT}"
echo "Push to hub:      ${PUSH_TO_HUB}"
echo "Video backend:    ${VIDEO_BACKEND}"
echo "Num workers:      ${NUM_WORKERS}"
echo "Resume:           ${RESUME}"
if [[ -n "${CONFIG_PATH}" ]]; then
    echo "Config path:      ${CONFIG_PATH}"
fi
if [[ -n "${DATASET_EPISODES_ARG}" ]]; then
    echo "Episode subset:   ${DATASET_EPISODES_ARG}"
fi
echo ""

TRAIN_ARGS=(
  "--dataset.root=${DATASET_ROOT}"
  "--dataset.repo_id=${DATASET_REPO_ID}"
  "--dataset.video_backend=${VIDEO_BACKEND}"
  "--policy.type=act"
  "--output_dir=${OUTPUT_DIR}"
  "--job_name=${RUN_NAME}"
  "--resume=${RESUME}"
  "--policy.device=${POLICY_DEVICE}"
  "--wandb.enable=${WANDB_ENABLE}"
  "--wandb.disable_artifact=${WANDB_DISABLE_ARTIFACT}"
  "--wandb.mode=${WANDB_MODE}"
  "--wandb.project=${WANDB_PROJECT}"
  "--policy.repo_id=${POLICY_REPO_ID}"
  "--policy.push_to_hub=${PUSH_TO_HUB}"
  "--num_workers=${NUM_WORKERS}"
  "--batch_size=${BATCH_SIZE}"
  "--steps=${STEPS}"
  "--log_freq=${LOG_FREQ}"
  "--save_freq=${SAVE_FREQ}"
)

if [[ -n "${DATASET_EPISODES_ARG}" ]]; then
  TRAIN_ARGS+=("--dataset.episodes=${DATASET_EPISODES_ARG}")
fi

if [[ -n "${CONFIG_PATH}" ]]; then
  TRAIN_ARGS+=("--config_path=${CONFIG_PATH}")
fi

lerobot-train "${TRAIN_ARGS[@]}" "$@"

echo "=== Training Complete ==="
