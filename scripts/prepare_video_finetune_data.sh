#!/bin/bash
#
# Prepare the Cosmos Video2World finetuning dataset from per-episode zarrs.
# Produces:
#   <FINETUNE_DATA_DIR>/video/episode_XXXXXX.mp4
#   <FINETUNE_DATA_DIR>/metas/episode_XXXXXX.txt
#   <FINETUNE_DATA_DIR>/t5_xxl/episode_XXXXXX.pickle    (from get_t5_embeddings.py)
#
# Run from the repo root after ./scripts/process_lerobot.sh.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODEL_DIR="${REPO_ROOT}/mimic-video/model"
MODEL_PYTHON="${MODEL_DIR}/.venv/bin/python"

# ---- Arguments ------------------------------------------------------------
EX_TYPE="${EX_TYPE:-ex2}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/staging/mimic-video}"
DATASET_NAME="${DATASET_NAME:-${EX_TYPE}_all_v4}"
ZARR_DIR="${ZARR_DIR:-${DATA_ROOT}/${DATASET_NAME}-zarr}"
FINETUNE_DATA_DIR="${FINETUNE_DATA_DIR:-${DATA_ROOT}/${DATASET_NAME}-cosmos-video}"
INSTRUCTIONS_JSON="${INSTRUCTIONS_JSON:-${REPO_ROOT}/config/language_instructions.json}"
FPS="${FPS:-10}"
OVERWRITE="${OVERWRITE:-false}"
SKIP_T5="${SKIP_T5:-false}"
# ---------------------------------------------------------------------------

export PATH="/sbin:/usr/sbin:${PATH}"
export CUDA_HOME="${MODEL_DIR}/.venv/lib/python3.10/site-packages/nvidia/cuda_nvrtc"
export CUDA_PATH="${CUDA_HOME}"
export LD_LIBRARY_PATH="${MODEL_DIR}/.venv/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib:${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${MODEL_DIR}/.venv/lib/python3.10/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"

if [[ ! -d "${ZARR_DIR}" ]]; then
  echo "Zarr directory not found: ${ZARR_DIR}" >&2
  echo "Run ./scripts/process_lerobot.sh first." >&2
  exit 1
fi

echo "=== Prepare Cosmos Video2World finetuning data ==="
echo "Ex type:       ${EX_TYPE}"
echo "Zarr dir:      ${ZARR_DIR}"
echo "Finetune dir:  ${FINETUNE_DATA_DIR}"
echo "FPS:           ${FPS}"
echo "Overwrite:     ${OVERWRITE}"
echo

prep_args=(
  --zarr-dir "${ZARR_DIR}"
  --out-dir "${FINETUNE_DATA_DIR}"
  --ex-type "${EX_TYPE}"
  --instructions "${INSTRUCTIONS_JSON}"
  --fps "${FPS}"
)
if [[ "${OVERWRITE}" == "true" ]]; then
  prep_args+=(--overwrite)
fi

echo "=== Step 1: zarr -> video/ + metas/ ==="
cd "${REPO_ROOT}"
"${MODEL_PYTHON}" "${REPO_ROOT}/helpers/prepare_video_finetune_data.py" "${prep_args[@]}"

if [[ "${SKIP_T5}" == "true" ]]; then
  echo "SKIP_T5=true — leaving t5_xxl/ empty."
  exit 0
fi

echo "=== Step 2: T5 embeddings -> t5_xxl/ ==="
T5_SCRIPT="${REPO_ROOT}/mimic-video/data_preprocessing/video/get_t5_embeddings.py"
cd "${MODEL_DIR}"  # imports rely on the mimic-video model package
PYTHONPATH="${MODEL_DIR}:${PYTHONPATH:-}" "${MODEL_PYTHON}" "${T5_SCRIPT}" \
    --dataset_path "${FINETUNE_DATA_DIR}"

echo "=== Done ==="
echo "Dataset ready at: ${FINETUNE_DATA_DIR}"
