#!/bin/bash
#
# Export existing zarr episodes to MP4 inputs plus a Video2World batch JSON.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODEL_DIR="${REPO_ROOT}/mimic-video/model"

# ---- Arguments ------------------------------------------------------------
EX_TYPE="${EX_TYPE:-ex1}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data}"
DATASET_NAME="${DATASET_NAME:-${EX_TYPE}_merged}"
INPUT_DIR="${INPUT_DIR:-${DATA_ROOT}/${DATASET_NAME}}"
ZARR_DIR="${ZARR_DIR:-${DATA_ROOT}/${DATASET_NAME}-zarr}"
RUN_DIR="${RUN_DIR:-${REPO_ROOT}/runs/video_inference/${DATASET_NAME}}"
EXPORTED_VIDEO_DIR="${EXPORTED_VIDEO_DIR:-${RUN_DIR}/inputs}"
OUTPUT_DIR="${OUTPUT_DIR:-${RUN_DIR}/outputs}"
BATCH_JSON="${BATCH_JSON:-${RUN_DIR}/batch.json}"
INSTRUCTIONS_JSON="${INSTRUCTIONS_JSON:-${REPO_ROOT}/config/language_instructions.json}"
OVERWRITE="${OVERWRITE:-false}"
MAX_EPISODES="${MAX_EPISODES:-}" # empty means all episodes
EPISODES=()     # e.g. (0 4 7)
FPS="${FPS:-10}"
# ---------------------------------------------------------------------------

MODEL_PYTHON="${MODEL_DIR}/.venv/bin/python"
LANGUAGE_INSTRUCTION="$(python3 "${REPO_ROOT}/helpers/utils/language_instructions.py" "${EX_TYPE}" --instructions "${INSTRUCTIONS_JSON}")"

export PATH="/sbin:/usr/sbin:${PATH}"
export CUDA_HOME="${MODEL_DIR}/.venv/lib/python3.10/site-packages/nvidia/cuda_nvrtc"
export CUDA_PATH="${CUDA_HOME}"
export LD_LIBRARY_PATH="${MODEL_DIR}/.venv/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib:${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${MODEL_DIR}/.venv/lib/python3.10/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"

echo "=== Export Video2World inputs ==="
echo "Ex type:       ${EX_TYPE}"
echo "Dataset:       ${DATASET_NAME}"
echo "Zarr dir:      ${ZARR_DIR}"
echo "Run dir:       ${RUN_DIR}"
echo "Batch JSON:    ${BATCH_JSON}"
echo "Overwrite:     ${OVERWRITE}"
echo "Instruction:   ${LANGUAGE_INSTRUCTION}"
echo

if [[ "${OVERWRITE}" != "true" && "${OVERWRITE}" != "false" ]]; then
  echo "OVERWRITE must be true or false" >&2
  exit 2
fi

if [[ ! -d "${ZARR_DIR}" ]]; then
  echo "Zarr directory not found: ${ZARR_DIR}" >&2
  echo "Run ./scripts/process_lerobot.sh first." >&2
  exit 1
fi

export_args=(
  --zarr-dir "${ZARR_DIR}"
  --video-dir "${EXPORTED_VIDEO_DIR}"
  --output-dir "${OUTPUT_DIR}"
  --batch-json "${BATCH_JSON}"
  --ex-type "${EX_TYPE}"
  --instructions "${INSTRUCTIONS_JSON}"
  --prompt "${LANGUAGE_INSTRUCTION}"
  --fps "${FPS}"
)

if [[ -n "${MAX_EPISODES}" ]]; then
  export_args+=(--max-episodes "${MAX_EPISODES}")
fi

if [[ "${#EPISODES[@]}" -gt 0 ]]; then
  export_args+=(--episodes "${EPISODES[@]}")
fi

if [[ "${OVERWRITE}" == "true" ]]; then
  export_args+=(--overwrite)
fi

echo "=== Exporting zarr episodes to MP4 batch inputs ==="
cd "${REPO_ROOT}"
"${MODEL_PYTHON}" "${REPO_ROOT}/helpers/export_zarr_videos.py" "${export_args[@]}"

echo "=== Export complete ==="
echo "Batch JSON: ${BATCH_JSON}"
echo "Inputs:     ${EXPORTED_VIDEO_DIR}"
echo "Outputs:    ${OUTPUT_DIR}"
