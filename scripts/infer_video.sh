#!/bin/bash
#
# Run Video2World inference from a previously exported batch JSON.
# Edit the arguments below, then run:
#   ./entrypoints/infer_video.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODEL_DIR="${REPO_ROOT}/mimic-video/model"

# ---- Arguments ------------------------------------------------------------
EX_TYPE="${EX_TYPE:-ex1}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data}"
DATASET_NAME="${DATASET_NAME:-ex1_merged_2026-05-13_12-20-44}"
RUN_DIR="${RUN_DIR:-${REPO_ROOT}/runs/video_inference/${DATASET_NAME}}"
OUTPUT_DIR="${OUTPUT_DIR:-${RUN_DIR}/outputs}"
BATCH_JSON="${BATCH_JSON:-${RUN_DIR}/batch.json}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${REPO_ROOT}/mimic-video/model/checkpoints}"
VIDEO_DIT_PATH="${VIDEO_DIT_PATH:-${CHECKPOINT_DIR}/video_backbone/v2w_pretrained_cosmos.pt}"
NUM_CONDITIONAL_FRAMES="${NUM_CONDITIONAL_FRAMES:-5}"
GUIDANCE="${GUIDANCE:-7}"
SEED="${SEED:-0}"
# ---------------------------------------------------------------------------

MODEL_PYTHON="${MODEL_DIR}/.venv/bin/python"

export PATH="/sbin:/usr/sbin:${PATH}"
export CUDA_HOME="${MODEL_DIR}/.venv/lib/python3.10/site-packages/nvidia/cuda_nvrtc"
export CUDA_PATH="${CUDA_HOME}"
export LD_LIBRARY_PATH="${MODEL_DIR}/.venv/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib:${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${MODEL_DIR}/.venv/lib/python3.10/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"
export COSMOS_PREDICT2_ARGS="${COSMOS_PREDICT2_ARGS:---checkpoints ${CHECKPOINT_DIR}}"

echo "=== Running Video2World ==="
echo "Ex type:       ${EX_TYPE}"
echo "Dataset:       ${DATASET_NAME}"
echo "Run dir:       ${RUN_DIR}"
echo "Batch JSON:    ${BATCH_JSON}"
echo "Video ckpt:    ${VIDEO_DIT_PATH}"
echo

if [[ ! -f "${BATCH_JSON}" ]]; then
  echo "Batch JSON not found: ${BATCH_JSON}" >&2
  echo "Run ./entrypoints/export_video_inputs.sh first." >&2
  exit 1
fi

cd "${MODEL_DIR}"
"${MODEL_PYTHON}" scripts/run_video2world.py \
  --dit_path "${VIDEO_DIT_PATH}" \
  --batch_input_json "${BATCH_JSON}" \
  --num_conditional_frames "${NUM_CONDITIONAL_FRAMES}" \
  --guidance "${GUIDANCE}" \
  --seed "${SEED}" \
  --disable_guardrail

echo "=== Inference complete ==="
echo "Outputs: ${OUTPUT_DIR}"
