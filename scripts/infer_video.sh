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
RUN_DIR="$(cd "${RUN_DIR:-${REPO_ROOT}/runs/video_inference/${DATASET_NAME}}" && pwd)"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${REPO_ROOT}/mimic-video/model/checkpoints}"
VIDEO_DIT_PATH="${VIDEO_DIT_PATH:-${REPO_ROOT}/checkpoints/model/old610.pt}"
MODEL_NAME="${MODEL_NAME:-$(basename "${VIDEO_DIT_PATH}" .pt)}"
OUTPUT_DIR="${OUTPUT_DIR:-${RUN_DIR}/outputs/${MODEL_NAME}}"
BATCH_JSON="${BATCH_JSON:-${RUN_DIR}/batch.json}"
NUM_CONDITIONAL_FRAMES="${NUM_CONDITIONAL_FRAMES:-5}"
GUIDANCE="${GUIDANCE:-7}"
SEED="${SEED:-0}"
EPISODE_STRIDE="${EPISODE_STRIDE:-5}"  # run every Nth episode; 1 = all episodes
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

PATCHED_BATCH_JSON="$(mktemp --suffix=.json)"
trap 'rm -f "${PATCHED_BATCH_JSON}"' EXIT
sed "s|${RUN_DIR}/outputs/|${OUTPUT_DIR}/|g" "${BATCH_JSON}" \
  | python3 -c "import json,sys; d=json.load(sys.stdin); print(json.dumps(d[::${EPISODE_STRIDE}], indent=2))" \
  > "${PATCHED_BATCH_JSON}"
echo "Episodes selected: $(python3 -c "import json; d=json.load(open('${PATCHED_BATCH_JSON}')); print(len(d))")" \
     "(stride=${EPISODE_STRIDE})"
mkdir -p "${OUTPUT_DIR}"

cd "${MODEL_DIR}"
"${MODEL_PYTHON}" scripts/run_video2world.py \
  --dit_path "${VIDEO_DIT_PATH}" \
  --batch_input_json "${PATCHED_BATCH_JSON}" \
  --num_conditional_frames "${NUM_CONDITIONAL_FRAMES}" \
  --guidance "${GUIDANCE}" \
  --seed "${SEED}" \
  --disable_guardrail

echo "=== Inference complete ==="
echo "Outputs: ${OUTPUT_DIR}"
