#!/bin/bash
#
# Standalone script for converting a LeRobot v3 dataset to mimic-video's
# per-episode .zarr layout, then precomputing T5 language embeddings.
# Can also be invoked by slurm_scripts/process_lerobot.sbatch.
#
# NOTE: The lerobot conda env needs zarr<3 / numcodecs<0.16 installed.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODEL_DIR="${REPO_ROOT}/mimic-video/model"

# ---- Arguments ------------------------------------------------------------
EX_TYPE="${EX_TYPE:-ex1}"
CONDA_ENV="${CONDA_ENV:-lerobot}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data}"
DATASET_NAME="${DATASET_NAME:-${EX_TYPE}_merged}"
INPUT_DIR="${INPUT_DIR:-${DATA_ROOT}/${DATASET_NAME}}"
OUTPUT_DIR="${OUTPUT_DIR:-${DATA_ROOT}/${DATASET_NAME}-zarr}"
INSTRUCTIONS_JSON="${INSTRUCTIONS_JSON:-${REPO_ROOT}/config/language_instructions.json}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${REPO_ROOT}/mimic-video/model/checkpoints}"
CONDA_BASE="${CONDA_BASE:-$(conda info --base 2>/dev/null || echo "${HOME}/miniconda3")}"
# ---------------------------------------------------------------------------

LANGUAGE_INSTRUCTION="$(python3 "${REPO_ROOT}/helpers/utils/language_instructions.py" "${EX_TYPE}" --instructions "${INSTRUCTIONS_JSON}")"

source "${CONDA_BASE}/bin/activate"
conda activate "${CONDA_ENV}"

export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

echo "=== LeRobot -> zarr conversion ==="
echo "Node:        $(hostname)"
echo "Conda env:   ${CONDA_ENV}"
echo "Ex type:     ${EX_TYPE}"
echo "Dataset:     ${DATASET_NAME}"
echo "Input dir:   ${INPUT_DIR}"
echo "Output dir:  ${OUTPUT_DIR}"
echo "Checkpoints: ${CHECKPOINT_DIR}"
echo "Instruction: ${LANGUAGE_INSTRUCTION}"
echo

cd "${REPO_ROOT}"

python mimic-video/data_preprocessing/action/process_lerobot.py \
  --repo-id "${DATASET_NAME}" \
  --root "${INPUT_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --language-instruction "${LANGUAGE_INSTRUCTION}" \
  --overwrite

echo "=== Precomputing T5 language embeddings ==="

conda deactivate
source "${MODEL_DIR}/.venv/bin/activate"

export PATH="/sbin:/usr/sbin:${PATH}"
export CUDA_HOME="${MODEL_DIR}/.venv/lib/python3.10/site-packages/nvidia/cuda_nvrtc"
export CUDA_PATH="${CUDA_HOME}"
export LD_LIBRARY_PATH="${MODEL_DIR}/.venv/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib:${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${MODEL_DIR}/.venv/lib/python3.10/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"
export COSMOS_PREDICT2_ARGS="${COSMOS_PREDICT2_ARGS:---checkpoints ${CHECKPOINT_DIR}}"

python mimic-video/data_preprocessing/action/precompute_t5.py \
  --dataset-path "${OUTPUT_DIR}" \
  --prompt "${LANGUAGE_INSTRUCTION}"

echo "=== Conversion + T5 Precompute Complete ==="
