#!/bin/bash
#
# SLURM batch script for converting a LeRobot v3 dataset to mimic-video's
# per-episode .zarr layout via process_lerobot.py.

#SBATCH --job-name=process-lerobot
#SBATCH --output=/cluster/scratch/samfoo/slurm_runs/slurm-process-lerobot-%j.out
#SBATCH --error=/cluster/scratch/samfoo/slurm_runs/slurm-process-lerobot-%j.err
#SBATCH --nodes=1
#SBATCH --gpus-per-node=rtx_3090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --tmp=200G
#SBATCH --time=4:00:00

# NOTE: The lerobot conda env needs zarr<3 / numcodecs<0.16 installed

set -euo pipefail

REPO_ROOT="/cluster/home/samfoo/workspaces/robot_learning_project"
MODEL_DIR="${REPO_ROOT}/mimic-video/model"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/cluster/work/rsl/samfoo/robot_learning/mimic-video/checkpoints}"

CONDA_ENV="${CONDA_ENV:-lerobot}"
DATASET_NAME="${DATASET_NAME:-ex1_merged}"
DATA_ROOT="${DATA_ROOT:-/cluster/scratch/samfoo/robot_learning/data}"
INPUT_DIR="${INPUT_DIR:-${DATA_ROOT}/${DATASET_NAME}}"
OUTPUT_DIR="${OUTPUT_DIR:-${DATA_ROOT}/${DATASET_NAME}-zarr}"
LANGUAGE_INSTRUCTION="${LANGUAGE_INSTRUCTION:-Push a small object from left to right along a straight line. The start and end regions are marked by a white circle of 5 cm radius. The object must stay within the corridor marked by the white horizontal lines. Complete the task in as little pushes as possible. A push is counted every time the gripper makes contact with the object and then leaves it, regardless of whether the object moved.}"

module load stack/2024-06
module load cuda/12.4.1
module load eth_proxy

source /cluster/project/rsl/$USER/miniconda3/bin/activate
conda activate "${CONDA_ENV}"

# Required by torchcodec when reading lerobot v3 video frames.
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

echo "=== LeRobot -> zarr conversion ==="
echo "Node:        $(hostname)"
echo "Conda env:   ${CONDA_ENV}"
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

# Some Euler job environments do not expose ldconfig, so point Transformer
# Engine/CUDA discovery at the CUDA runtime libraries in the Python environment.
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
