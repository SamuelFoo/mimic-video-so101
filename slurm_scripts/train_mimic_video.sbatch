#!/bin/bash
#
# SLURM batch script for training a mimic-video action decoder on a
# preprocessed LeRobot dataset (joint-space, single front camera).
#
# Multi-node + multi-GPU friendly. To scale up, set:
#   --nodes=N
#   --gpus-per-node=type:K
# and torchrun will be launched with rendezvous on the first node.
#
# PIPELINE OVERVIEW
#   1. Run data_preprocessing/action/process_lerobot.py with the lerobot env to
#      convert your LeRobot v3 dataset to per-episode .zarr files.
#   2. Run data_preprocessing/action/precompute_t5.py with the cosmos uv env to
#      add language embeddings.
#   3. Submit this script to train. Set EXPERIMENT to the auto-registered name
#      from cosmos_predict2/configs/experiment/world2action.py for the
#      "lerobot" data_config (the grid in that file enumerates names of the
#      form  w2a_lerobot_<video_ckpt>_lr<...>_layer20_bsz<...>).

#SBATCH --job-name=train-mimic-video
#SBATCH --output=/cluster/scratch/samfoo/slurm_runs/slurm-mimic-video-%j.out
#SBATCH --error=/cluster/scratch/samfoo/slurm_runs/slurm-mimic-video-%j.err
#SBATCH --nodes=1
#SBATCH --gpus-per-node=rtx_4090:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G
#SBATCH --tmp=400G
#SBATCH --time=24:00:00

set -euo pipefail

REPO_ROOT="/cluster/home/samfoo/workspaces/robot_learning_project/mimic-video"
MODEL_DIR="${REPO_ROOT}/model"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/cluster/work/rsl/samfoo/robot_learning/mimic-video/checkpoints}"

# The experiment grid in cosmos_predict2/configs/experiment/world2action.py
# auto-registers names of the form:
#   w2a_<data_config>_<video_ckpt>_lr<...>_layer20_bsz<...>
# Override EXPERIMENT before sbatch to pick a different one (e.g. bsz=256
# when you have enough GPUs).
EXPERIMENT="${EXPERIMENT:-w2a_lerobot_v2w_pretrained_cosmos_lr1.000e-04_layer20_bsz128}"

if [[ "${EXPERIMENT}" == *"v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043_fused"* ]]; then
    VIDEO_DIT_PATH="${VIDEO_DIT_PATH:-${CHECKPOINT_DIR}/video_backbone/v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043_fused.pt}"
else
    VIDEO_DIT_PATH="${VIDEO_DIT_PATH:-${CHECKPOINT_DIR}/video_backbone/v2w_pretrained_cosmos.pt}"
fi

# Path to the directory containing the preprocessed .zarr episodes
# (output of process_lerobot.py + precompute_t5.py).
DATASET_SRC="${DATASET_SRC:-/cluster/scratch/samfoo/robot_learning/data/ex1_merged-zarr}"
DATASET_DST="${TMPDIR}/mimic_video_dataset"
export MIMIC_VIDEO_DATASET_DIR="${DATASET_DST}"

OUTPUT_DIR="${OUTPUT_DIR:-/cluster/work/rsl/samfoo/robot_learning/mimic_video_runs/${EXPERIMENT}_$(date +%Y%m%d_%H%M%S)}"

WANDB_ENABLED="${WANDB_ENABLED:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-mimic-video}"
WANDB_ENTITY="${WANDB_ENTITY:-robot-learning-wm}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_LOG_EVERY_N="${WANDB_LOG_EVERY_N:-100}"
WANDB_DIR="${WANDB_DIR:-${OUTPUT_DIR}/wandb}"
WANDB__SERVICE_WAIT="${WANDB__SERVICE_WAIT:-120}"
WANDB_START_METHOD="${WANDB_START_METHOD:-thread}"

# Distributed setup
GPUS_PER_NODE="${GPUS_PER_NODE:-$(echo "${SLURM_GPUS_ON_NODE:-${SLURM_GPUS_PER_NODE:-1}}" | tr -dc '0-9')}"
NNODES="${SLURM_NNODES:-1}"
MASTER_PORT="${MASTER_PORT:-12341}"

module load stack/2024-06
module load cuda/12.4.1
module load eth_proxy

export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/cluster/work/rsl/samfoo/cache}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${XDG_CACHE_HOME}/uv}"
export HF_HOME="${HF_HOME:-${XDG_CACHE_HOME}/huggingface}"
export TORCH_HOME="${TORCH_HOME:-${XDG_CACHE_HOME}/torch}"
export WANDB_PROJECT
export WANDB_ENTITY
export WANDB_MODE
export WANDB_DIR
export WANDB__SERVICE_WAIT
export WANDB_START_METHOD
export IMAGINAIRE_OUTPUT_ROOT="${OUTPUT_DIR}"
mkdir -p "${UV_CACHE_DIR}" "${HF_HOME}" "${TORCH_HOME}" "${WANDB_DIR}"

source "${MODEL_DIR}/.venv/bin/activate"

# Transformer Engine tries to locate libnvrtc during import. Some Euler job
# environments do not expose ldconfig, so point it directly at the CUDA runtime
# libraries installed in the Python environment.
export PATH="/sbin:/usr/sbin:${PATH}"
export CUDA_HOME="${MODEL_DIR}/.venv/lib/python3.10/site-packages/nvidia/cuda_nvrtc"
export CUDA_PATH="${CUDA_HOME}"
export LD_LIBRARY_PATH="${MODEL_DIR}/.venv/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib:${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${MODEL_DIR}/.venv/lib/python3.10/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"
echo "NVRTC lib path: ${MODEL_DIR}/.venv/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib"

# Point Cosmos-Predict2 constants at the downloaded mimic-video checkpoint root.
export COSMOS_PREDICT2_ARGS="${COSMOS_PREDICT2_ARGS:---checkpoints ${CHECKPOINT_DIR}}"

# Required by Cosmos-Predict2 / Megatron / NCCL on multi-GPU runs.
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_FUSED_ATTN=0

# Hydra writes outputs relative to CWD; pin it to MODEL_DIR like the README does.
cd "${MODEL_DIR}"

# ---------------------------------------------------------------------------
# Stage data to local SSD on every node
# ---------------------------------------------------------------------------
echo "=== mimic-video Action Decoder Training ==="
echo "Node:        $(hostname)"
echo "Experiment:  ${EXPERIMENT}"
echo "Checkpoints: ${CHECKPOINT_DIR}"
echo "Video ckpt:  ${VIDEO_DIT_PATH}"
echo "Output dir:  ${OUTPUT_DIR}"
echo "WandB:       enabled=${WANDB_ENABLED}, project=${WANDB_PROJECT}, mode=${WANDB_MODE}"
echo "Nodes:       ${NNODES}"
echo "GPUs/node:   ${GPUS_PER_NODE}"
echo

# Stage on every allocated node (one rsync per node; srun handles that).
echo "Staging dataset ${DATASET_SRC} -> ${DATASET_DST} on all nodes ..."
srun --ntasks-per-node=1 bash -c "mkdir -p '${DATASET_DST}' && rsync -a '${DATASET_SRC}/' '${DATASET_DST}/'"
echo "Dataset stage done."

mkdir -p "${OUTPUT_DIR}"

# ---------------------------------------------------------------------------
# Resolve rendezvous endpoint (head node IP) for torchrun
# ---------------------------------------------------------------------------
nodes_array=( $(scontrol show hostnames "$SLURM_JOB_NODELIST") )
head_node="${nodes_array[0]}"
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "${head_node}" hostname --ip-address)
echo "Rendezvous endpoint: ${head_node_ip}:${MASTER_PORT}"

# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
# One srun task per node; each task launches torchrun with GPUS_PER_NODE
# local workers. torchrun handles intra-node DDP and joins inter-node via
# the c10d rendezvous backend.
srun --nodes="${NNODES}" --ntasks-per-node=1 \
    torchrun \
        --nnodes="${NNODES}" \
        --nproc_per_node="${GPUS_PER_NODE}" \
        --rdzv_id="${SLURM_JOB_ID}" \
        --rdzv_backend=c10d \
        --rdzv_endpoint="${head_node_ip}:${MASTER_PORT}" \
        -m scripts.train \
        --config=cosmos_predict2/configs/config.py \
        -- experiment="${EXPERIMENT}" \
           model.config.video_dit_path="${VIDEO_DIT_PATH}" \
           trainer.callbacks.wandb.enabled="${WANDB_ENABLED}" \
           trainer.callbacks.wandb.project="${WANDB_PROJECT}" \
           trainer.callbacks.wandb.entity="${WANDB_ENTITY}" \
           trainer.callbacks.wandb.mode="${WANDB_MODE}" \
           trainer.callbacks.wandb.log_every_n="${WANDB_LOG_EVERY_N}" \
           job.name="${EXPERIMENT}_${SLURM_JOB_ID}"

echo "=== Training Complete ==="
