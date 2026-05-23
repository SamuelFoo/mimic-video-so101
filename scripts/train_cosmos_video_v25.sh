#!/bin/bash
#
# Standalone training script for finetuning the Cosmos-Predict2.5 video backbone
# (Video2World) on custom robot data in the mimic-video layout.
# Runs on a single machine with one or more GPUs.
#
# PIPELINE OVERVIEW
#   1. Run ./scripts/prepare_video_finetune_data.sh to convert zarrs into the
#      Cosmos Video2World finetuning layout (video/ + t5_xxl/).
#   2. Make sure the dataset paths in
#        cosmos-predict2.5/cosmos_predict2/experiments/base/mimic_video.py
#      match your data root (_DATA_ROOT).
#   3. Run this script. Set EXPERIMENT to the desired config name, e.g.:
#        predict2_v2w_lora_rank32_ex1_all_v4
#        predict2_v2w_lora_rank32_ex1_ex2_ex3_merged

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODEL_DIR="${REPO_ROOT}/cosmos-predict2.5"

# ---- Arguments ------------------------------------------------------------
DATASET_NAME="${DATASET_NAME:-ex1_ex2_ex3_merged}"
LORA_RANK="${LORA_RANK:-32}"
EXPERIMENT="${EXPERIMENT:-predict2_v2w_lora_rank${LORA_RANK}_${DATASET_NAME}}"

TIMESTAMP="${TIMESTAMP:-$(date +%Y-%m-%d_%H-%M-%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/runs/cosmos_video_v25/${EXPERIMENT}_${TIMESTAMP}}"

MAX_ITER="${MAX_ITER:-1000000}"
LOGGING_ITER="${LOGGING_ITER:-10}"
GRAD_ACCUM_ITER="${GRAD_ACCUM_ITER:-1}"
TRAIN_LOCAL_BATCH_SIZE="${TRAIN_LOCAL_BATCH_SIZE:-4}"
SAVE_ITER="${SAVE_ITER:-25}"
LR="${LR:-2^(-14.5)}"

# Checkpoint — pass an explicit .pt path to warm-start from a local file;
# leave empty to download the default predict2.5 pre-trained weights.
LOAD_PATH="${LOAD_PATH:-}"

# Wandb
WANDB_PROJECT="${WANDB_PROJECT:-cosmos-video-v25-finetune}"
WANDB_ENTITY="${WANDB_ENTITY:-robot-learning-wm}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_DIR="${WANDB_DIR:-${OUTPUT_DIR}/wandb}"
WANDB__SERVICE_WAIT="${WANDB__SERVICE_WAIT:-120}"
WANDB_START_METHOD="${WANDB_START_METHOD:-thread}"

# Distributed
NNODES="${NNODES:-${SLURM_NNODES:-1}}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-12343}"
GPUS_PER_NODE="${GPUS_PER_NODE:-$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || echo 1)}"
JOB_ID="${SLURM_JOB_ID:-$$}"
# ---------------------------------------------------------------------------

export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${HOME}/.cache}"
export HF_HOME="${HF_HOME:-${XDG_CACHE_HOME}/huggingface}"
export TORCH_HOME="${TORCH_HOME:-${XDG_CACHE_HOME}/torch}"
export WANDB_PROJECT
export WANDB_ENTITY
export WANDB_MODE
export WANDB_DIR
export WANDB__SERVICE_WAIT
export WANDB_START_METHOD
export IMAGINAIRE_OUTPUT_ROOT="${OUTPUT_DIR}"
mkdir -p "${HF_HOME}" "${TORCH_HOME}" "${WANDB_DIR}" "${OUTPUT_DIR}"

source "${MODEL_DIR}/.venv/bin/activate"

export PATH="/sbin:/usr/sbin:${PATH}"
export CUDA_HOME="${MODEL_DIR}/.venv/lib/python3.10/site-packages/nvidia/cuda_nvrtc"
export CUDA_PATH="${CUDA_HOME}"
_CUDA_RUNTIME_LIB="${MODEL_DIR}/.venv/lib/python3.10/site-packages/nvidia/cuda_runtime/lib"
if [[ -f "${_CUDA_RUNTIME_LIB}/libcudart.so.12" && ! -e "${_CUDA_RUNTIME_LIB}/libcudart.so" ]]; then
    ln -sf libcudart.so.12 "${_CUDA_RUNTIME_LIB}/libcudart.so"
fi
export LD_LIBRARY_PATH="${_CUDA_RUNTIME_LIB}:${MODEL_DIR}/.venv/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib:${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${MODEL_DIR}/.venv/lib/python3.10/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"

export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_FUSED_ATTN=0

cd "${MODEL_DIR}"

echo "=== Cosmos-Predict2.5 Video2World Finetuning ==="
printf "Node:        %s\n" "$(hostname)"
printf "Experiment:  %s\n" "${EXPERIMENT}"
printf "LoRA rank:   %s\n" "${LORA_RANK}"
printf "Dataset:     %s\n" "${DATASET_NAME}"
printf "Output dir:  %s\n" "${OUTPUT_DIR}"
printf "WandB:       project=%s, mode=%s\n" "${WANDB_PROJECT}" "${WANDB_MODE}"
printf "Nodes:       %s\n" "${NNODES}"
printf "GPUs/node:   %s\n" "${GPUS_PER_NODE}"
printf "Max iter:    %s\n" "${MAX_ITER}"
printf "Save every:  %s iter\n" "${SAVE_ITER}"
printf "Grad accum:  %s\n" "${GRAD_ACCUM_ITER}"
printf "Local bsz:   %s\n" "${TRAIN_LOCAL_BATCH_SIZE}"
echo

extra_args=()
if [[ -n "${LOAD_PATH}" ]]; then
    extra_args+=("checkpoint.load_path=${LOAD_PATH}")
fi

torchrun \
    --nnodes="${NNODES}" \
    --nproc_per_node="${GPUS_PER_NODE}" \
    --rdzv_id="${JOB_ID}" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    -m scripts.train \
    --config=cosmos_predict2/_src/predict2/configs/video2world/config.py \
    -- experiment="${EXPERIMENT}" \
       trainer.max_iter="${MAX_ITER}" \
       trainer.logging_iter="${LOGGING_ITER}" \
       trainer.grad_accum_iter="${GRAD_ACCUM_ITER}" \
       checkpoint.save_iter="${SAVE_ITER}" \
       dataloader_train.batch_size="${TRAIN_LOCAL_BATCH_SIZE}" \
       job.project="${WANDB_PROJECT}" \
       job.wandb_mode="${WANDB_MODE}" \
       job.name="${EXPERIMENT}_${JOB_ID}" \
       "${extra_args[@]}"

echo "=== Training Complete ==="
echo "Output: ${OUTPUT_DIR}"
