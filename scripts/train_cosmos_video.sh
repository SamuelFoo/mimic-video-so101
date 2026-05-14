#!/bin/bash
#
# Standalone training script for finetuning the Cosmos-Predict2 video backbone
# (Video2World) on a custom dataset. Runs on a single machine with one or more
# GPUs.
#
# PIPELINE OVERVIEW
#   1. Run ./scripts/process_lerobot.sh to produce <DATASET_NAME>-zarr/.
#   2. Run ./scripts/prepare_video_finetune_data.sh to convert the zarrs into
#      the Cosmos Video2World finetuning layout (video/ + metas/ + t5_xxl/).
#   3. Make sure the corresponding entry in
#        mimic-video/model/cosmos_predict2/configs/defaults/data_video.py
#      points at <FINETUNE_DATA_DIR>. The grid in
#      cosmos_predict2/configs/experiment/video2world.py auto-registers an
#      experiment named:
#        v2w_<dataset>_lora_rank256_lr1.778e-04_bsz32
#   4. Run this script. Set EXPERIMENT to override the default experiment.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODEL_DIR="${REPO_ROOT}/mimic-video/model"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${MODEL_DIR}/checkpoints}"

# ---- Arguments ------------------------------------------------------------
# EX_TYPE selects the dataset registered in
# cosmos_predict2/configs/defaults/data_video.py. Valid values today:
#   ex1        -> ex1_merged       (data/ex1_merged-cosmos-video)
#   ex2        -> ex2_merged       (data/ex2_merged-cosmos-video)
#   ex1_ex2    -> ex1_ex2_merged   (MultiDataset mixing ex1 + ex2)
EX_TYPE="${EX_TYPE:-ex1_ex2}"
DATASET_NAME="${DATASET_NAME:-${EX_TYPE}_merged}"
# LoRA rank — must be one of the values in `ranks` in
# cosmos_predict2/configs/experiment/video2world.py. Lower rank = less capacity
# (helps with overfitting)
LORA_RANK="${LORA_RANK:-32}"

# The video backbone is loaded from
#   ${CHECKPOINT_DIR}/video_backbone/v2w_pretrained_cosmos.pt
# (see imaginaire/constants.py:get_cosmos_predict2_video2world_checkpoint).
# To start from a different checkpoint, replace that file (e.g. via symlink)
# rather than overriding through hydra — the model config is a struct that
# rejects new keys for model_manager_config.
VIDEO_DIT_PATH="${CHECKPOINT_DIR}/video_backbone/v2w_pretrained_cosmos.pt"

TIMESTAMP="${TIMESTAMP:-$(date +%Y-%m-%d_%H-%M-%S)}"

# Trainer — defaults mirror cosmos_predict2/configs/experiment/video2world.py.
MAX_ITER="${MAX_ITER:-1000000}"      # author: 1_000_000
LOGGING_ITER="${LOGGING_ITER:-1}" # author: 1_000
GRAD_ACCUM_ITER="${GRAD_ACCUM_ITER:-16}"  # author: 1
TRAIN_LOCAL_BATCH_SIZE="${TRAIN_LOCAL_BATCH_SIZE:-2}" # author: 32
# Checkpoint cadence (the upstream "boundary window" auto-save in
# imaginaire/trainer.py has been removed, so this is the only schedule).
SAVE_ITER="${SAVE_ITER:-5}"
# Optional learning-rate override. Unset = use the experiment grid's LR
# (encoded in the experiment name, e.g. 1.778e-04 for the default).
LR="${LR:-5.623e-05}"
EXPERIMENT="${EXPERIMENT:-v2w_${DATASET_NAME}_lora_rank${LORA_RANK}_lr${LR}_bsz32}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/runs/cosmos_video/${EXPERIMENT}_${TIMESTAMP}}"

# Wandb
WANDB_ENABLED="${WANDB_ENABLED:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-cosmos-video-finetune}"
WANDB_ENTITY="${WANDB_ENTITY:-robot-learning-wm}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_LOG_EVERY_N="${WANDB_LOG_EVERY_N:-${LOGGING_ITER}}"
WANDB_DIR="${WANDB_DIR:-${OUTPUT_DIR}/wandb}"
WANDB__SERVICE_WAIT="${WANDB__SERVICE_WAIT:-120}"
WANDB_START_METHOD="${WANDB_START_METHOD:-thread}"

# Distributed
NNODES="${NNODES:-${SLURM_NNODES:-1}}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-12342}"
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
export LD_LIBRARY_PATH="${MODEL_DIR}/.venv/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib:${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${MODEL_DIR}/.venv/lib/python3.10/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"
export COSMOS_PREDICT2_ARGS="${COSMOS_PREDICT2_ARGS:---checkpoints ${CHECKPOINT_DIR}}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_FUSED_ATTN=0

cd "${MODEL_DIR}"

echo "=== Cosmos Video2World Finetuning ==="
echo "Node:        $(hostname)"
echo "Experiment:  ${EXPERIMENT}"
echo "LoRA rank:   ${LORA_RANK}"
echo "Video ckpt:  ${VIDEO_DIT_PATH}"
echo "Output dir:  ${OUTPUT_DIR}"
echo "WandB:       enabled=${WANDB_ENABLED}, project=${WANDB_PROJECT}, mode=${WANDB_MODE}"
echo "Nodes:       ${NNODES}"
echo "GPUs/node:   ${GPUS_PER_NODE}"
echo "Max iter:    ${MAX_ITER}"
echo "Logging:     every ${LOGGING_ITER} iter"
echo "Save:        every ${SAVE_ITER} iter"
echo "Grad accum:  ${GRAD_ACCUM_ITER}"
echo "Local bsz:   ${TRAIN_LOCAL_BATCH_SIZE:-<grid: global_bsz/world_size>}"
echo "LR:          ${LR}"
echo "Rendezvous:  ${MASTER_ADDR}:${MASTER_PORT}"
echo

extra_args=()
if [[ -n "${TRAIN_LOCAL_BATCH_SIZE}" ]]; then
    extra_args+=("dataloader_train.batch_size=${TRAIN_LOCAL_BATCH_SIZE}")
fi

torchrun \
    --nnodes="${NNODES}" \
    --nproc_per_node="${GPUS_PER_NODE}" \
    --rdzv_id="${JOB_ID}" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    -m scripts.train \
    --config=cosmos_predict2/configs/config.py \
    -- experiment="${EXPERIMENT}" \
       trainer.max_iter="${MAX_ITER}" \
       trainer.logging_iter="${LOGGING_ITER}" \
       trainer.grad_accum_iter="${GRAD_ACCUM_ITER}" \
       checkpoint.save_iter="${SAVE_ITER}" \
       optimizer.lr="${LR}" \
       trainer.callbacks.wandb.enabled="${WANDB_ENABLED}" \
       trainer.callbacks.wandb.project="${WANDB_PROJECT}" \
       trainer.callbacks.wandb.entity="${WANDB_ENTITY}" \
       trainer.callbacks.wandb.mode="${WANDB_MODE}" \
       trainer.callbacks.wandb.log_every_n="${WANDB_LOG_EVERY_N}" \
       job.name="${EXPERIMENT}_${JOB_ID}" \
       "${extra_args[@]}"

echo "=== Training Complete ==="
echo "Output: ${OUTPUT_DIR}"
