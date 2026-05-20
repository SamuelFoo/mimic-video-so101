#!/bin/bash
#
# Training script for the mimic-video action decoder.
# Runs on a single machine with one or more GPUs.
#
# PIPELINE OVERVIEW
#   1. Run scripts/process_lerobot.sh to convert your LeRobot v3 dataset to
#      per-episode .zarr files with T5 language embeddings.
#   2. Run this script. Set EXPERIMENT to the auto-registered name from
#      cosmos_predict2/configs/experiment/world2action.py for the "lerobot"
#      data_config (the grid enumerates names of the form
#      w2a_lerobot_<video_ckpt>_lr<...>_layer20_bsz<...>).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODEL_DIR="${REPO_ROOT}/mimic-video/model"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${REPO_ROOT}/mimic-video/model/checkpoints}"

EXPERIMENT="${EXPERIMENT:-w2a_lerobot_iter_000000650_fused_lr1.000e-04_layer20_bsz128}"
VIDEO_DIT_PATH="${VIDEO_DIT_PATH:-${CHECKPOINT_DIR}/video_backbone/iter_000000650_fused.pt}"

# MimicDataset finds episodes via glob("**/*.zarr") under MIMIC_VIDEO_DATASET_DIR
export MIMIC_VIDEO_DATASET_DIR="${MIMIC_VIDEO_DATASET_DIR:-${REPO_ROOT}/staging/mimic-video}"

# Set to existing dir to resume
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/runs/mimic_video/${EXPERIMENT}_$(date +%Y%m%d_%H%M%S)}"

WANDB_ENABLED="${WANDB_ENABLED:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-mimic-video}"
WANDB_ENTITY="${WANDB_ENTITY:-robot-learning-wm}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_LOG_EVERY_N="${WANDB_LOG_EVERY_N:-1}"
WANDB_DIR="${WANDB_DIR:-${OUTPUT_DIR}/wandb}"

MAX_VAL_ITER="${MAX_VAL_ITER:-8}" # author: null (no cap); set to e.g. 8 to limit long validation at iter 0
VAL_SHUFFLE="${VAL_SHUFFLE:-True}"
VAL_NUM_SAMPLING_STEPS="${VAL_NUM_SAMPLING_STEPS:-12}"  # author: 12
VAL_RUN_GENERATED_VIDEO="${VAL_RUN_GENERATED_VIDEO:-False}"

# lerobot.yaml has num_val_episodes: 0
# skip validation to avoid iterating nothing.
RUN_VALIDATION="${RUN_VALIDATION:-False}"

SAVE_ITER="${SAVE_ITER:-50}"

# Warm-start: path to a model/iter_*.pt to initialize weights from.
# Loads model weights only; optimizer/scheduler/iteration are reset.
# Ignored if OUTPUT_DIR already contains a latest_checkpoint.txt (in-place resume wins).
LOAD_PATH="${LOAD_PATH:-/ephemeral/robot_learning_project/checkpoints/action/iter_000003750.pt}"

# Action decoder architecture — author defaults from world2action_pipe.py.
XATTN_VIDEO_PREFIX_LENGTH="${XATTN_VIDEO_PREFIX_LENGTH:-null}" # null = state_t in DiT; set to < state_t to slice
ACTION_MODEL_CHANNELS="${ACTION_MODEL_CHANNELS:-512}" # author: 1024
ACTION_MODEL_BLOCKS="${ACTION_MODEL_BLOCKS:-12}" # author: 24
ACTION_MODEL_HEADS="${ACTION_MODEL_HEADS:-8}" # author: 8
ACTION_MODEL_PAIR_TIMESTEP_FEATURE_RANK="${ACTION_MODEL_PAIR_TIMESTEP_FEATURE_RANK:-512}" # author: 1024
ACTION_MODEL_ADALN_LORA_DIM="${ACTION_MODEL_ADALN_LORA_DIM:-64}" # author: 128

# Dataloader — author defaults from data_action.py.
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-12}"           # author: 12
DATALOADER_PREFETCH_FACTOR="${DATALOADER_PREFETCH_FACTOR:-8}"    # author: 8
DATALOADER_PERSISTENT_WORKERS="${DATALOADER_PERSISTENT_WORKERS:-True}"  # author: True
VAL_DATALOADER_NUM_WORKERS="${VAL_DATALOADER_NUM_WORKERS:-0}"    # author: 0
MIMIC_STATS_NUM_WORKERS="${MIMIC_STATS_NUM_WORKERS:-8}"          # author: max(1, cpu_count() // 4)
MIMIC_STATS_BATCH_SIZE="${MIMIC_STATS_BATCH_SIZE:-8}"            # author: max(1, cpu_count() // 4)
if [[ "${DATALOADER_NUM_WORKERS}" == "0" ]]; then
    DATALOADER_PREFETCH_FACTOR="None"
    DATALOADER_PERSISTENT_WORKERS="False"
fi

# Batch size — author grid uses global_bsz / world_size per GPU. For the
# default bsz128 experiment on 1 GPU the local batch is 128; on 2 GPUs it
# would be 64. Increase EXPERIMENT to a bsz256 variant when scaling up.
TRAIN_LOCAL_BATCH_SIZE="${TRAIN_LOCAL_BATCH_SIZE:-32}" # author: global_bsz / world_size (128 for bsz128 on 1 GPU)
GRAD_ACCUM_ITER="${GRAD_ACCUM_ITER:-2}" # author: 1

WANDB__SERVICE_WAIT="${WANDB__SERVICE_WAIT:-120}"
WANDB_START_METHOD="${WANDB_START_METHOD:-thread}"

# Single-node multi-GPU. Defaults to all visible GPUs; override NUM_GPUS to
# pin a smaller subset.
NUM_GPUS="${NUM_GPUS:-$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || echo 1)}"

export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${HOME}/.cache}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${XDG_CACHE_HOME}/uv}"
export HF_HOME="${HF_HOME:-${XDG_CACHE_HOME}/huggingface}"
export TORCH_HOME="${TORCH_HOME:-${XDG_CACHE_HOME}/torch}"
export WANDB_PROJECT
export WANDB_ENTITY
export WANDB_MODE
export WANDB_DIR
export WANDB__SERVICE_WAIT
export WANDB_START_METHOD
export MIMIC_STATS_NUM_WORKERS
export MIMIC_STATS_BATCH_SIZE
export IMAGINAIRE_OUTPUT_ROOT="${OUTPUT_DIR}"
mkdir -p "${UV_CACHE_DIR}" "${HF_HOME}" "${TORCH_HOME}" "${WANDB_DIR}"

source "${MODEL_DIR}/.venv/bin/activate"

export PATH="/sbin:/usr/sbin:${PATH}"
export CUDA_HOME="${MODEL_DIR}/.venv/lib/python3.10/site-packages/nvidia/cuda_nvrtc"
export CUDA_PATH="${CUDA_HOME}"
_CUDA_RUNTIME_LIB="${MODEL_DIR}/.venv/lib/python3.10/site-packages/nvidia/cuda_runtime/lib"
# dlopen("libcudart.so") needs the unversioned name; create symlink if missing
if [[ -f "${_CUDA_RUNTIME_LIB}/libcudart.so.12" && ! -e "${_CUDA_RUNTIME_LIB}/libcudart.so" ]]; then
    ln -sf libcudart.so.12 "${_CUDA_RUNTIME_LIB}/libcudart.so"
fi
export LD_LIBRARY_PATH="${_CUDA_RUNTIME_LIB}:${MODEL_DIR}/.venv/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib:${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${MODEL_DIR}/.venv/lib/python3.10/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH}"
echo "NVRTC lib path: ${MODEL_DIR}/.venv/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib"
echo "CUDART lib path: ${_CUDA_RUNTIME_LIB}"

export COSMOS_PREDICT2_ARGS="${COSMOS_PREDICT2_ARGS:---checkpoints ${CHECKPOINT_DIR}}"

export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_FUSED_ATTN=0

cd "${MODEL_DIR}"

echo "=== mimic-video Action Decoder Training ==="
echo "Node:        $(hostname)"
echo "Experiment:  ${EXPERIMENT}"
echo "Checkpoints: ${CHECKPOINT_DIR}"
echo "Video ckpt:  ${VIDEO_DIT_PATH}"
echo "Dataset dir: ${MIMIC_VIDEO_DATASET_DIR}"
echo "Output dir:  ${OUTPUT_DIR}"
echo "WandB:       enabled=${WANDB_ENABLED}, project=${WANDB_PROJECT}, mode=${WANDB_MODE}"
echo "GPUs:        ${NUM_GPUS}"
echo "Local batch: ${TRAIN_LOCAL_BATCH_SIZE}"
echo "Grad accum:  ${GRAD_ACCUM_ITER}"
echo "Workers:     train=${DATALOADER_NUM_WORKERS}, val=${VAL_DATALOADER_NUM_WORKERS}, stats=${MIMIC_STATS_NUM_WORKERS}"
echo "Prefetch:    train=${DATALOADER_PREFETCH_FACTOR}"
echo

mkdir -p "${OUTPUT_DIR}"

torchrun \
    --standalone \
    --nproc_per_node="${NUM_GPUS}" \
    -m scripts.train \
    --config=cosmos_predict2/configs/config.py \
    -- experiment="${EXPERIMENT}" \
       model.config.video_dit_path="${VIDEO_DIT_PATH}" \
       trainer.callbacks.wandb.enabled="${WANDB_ENABLED}" \
       trainer.callbacks.wandb.project="${WANDB_PROJECT}" \
       trainer.callbacks.wandb.entity="${WANDB_ENTITY}" \
       trainer.callbacks.wandb.mode="${WANDB_MODE}" \
       trainer.callbacks.wandb.log_every_n="${WANDB_LOG_EVERY_N}" \
       trainer.max_val_iter="${MAX_VAL_ITER}" \
       trainer.run_validation="${RUN_VALIDATION}" \
       checkpoint.save_iter="${SAVE_ITER}" \
       checkpoint.load_path="${LOAD_PATH}" \
       dataloader_val.sampler.shuffle="${VAL_SHUFFLE}" \
       model.config.validation_num_sampling_steps="${VAL_NUM_SAMPLING_STEPS}" \
       model.config.validation_run_generated_video="${VAL_RUN_GENERATED_VIDEO}" \
       dataloader_train.batch_size="${TRAIN_LOCAL_BATCH_SIZE}" \
       trainer.grad_accum_iter="${GRAD_ACCUM_ITER}" \
       world2action_pipe.net.model_channels="${ACTION_MODEL_CHANNELS}" \
       world2action_pipe.net.num_blocks="${ACTION_MODEL_BLOCKS}" \
       world2action_pipe.net.num_heads="${ACTION_MODEL_HEADS}" \
       world2action_pipe.net.pair_timestep_feature_rank="${ACTION_MODEL_PAIR_TIMESTEP_FEATURE_RANK}" \
       world2action_pipe.net.adaln_lora_dim="${ACTION_MODEL_ADALN_LORA_DIM}" \
       model.config.pipe_config.net.model_channels="${ACTION_MODEL_CHANNELS}" \
       model.config.pipe_config.net.num_blocks="${ACTION_MODEL_BLOCKS}" \
       model.config.pipe_config.net.num_heads="${ACTION_MODEL_HEADS}" \
       model.config.pipe_config.net.pair_timestep_feature_rank="${ACTION_MODEL_PAIR_TIMESTEP_FEATURE_RANK}" \
       model.config.pipe_config.net.adaln_lora_dim="${ACTION_MODEL_ADALN_LORA_DIM}" \
       world2action_pipe.xattn_video_prefix_length="${XATTN_VIDEO_PREFIX_LENGTH}" \
       model.config.pipe_config.xattn_video_prefix_length="${XATTN_VIDEO_PREFIX_LENGTH}" \
       dataloader_train.num_workers="${DATALOADER_NUM_WORKERS}" \
       dataloader_train.prefetch_factor="${DATALOADER_PREFETCH_FACTOR}" \
       dataloader_train.persistent_workers="${DATALOADER_PERSISTENT_WORKERS}" \
       dataloader_val.num_workers="${VAL_DATALOADER_NUM_WORKERS}" \
       job.name="${EXPERIMENT}"

echo "=== Training Complete ==="
