#!/bin/bash
#
# Standalone training script for the mimic-video action decoder.
# Runs on a single machine with one or more GPUs.
#
# Can also be invoked by slurm_scripts/train_mimic_video.sbatch for SLURM
# multi-node runs — the sbatch wrapper sets MASTER_ADDR, NNODES, and
# GPUS_PER_NODE before calling this script via srun.
#
# PIPELINE OVERVIEW
#   1. Run scripts/process_lerobot.sh (or slurm_scripts/process_lerobot.sbatch)
#      to convert your LeRobot v3 dataset to per-episode .zarr files with T5
#      language embeddings.
#   2. Run this script. Set EXPERIMENT to the auto-registered name from
#      cosmos_predict2/configs/experiment/world2action.py for the "lerobot"
#      data_config (the grid enumerates names of the form
#      w2a_lerobot_<video_ckpt>_lr<...>_layer20_bsz<...>).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODEL_DIR="${REPO_ROOT}/mimic-video/model"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${REPO_ROOT}/mimic-video/model/checkpoints}"

EXPERIMENT="${EXPERIMENT:-w2a_lerobot_iter_000000375_fused_lr1.000e-04_layer20_bsz128}"

if [[ "${EXPERIMENT}" == *"v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043_fused"* ]]; then
    VIDEO_DIT_PATH="${VIDEO_DIT_PATH:-${CHECKPOINT_DIR}/video_backbone/v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043_fused.pt}"
elif [[ "${EXPERIMENT}" == *"iter_000000375_fused"* ]]; then
    VIDEO_DIT_PATH="${VIDEO_DIT_PATH:-${CHECKPOINT_DIR}/video_backbone/iter_000000375_fused.pt}"
else
    VIDEO_DIT_PATH="${VIDEO_DIT_PATH:-${CHECKPOINT_DIR}/video_backbone/v2w_pretrained_cosmos.pt}"
fi

# MimicDataset finds episodes via glob("**/*.zarr") under MIMIC_VIDEO_DATASET_DIR
export MIMIC_VIDEO_DATASET_DIR="${MIMIC_VIDEO_DATASET_DIR:-${REPO_ROOT}/data}"

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

# Action decoder architecture — author defaults from world2action_pipe.py.
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
GRAD_ACCUM_ITER="${GRAD_ACCUM_ITER:-4}" # author: 1

WANDB__SERVICE_WAIT="${WANDB__SERVICE_WAIT:-120}"
WANDB_START_METHOD="${WANDB_START_METHOD:-thread}"

# Distributed config. NNODES/MASTER_ADDR/GPUS_PER_NODE are exported by the
# SLURM wrapper when running on a cluster. Single-machine defaults are 1 node,
# localhost rendezvous, and all visible GPUs.
NNODES="${NNODES:-${SLURM_NNODES:-1}}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-12341}"
GPUS_PER_NODE="${GPUS_PER_NODE:-$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || echo 1)}"

# Job ID for naming — use SLURM job ID when available, else process ID.
JOB_ID="${SLURM_JOB_ID:-$$}"

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
export LD_LIBRARY_PATH="${MODEL_DIR}/.venv/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib:${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${MODEL_DIR}/.venv/lib/python3.10/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"
echo "NVRTC lib path: ${MODEL_DIR}/.venv/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib"

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
echo "Nodes:       ${NNODES}"
echo "GPUs/node:   ${GPUS_PER_NODE}"
echo "Local batch: ${TRAIN_LOCAL_BATCH_SIZE}"
echo "Grad accum:  ${GRAD_ACCUM_ITER}"
echo "Workers:     train=${DATALOADER_NUM_WORKERS}, val=${VAL_DATALOADER_NUM_WORKERS}, stats=${MIMIC_STATS_NUM_WORKERS}"
echo "Prefetch:    train=${DATALOADER_PREFETCH_FACTOR}"
echo "Rendezvous:  ${MASTER_ADDR}:${MASTER_PORT}"
echo

mkdir -p "${OUTPUT_DIR}"

torchrun \
    --nnodes="${NNODES}" \
    --nproc_per_node="${GPUS_PER_NODE}" \
    --rdzv_id="${JOB_ID}" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
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
       dataloader_train.num_workers="${DATALOADER_NUM_WORKERS}" \
       dataloader_train.prefetch_factor="${DATALOADER_PREFETCH_FACTOR}" \
       dataloader_train.persistent_workers="${DATALOADER_PERSISTENT_WORKERS}" \
       dataloader_val.num_workers="${VAL_DATALOADER_NUM_WORKERS}" \
       job.name="${EXPERIMENT}_${JOB_ID}"

echo "=== Training Complete ==="
