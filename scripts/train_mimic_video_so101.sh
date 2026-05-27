#!/bin/bash
#
# Fine-tune the action decoder with the mimic-video-so101 codebase.
#
# DIFFERENCES FROM train_mimic_video.sh
#   • MODEL_DIR   → mimic-video-so101/model  (safetensors dataset reader)
#   • DATASET     → staging/mimic-video-st   (.safetensors episodes)
#   • LOAD_PATH   → auto-detected latest model/iter_*.pt in runs/
#                   (the zarr-trained checkpoint, loaded as a warm-start)
#   • OUTPUT_DIR  → runs/mimic_video_so101/  (separate from zarr runs)
#   • VENV        → shared from mimic-video/model/.venv
#
# QUICK START
#   bash scripts/train_mimic_video_so101.sh
#
# RESUME an existing so101 run exactly where it left off:
#   OUTPUT_DIR=runs/mimic_video_so101/<run_dir> bash scripts/train_mimic_video_so101.sh
#   (latest_checkpoint.txt inside OUTPUT_DIR makes the trainer resume in-place;
#    LOAD_PATH is then ignored automatically.)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# so101 source code lives here; venv is shared from mimic-video.
MODEL_DIR="${REPO_ROOT}/mimic-video-so101/model"
VENV_DIR="${REPO_ROOT}/mimic-video/model/.venv"

# Video backbone checkpoints (shared with mimic-video; no separate so101 download needed).
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${REPO_ROOT}/mimic-video/model/checkpoints}"
VIDEO_DIT_PATH="${VIDEO_DIT_PATH:-${CHECKPOINT_DIR}/video_backbone/iter_000001410_fused.pt}"

# Experiment — must match a name registered in so101's world2action.py.
EXPERIMENT="${EXPERIMENT:-w2a_lerobot_iter_000001410_fused_lr1.000e-04_layer20_bsz128}"

# Safetensors dataset (produced by scripts/convert_zarr_to_safetensors.py).
# Latent cache must already exist under ${MIMIC_VIDEO_DATASET_DIR}/.latent_cache/
# (produced by scripts/precompute_video_latents_so101.sh).
export MIMIC_VIDEO_DATASET_DIR="${MIMIC_VIDEO_DATASET_DIR:-${REPO_ROOT}/staging/mimic-video-st}"

# Warm-start: path to a model/iter_*.pt to initialize weights from.
# Loads model weights only; optimizer/scheduler/iteration are reset.
# Ignored if OUTPUT_DIR already contains a latest_checkpoint.txt (in-place resume wins).
LOAD_PATH="${LOAD_PATH:-/ephemeral/robot_learning_project/runs/mimic_video/w2a_lerobot_iter_000001410_fused_lr1.000e-04_layer20_bsz128_20260527_085926/vam/lerobot/w2a_lerobot_iter_000001410_fused_lr1.000e-04_layer20_bsz128/checkpoints/model/iter_000004900.pt}"


# ── Output ──────────────────────────────────────────────────────────────────
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/runs/mimic_video_so101/${EXPERIMENT}_$(date +%Y%m%d_%H%M%S)}"

# ── Weights & Biases ────────────────────────────────────────────────────────
# so101's WandBCallback uses mode/project_name/entity_name (no "enabled" bool).
# Set WANDB_MODE=disabled to turn off wandb entirely.
WANDB_PROJECT="${WANDB_PROJECT:-mimic-video}"
WANDB_ENTITY="${WANDB_ENTITY:-robot-learning-wm}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_DIR="${WANDB_DIR:-${OUTPUT_DIR}/wandb}"

# ── Validation ──────────────────────────────────────────────────────────────
MAX_VAL_ITER="${MAX_VAL_ITER:-8}"
VAL_SHUFFLE="${VAL_SHUFFLE:-True}"
# lerobot.yaml has num_val_episodes: 0 — skip validation (nothing to iterate).
RUN_VALIDATION="${RUN_VALIDATION:-False}"

# ── Checkpointing ───────────────────────────────────────────────────────────
SAVE_ITER="${SAVE_ITER:-100}"
STRICT_RESUME="${STRICT_RESUME:-True}"

# ── Action decoder architecture ─────────────────────────────────────────────
# These must match the dimensions used when training LOAD_PATH; changing them
# here would make the warm-start fail with a shape mismatch.
# Note: so101 has no xattn_video_prefix_length field (removed vs mimic-video).
ACTION_MODEL_CHANNELS="${ACTION_MODEL_CHANNELS:-512}"
ACTION_MODEL_BLOCKS="${ACTION_MODEL_BLOCKS:-12}"
ACTION_MODEL_HEADS="${ACTION_MODEL_HEADS:-8}"
ACTION_MODEL_PAIR_TIMESTEP_FEATURE_RANK="${ACTION_MODEL_PAIR_TIMESTEP_FEATURE_RANK:-512}"
ACTION_MODEL_ADALN_LORA_DIM="${ACTION_MODEL_ADALN_LORA_DIM:-64}"

# ── Dataloader ──────────────────────────────────────────────────────────────
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-12}"
DATALOADER_PREFETCH_FACTOR="${DATALOADER_PREFETCH_FACTOR:-8}"
DATALOADER_PERSISTENT_WORKERS="${DATALOADER_PERSISTENT_WORKERS:-True}"
VAL_DATALOADER_NUM_WORKERS="${VAL_DATALOADER_NUM_WORKERS:-0}"
export MIMIC_STATS_NUM_WORKERS="${MIMIC_STATS_NUM_WORKERS:-8}"
export MIMIC_STATS_BATCH_SIZE="${MIMIC_STATS_BATCH_SIZE:-8}"
if [[ "${DATALOADER_NUM_WORKERS}" == "0" ]]; then
    DATALOADER_PREFETCH_FACTOR="None"
    DATALOADER_PERSISTENT_WORKERS="False"
fi

# ── Batch size ───────────────────────────────────────────────────────────────
# For bsz128 on 8 GPUs: local batch = 128 / 8 = 16.  Adjust TRAIN_LOCAL_BATCH_SIZE
# or switch to a bsz256 experiment when scaling up.
TRAIN_LOCAL_BATCH_SIZE="${TRAIN_LOCAL_BATCH_SIZE:-16}"
GRAD_ACCUM_ITER="${GRAD_ACCUM_ITER:-1}"

# ── GPUs ────────────────────────────────────────────────────────────────────
NUM_GPUS="${NUM_GPUS:-$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || echo 1)}"

# ── Paths & caches ──────────────────────────────────────────────────────────
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${HOME}/.cache}"
export HF_HOME="${HF_HOME:-${XDG_CACHE_HOME}/huggingface}"
export TORCH_HOME="${TORCH_HOME:-${XDG_CACHE_HOME}/torch}"
export WANDB_PROJECT WANDB_ENTITY WANDB_MODE WANDB_DIR
export WANDB__SERVICE_WAIT="${WANDB__SERVICE_WAIT:-120}"
export WANDB_START_METHOD="${WANDB_START_METHOD:-thread}"
export IMAGINAIRE_OUTPUT_ROOT="${OUTPUT_DIR}"
mkdir -p "${HF_HOME}" "${TORCH_HOME}" "${WANDB_DIR}" "${OUTPUT_DIR}"

# ── Activate shared venv ─────────────────────────────────────────────────────
source "${VENV_DIR}/bin/activate"

# ── CUDA library paths (from shared venv) ────────────────────────────────────
export PATH="/sbin:/usr/sbin:${PATH}"
export CUDA_HOME="${VENV_DIR}/lib/python3.10/site-packages/nvidia/cuda_nvrtc"
export CUDA_PATH="${CUDA_HOME}"
_CUDA_RUNTIME_LIB="${VENV_DIR}/lib/python3.10/site-packages/nvidia/cuda_runtime/lib"
if [[ -f "${_CUDA_RUNTIME_LIB}/libcudart.so.12" && ! -e "${_CUDA_RUNTIME_LIB}/libcudart.so" ]]; then
    ln -sf libcudart.so.12 "${_CUDA_RUNTIME_LIB}/libcudart.so"
fi
export LD_LIBRARY_PATH="${_CUDA_RUNTIME_LIB}:${VENV_DIR}/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib:${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${VENV_DIR}/lib/python3.10/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH}"

# ── Cosmos / NCCL settings ───────────────────────────────────────────────────
export COSMOS_PREDICT2_ARGS="${COSMOS_PREDICT2_ARGS:---checkpoints ${CHECKPOINT_DIR}}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_FUSED_ATTN=0
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# Disable P2P DMA for NCCL ring all-reduce: on PCIe-only topologies (no NVLink)
# the ring algorithm needs bidirectional P2P transfers which can hang in VMs.
# With P2P disabled, NCCL falls back to shared-memory (SHM) transfers which are
# reliable on single-node multi-GPU without NVLink.
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"

# ── Run from so101 model directory so Python imports its cosmos_predict2 ─────
cd "${MODEL_DIR}"

echo "=== mimic-video-so101 Action Decoder Fine-tuning ==="
echo "Node:        $(hostname)"
echo "Experiment:  ${EXPERIMENT}"
echo "Video ckpt:  ${VIDEO_DIT_PATH}"
echo "Dataset:     ${MIMIC_VIDEO_DATASET_DIR}"
echo "Load path:   ${LOAD_PATH:-<none — scratch>}"
echo "Output dir:  ${OUTPUT_DIR}"
echo "WandB:       project=${WANDB_PROJECT}, entity=${WANDB_ENTITY}, mode=${WANDB_MODE}"
echo "GPUs:        ${NUM_GPUS}"
echo "Local batch: ${TRAIN_LOCAL_BATCH_SIZE}  (grad_accum=${GRAD_ACCUM_ITER})"
echo "Workers:     train=${DATALOADER_NUM_WORKERS}, val=${VAL_DATALOADER_NUM_WORKERS}, stats=${MIMIC_STATS_NUM_WORKERS}"
echo "Action decoder: channels=${ACTION_MODEL_CHANNELS}, blocks=${ACTION_MODEL_BLOCKS}, heads=${ACTION_MODEL_HEADS}"
echo

torchrun \
    --standalone \
    --nproc_per_node="${NUM_GPUS}" \
    -m scripts.train \
    --config=cosmos_predict2/configs/config.py \
    -- experiment="${EXPERIMENT}" \
       model.config.video_dit_path="${VIDEO_DIT_PATH}" \
       trainer.callbacks.wandb.mode="${WANDB_MODE}" \
       trainer.callbacks.wandb.project_name="${WANDB_PROJECT}" \
       trainer.callbacks.wandb.entity_name="${WANDB_ENTITY}" \
       trainer.max_val_iter="${MAX_VAL_ITER}" \
       trainer.run_validation="${RUN_VALIDATION}" \
       checkpoint.save_iter="${SAVE_ITER}" \
       checkpoint.load_path="${LOAD_PATH:-null}" \
       dataloader_val.sampler.shuffle="${VAL_SHUFFLE}" \
       dataloader_train.batch_size="${TRAIN_LOCAL_BATCH_SIZE}" \
       trainer.grad_accum_iter="${GRAD_ACCUM_ITER}" \
       action_pipe.net.model_channels="${ACTION_MODEL_CHANNELS}" \
       action_pipe.net.num_blocks="${ACTION_MODEL_BLOCKS}" \
       action_pipe.net.num_heads="${ACTION_MODEL_HEADS}" \
       action_pipe.net.pair_timestep_feature_rank="${ACTION_MODEL_PAIR_TIMESTEP_FEATURE_RANK}" \
       action_pipe.net.adaln_lora_dim="${ACTION_MODEL_ADALN_LORA_DIM}" \
       model.config.pipe_config.net.model_channels="${ACTION_MODEL_CHANNELS}" \
       model.config.pipe_config.net.num_blocks="${ACTION_MODEL_BLOCKS}" \
       model.config.pipe_config.net.num_heads="${ACTION_MODEL_HEADS}" \
       model.config.pipe_config.net.pair_timestep_feature_rank="${ACTION_MODEL_PAIR_TIMESTEP_FEATURE_RANK}" \
       model.config.pipe_config.net.adaln_lora_dim="${ACTION_MODEL_ADALN_LORA_DIM}" \
       dataloader_train.num_workers="${DATALOADER_NUM_WORKERS}" \
       dataloader_train.prefetch_factor="${DATALOADER_PREFETCH_FACTOR}" \
       dataloader_train.persistent_workers="${DATALOADER_PERSISTENT_WORKERS}" \
       dataloader_val.num_workers="${VAL_DATALOADER_NUM_WORKERS}" \
       checkpoint.strict_resume="${STRICT_RESUME}" \
       job.name="${EXPERIMENT}"

echo "=== Training Complete ==="
