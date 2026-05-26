#!/bin/bash
#
# Precompute the VAE latents for every training and validation sample.
#
# Mirrors the env setup of train_mimic_video.sh, then runs
# scripts.precompute_video_latents once on a single GPU. Outputs go to
# ${MIMIC_VIDEO_DATASET_DIR}/.latent_cache/{train,val}_<stats_id>.pt and are
# auto-detected by MimicDataset on the next training run.
#
# Run this once after any change that invalidates the dataset (new episodes,
# different transform pipeline, different policy_io). If the cache files
# already exist for the current stats_id the script will skip re-encoding.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODEL_DIR="${REPO_ROOT}/mimic-video/model"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${REPO_ROOT}/mimic-video/model/checkpoints}"

EXPERIMENT="${EXPERIMENT:-w2a_lerobot_iter_000001410_fused_lr1.000e-04_layer20_bsz128}"

# Must match train_mimic_video.sh — MimicDataset globs **/*.zarr under this dir.
export MIMIC_VIDEO_DATASET_DIR="${MIMIC_VIDEO_DATASET_DIR:-${REPO_ROOT}/staging/mimic-video}"
export LATENT_PRECOMPUTE_BATCH_SIZE="${LATENT_PRECOMPUTE_BATCH_SIZE:-2}"
export LATENT_PRECOMPUTE_NUM_WORKERS="${LATENT_PRECOMPUTE_NUM_WORKERS:-12}"

export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${HOME}/.cache}"
export HF_HOME="${HF_HOME:-${XDG_CACHE_HOME}/huggingface}"
export TORCH_HOME="${TORCH_HOME:-${XDG_CACHE_HOME}/torch}"
mkdir -p "${HF_HOME}" "${TORCH_HOME}"

source "${MODEL_DIR}/.venv/bin/activate"

export PATH="/sbin:/usr/sbin:${PATH}"
export CUDA_HOME="${MODEL_DIR}/.venv/lib/python3.10/site-packages/nvidia/cuda_nvrtc"
export CUDA_PATH="${CUDA_HOME}"
export LD_LIBRARY_PATH="${MODEL_DIR}/.venv/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib:${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${MODEL_DIR}/.venv/lib/python3.10/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"
# distributed.init() does ctypes.CDLL("libcudart.so") for the L2-fetch tweak.
export LD_LIBRARY_PATH="${MODEL_DIR}/.venv/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:${LD_LIBRARY_PATH:-}"

export COSMOS_PREDICT2_ARGS="${COSMOS_PREDICT2_ARGS:---checkpoints ${CHECKPOINT_DIR}}"

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_FUSED_ATTN=0
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cd "${MODEL_DIR}"

# torchrun is required because imaginaire's distributed.init() calls
# init_process_group(init_method="env://"), which needs RANK / WORLD_SIZE /
# MASTER_ADDR set even for single-GPU runs. NPROC_PER_NODE defaults to every
# GPU on the node; set it explicitly to override.
NPROC_PER_NODE="${NPROC_PER_NODE:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"

echo "=== Precompute VAE latents ==="
echo "Node:        $(hostname)"
echo "Experiment:  ${EXPERIMENT}"
echo "Dataset:     ${MIMIC_VIDEO_DATASET_DIR}"
echo "GPUs:        ${NPROC_PER_NODE}"
echo

torchrun \
    --nnodes=1 \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --rdzv_id="precompute_$$" \
    --rdzv_backend=c10d \
    --rdzv_endpoint="localhost:12342" \
    -m scripts.precompute_video_latents \
    --config=cosmos_predict2/configs/config.py \
    -- experiment="${EXPERIMENT}"

echo "=== Done. Caches written under ${MIMIC_VIDEO_DATASET_DIR}/.latent_cache/ ==="
