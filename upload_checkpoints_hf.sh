#!/bin/bash
set -euo pipefail

RUN_DIR="/ephemeral/robot_learning_project/runs/cosmos_video/v2w_ex1_ex2_merged_lora_rank32_lr5.623e-05_bsz32_2026-05-13_21-52-49/posttraining/video2world/v2w_ex1_ex2_merged_lora_rank32_lr5.623e-05_bsz32_35213"
CKPT_DIR="$RUN_DIR/checkpoints"
DATA_DIR="/ephemeral/robot_learning_project/data"
N_CHECKPOINTS=5
HF_ORG="robot-learning"
CONDA_ENV="${CONDA_ENV:-lerobot}"

RUN_NAME=$(basename "$RUN_DIR")
HF_REPO="$HF_ORG/$RUN_NAME"

echo "==> Activating conda environment: $CONDA_ENV"
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

echo "==> Installing dependencies"
pip install -q huggingface_hub hf_transfer

export HF_HUB_ENABLE_HF_TRANSFER=1

echo "==> Creating private repo: $HF_REPO"
hf repo create "$RUN_NAME" \
    --type model \
    --organization "$HF_ORG" \
    --private 2>&1 || echo "(repo may already exist, continuing)"

# Collect the last N unique iteration numbers from the model subdir
mapfile -t ITERS < <(
    ls "$CKPT_DIR/model/" \
    | grep -oP 'iter_\d+(?=\.pt$)' \
    | sort -u \
    | tail -"$N_CHECKPOINTS"
)

echo "==> Uploading last $N_CHECKPOINTS checkpoints: ${ITERS[*]}"

INCLUDE_ARGS=()
for iter in "${ITERS[@]}"; do
    for subdir in model optim scheduler trainer; do
        # glob matches both iter_XXXXXX.pt and iter_XXXXXX_fused.pt
        INCLUDE_ARGS+=(--include "checkpoints/${subdir}/${iter}*")
    done
done

# Config and resume pointer
INCLUDE_ARGS+=(--include "checkpoints/latest_checkpoint.txt")
INCLUDE_ARGS+=(--include "config.pkl")
INCLUDE_ARGS+=(--include "config.yaml")

echo "==> Starting upload to $HF_REPO"
hf upload \
    "$HF_REPO" \
    "$RUN_DIR" \
    . \
    --repo-type model \
    "${INCLUDE_ARGS[@]}"

echo "==> Uploading training data (~820 MB)"
hf upload \
    "$HF_REPO" \
    "$DATA_DIR" \
    data \
    --repo-type model

echo "==> Done. View at: https://huggingface.co/$HF_REPO"
