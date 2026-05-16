#!/bin/bash
#
# Upload a mimic-video training run (scripts/train_mimic_video.sh) to a
# private HF repo so it can later be restored with download_checkpoints_hf.sh
# for inference or to resume training.
#
# Edit RUN_DIR below per run. RUN_DIR is the leaf containing checkpoints/,
# config.pkl, config.yaml — for the train_mimic_video.sh layout that's:
#   ${OUTPUT_DIR}/vam/lerobot/${EXPERIMENT}_${JOB_ID}

set -euo pipefail

RUN_DIR="/ephemeral/robot_learning_project/runs/mimic_video/w2a_lerobot_iter_000000375_fused_lr1.000e-04_layer20_bsz128_20260514_213105/vam/lerobot/w2a_lerobot_iter_000000375_fused_lr1.000e-04_layer20_bsz128_880873"
CKPT_DIR="$RUN_DIR/checkpoints"
DATA_DIR="/ephemeral/robot_learning_project/staging/mimic-video"
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
hf repos create "$HF_REPO" \
    --type model \
    --private \
    --exist-ok

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

# Config and resume pointer. config.yaml is also what
# deployment/serve_mimic_video.sh reads to rebuild the action net.
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

# Training data. .statistics_cache/ inside this tree is what
# serve_mimic_video.sh falls back to for normalization stats.
echo "==> Uploading training data from $DATA_DIR"
hf upload \
    "$HF_REPO" \
    "$DATA_DIR" \
    data \
    --repo-type model

echo "==> Done. View at: https://huggingface.co/$HF_REPO"
