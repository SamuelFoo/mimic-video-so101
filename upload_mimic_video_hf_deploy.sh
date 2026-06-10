#!/bin/bash
#
# Upload a mimic-video training run to HF for deployment.
# Unlike upload_mimic_video_hf.sh (which preserves the full training layout),
# this produces a flat checkpoints/ tree that serve_mimic_video.sh can use
# directly after `hf download`:
#
#   checkpoints/
#     action/iter_XXXXXX_fused.pt   <- LoRA fused into base weights
#     action/config.yaml            <- action net architecture
#     video/<video_ckpt_filename>   <- V2W backbone the run was trained on
#     dataset_statistics.json       <- normalization stats (no .statistics_cache needed)
#
# Usage: edit RUN_DIR and VIDEO_DIT_PATH below, then run.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="${SCRIPT_DIR}/mimic-video/model"

RUN_DIR="/ephemeral/mimic-video-so101/runs/mimic_video/w2a_lerobot_iter_000000650_fused_lr1.000e-04_layer20_bsz128_20260521_023248/vam/lerobot/w2a_lerobot_iter_000000650_fused_lr1.000e-04_layer20_bsz128"
VIDEO_DIT_PATH="/ephemeral/mimic-video-so101/mimic-video/model/checkpoints/video_backbone/iter_000000650_fused.pt"
DATA_DIR="/ephemeral/mimic-video-so101/staging/mimic-video"

HF_ORG="robot-learning"
CONDA_ENV="${CONDA_ENV:-lerobot}"

CKPT_DIR="$RUN_DIR/checkpoints"

# Latest action model iteration
LATEST_ITER=$(cat "$CKPT_DIR/latest_checkpoint.txt" | grep -oP 'iter_\d+' | tail -1)
ACTION_CKPT="$CKPT_DIR/model/${LATEST_ITER}.pt"
FUSED_ACTION_CKPT="$CKPT_DIR/model/${LATEST_ITER}_fused.pt"

# RUN_NAME is the timestamped top-level dir
RUN_NAME=$(basename "$(dirname "$(dirname "$(dirname "$RUN_DIR")")")")
HF_REPO="$HF_ORG/$RUN_NAME"

echo "=== mimic-video deployment upload ==="
echo "Run dir:    $RUN_DIR"
echo "Iteration:  $LATEST_ITER"
echo "Video ckpt: $VIDEO_DIT_PATH"
echo "HF repo:    $HF_REPO"
echo

# Fuse LoRA adapters into the action checkpoint if not already done
if [[ ! -f "$FUSED_ACTION_CKPT" ]]; then
    echo "==> Fusing LoRA weights: $ACTION_CKPT"
    source "${MODEL_DIR}/.venv/bin/activate"
    python "${MODEL_DIR}/scripts/fuse_lora_ckpt.py" "$ACTION_CKPT"
    deactivate
else
    echo "==> Fused checkpoint already exists: $FUSED_ACTION_CKPT"
fi

# Pick the most recent stats file as dataset_statistics.json
STATS_FILE=$(ls -t "$DATA_DIR"/.statistics_cache/* 2>/dev/null | head -n1 || true)
if [[ -z "$STATS_FILE" ]]; then
    echo "ERROR: no stats file found in $DATA_DIR/.statistics_cache/" >&2
    exit 1
fi
echo "==> Using stats: $STATS_FILE"

# Activate conda for hf CLI
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

VIDEO_FILENAME=$(basename "$VIDEO_DIT_PATH")
ACTION_FILENAME=$(basename "$FUSED_ACTION_CKPT")

echo "==> Uploading action model -> checkpoints/action/$ACTION_FILENAME"
hf upload "$HF_REPO" "$FUSED_ACTION_CKPT" "checkpoints/action/$ACTION_FILENAME" --repo-type model

echo "==> Uploading video model -> checkpoints/video/$VIDEO_FILENAME"
hf upload "$HF_REPO" "$VIDEO_DIT_PATH" "checkpoints/video/$VIDEO_FILENAME" --repo-type model

echo "==> Uploading stats -> checkpoints/dataset_statistics.json"
hf upload "$HF_REPO" "$STATS_FILE" "checkpoints/dataset_statistics.json" --repo-type model

echo "==> Uploading config -> checkpoints/action/config.yaml"
hf upload "$HF_REPO" "$RUN_DIR/config.yaml" "checkpoints/action/config.yaml" --repo-type model

echo
echo "==> Done. View at: https://huggingface.co/$HF_REPO"
echo
echo "To serve, download and run:"
echo "  hf download $HF_REPO --repo-type model --local-dir ./checkpoints"
echo "  VIDEO_MODEL_PATH=./checkpoints/video/$VIDEO_FILENAME \\"
echo "  ACTION_MODEL_PATH=./checkpoints/action/$ACTION_FILENAME \\"
echo "  ./deployment/serve_mimic_video.sh"
