#!/bin/bash
set -euo pipefail

# Path to the inner run directory (the one containing checkpoints/, config.yaml, etc.)
# Override by setting RUN_DIR in the environment.
RUN_DIR="${RUN_DIR:-/ephemeral/robot_learning_project/runs/cosmos_video_v25/predict2_v2w_lora_rank32_ex1_ex2_ex3_merged_2026-05-23_11-20-06/cosmos-video-v25-finetune/video2world_mimic/predict2_v2w_lora_rank32_ex1_ex2_ex3_merged_246784}"
CKPT_DIR="$RUN_DIR/checkpoints"
N_CHECKPOINTS="${N_CHECKPOINTS:-5}"
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
hf repo create "$HF_REPO" \
    --repo-type model \
    --exist-ok

# Collect the last N iteration directory names (iter_XXXXXXXXX format)
mapfile -t ITERS < <(
    ls "$CKPT_DIR/" \
    | grep -oP 'iter_\d+' \
    | sort -u \
    | tail -"$N_CHECKPOINTS"
)

echo "==> Uploading last $N_CHECKPOINTS checkpoints: ${ITERS[*]}"

echo "==> Starting upload to $HF_REPO"

# Upload each checkpoint directory individually (--include globs don't recurse into .distcp dirs)
for iter in "${ITERS[@]}"; do
    echo "    uploading checkpoints/$iter ..."
    hf upload \
        "$HF_REPO" \
        "$CKPT_DIR/$iter" \
        "checkpoints/$iter" \
        --repo-type model
done

# Upload latest_checkpoint.txt
if [[ -f "$CKPT_DIR/latest_checkpoint.txt" ]]; then
    hf upload "$HF_REPO" "$CKPT_DIR/latest_checkpoint.txt" "checkpoints/latest_checkpoint.txt" --repo-type model
fi

# Upload config and job metadata files
for f in config.pkl config.yaml job_env.yaml launch_info.yaml; do
    if [[ -f "$RUN_DIR/$f" ]]; then
        echo "    uploading $f ..."
        hf upload "$HF_REPO" "$RUN_DIR/$f" "$f" --repo-type model
    fi
done

echo "==> Done. View at: https://huggingface.co/$HF_REPO"
