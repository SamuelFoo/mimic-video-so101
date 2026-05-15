#!/bin/bash
#
# Launch the mimic-video inference server, wired for the LeRobot pipeline
# trained by scripts/train_mimic_video.sh. Both state and actions are 6-D
# absolute joint angles in degrees (SO-ARM-101); no per-environment rotation
# or gripper conversion happens server-side — the world2action normalizer
# handles denormalization internally.
#
# Run from anywhere:
#   ./deployment/serve_mimic_video.sh
#
# Override any default below via env vars, e.g.:
#   ACTION_MODEL_PATH=/path/to/iter_000050000.pt ./deployment/serve_mimic_video.sh
#
# Reach it from a laptop:
#   ssh -L 8000:localhost:8000 user@<this host>      # then open http://localhost:8000
#   # or use Tailscale / Wireguard / VPN

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODEL_DIR="${REPO_ROOT}/mimic-video/model"
CHECKPOINT_DIR="${MIMIC_VIDEO_CHECKPOINT_DIR:-${MODEL_DIR}/checkpoints}"

# ---- Which trained run to serve -------------------------------------------
# Experiment name is auto-registered by the grid in
# mimic-video/model/cosmos_predict2/configs/experiment/world2action.py for the
# 'lerobot' data_config. The default below matches train_mimic_video.sh.
EXPERIMENT_NAME="${EXPERIMENT_NAME:-w2a_lerobot_iter_000000375_fused_lr1.000e-04_layer20_bsz128}"

# Video backbone — matches the iter_000000375_fused suffix on the experiment.
VIDEO_MODEL_PATH="${VIDEO_MODEL_PATH:-${REPO_ROOT}/checkpoints/video/iter_000000375_fused.pt}"

# Action decoder — newest checkpoint under runs/mimic_video unless overridden.
if [[ -z "${ACTION_MODEL_PATH:-}" ]]; then
  ACTION_MODEL_PATH="$(ls -t "${REPO_ROOT}"/runs/mimic_video/w2a_lerobot_*/vam/lerobot/*/checkpoints/model/iter_*.pt 2>/dev/null | head -n1 || true)"
  if [[ -z "${ACTION_MODEL_PATH}" ]]; then
    echo "ERROR: no trained action decoder found under ${REPO_ROOT}/runs/mimic_video/. " >&2
    echo "Set ACTION_MODEL_PATH=/path/to/iter_NNN.pt explicitly." >&2
    exit 1
  fi
fi

# Dataset normalization stats — MimicDataset writes them under
# ${MIMIC_VIDEO_DATASET_DIR}/.statistics_cache/<stats_id> at training time.
MIMIC_VIDEO_DATASET_DIR_DEFAULT="${MIMIC_VIDEO_DATASET_DIR:-${REPO_ROOT}/data}"
if [[ -z "${DATASET_STATS:-}" ]]; then
  DATASET_STATS="$(ls -t "${MIMIC_VIDEO_DATASET_DIR_DEFAULT}"/.statistics_cache/* 2>/dev/null | head -n1 || true)"
  if [[ -z "${DATASET_STATS}" ]]; then
    echo "ERROR: no dataset statistics file found under " >&2
    echo "  ${MIMIC_VIDEO_DATASET_DIR_DEFAULT}/.statistics_cache/  (run training to populate it)" >&2
    echo "Set DATASET_STATS=/path/to/<hash> explicitly to override." >&2
    exit 1
  fi
fi

# LeRobot policy_io defaults (from configs/dataloading/policy_io/lerobot.yaml).
IMG_HORIZON="${IMG_HORIZON:-5}"
LOWDIM_HORIZON="${LOWDIM_HORIZON:-1}"
# Client should send frames at the conditioning fps (5 Hz). If your camera runs
# faster and you want server-side downsampling, raise FRAME_STRIDE.
FRAME_STRIDE="${FRAME_STRIDE:-1}"
RESIZE_H="${RESIZE_H:-480}"
RESIZE_W="${RESIZE_W:-640}"
EXPECTED_STATE_DIM="${EXPECTED_STATE_DIM:-6}"
STOP_AFTER_STEP="${STOP_AFTER_STEP:-}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

# ---- Env (mirrors scripts/infer_video.sh) ---------------------------------
MODEL_PYTHON="${MODEL_DIR}/.venv/bin/python"

export PATH="/sbin:/usr/sbin:${PATH}"
export CUDA_HOME="${MODEL_DIR}/.venv/lib/python3.10/site-packages/nvidia/cuda_nvrtc"
export CUDA_PATH="${CUDA_HOME}"
export LD_LIBRARY_PATH="${MODEL_DIR}/.venv/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib:${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${MODEL_DIR}/.venv/lib/python3.10/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"
export TOKENIZERS_PARALLELISM="false"
# COSMOS_PREDICT2_ARGS controls where the pipeline looks for tokenizer / text
# encoder weights; mirror the convention used by infer_video.sh.
export COSMOS_PREDICT2_ARGS="${COSMOS_PREDICT2_ARGS:---checkpoints ${CHECKPOINT_DIR}}"
# Ensure cosmos_predict2.* imports resolve even though we cd into MODEL_DIR.
export PYTHONPATH="${MODEL_DIR}:${PYTHONPATH:-}"

EXTRA_ARGS=()
if [[ -n "${STOP_AFTER_STEP}" ]]; then
  EXTRA_ARGS+=("--stop-after-step" "${STOP_AFTER_STEP}")
fi

echo "=== mimic-video inference server (LeRobot) ==="
echo "Experiment:    ${EXPERIMENT_NAME}"
echo "Video ckpt:    ${VIDEO_MODEL_PATH}"
echo "Action ckpt:   ${ACTION_MODEL_PATH}"
echo "Stats:         ${DATASET_STATS}"
echo "img_horizon=${IMG_HORIZON}  lowdim_horizon=${LOWDIM_HORIZON}  frame_stride=${FRAME_STRIDE}"
echo "Listening on:  http://${HOST}:${PORT}"
echo "==============================================="

cd "${MODEL_DIR}"
exec "${MODEL_PYTHON}" "${SCRIPT_DIR}/serve_mimic_video.py" \
  --experiment-name "${EXPERIMENT_NAME}" \
  --video-model-path "${VIDEO_MODEL_PATH}" \
  --action-model-path "${ACTION_MODEL_PATH}" \
  --dataset-statistics-path "${DATASET_STATS}" \
  --img-horizon "${IMG_HORIZON}" \
  --lowdim-horizon "${LOWDIM_HORIZON}" \
  --frame-stride "${FRAME_STRIDE}" \
  --resize-h "${RESIZE_H}" \
  --resize-w "${RESIZE_W}" \
  --expected-state-dim "${EXPECTED_STATE_DIM}" \
  --host "${HOST}" \
  --port "${PORT}" \
  "${EXTRA_ARGS[@]}"
