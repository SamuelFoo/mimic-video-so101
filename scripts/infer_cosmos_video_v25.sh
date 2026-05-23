#!/bin/bash
#
# Run Video2World inference using the Cosmos-Predict2.5 pipeline.
# Analogous to scripts/infer_video.sh but for Cosmos-Predict2.5.
#
# Cosmos-Predict2.5 uses Cosmos-Reason1 embeddings (reason1_proj) instead of
# the T5 embeddings used by Cosmos 2.0.  The experiment configs and text-encoder
# setup are handled automatically through the registered mimic_video.py
# experiment (cosmos_predict2/experiments/base/mimic_video.py).
#
# PREREQUISITES
#   1. Run ./scripts/export_video_inputs.sh to produce the batch.json and
#      input MP4 clips from zarr episodes.
#   2. Have a Cosmos-Predict2.5 checkpoint from train_cosmos_video_v25.sh.
#      Checkpoints are saved in DCP (Distributed Checkpoint) format as
#      iter_XXXXXXXXX/ directories.  This script converts the checkpoint
#      to model_ema_bf16.pt automatically before inference (CONVERT_CKPT=true).
#
# USAGE
#   ./scripts/infer_cosmos_video_v25.sh
#
#   # Override per-run settings via env vars, e.g.:
#   EX_TYPE=ex2 \
#   RUN_DIR=/path/to/video_inference/run \
#   CHECKPOINT_PATH=/path/to/runs/cosmos_video_v25/.../checkpoints/iter_000000100 \
#   ./scripts/infer_cosmos_video_v25.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODEL_DIR="${REPO_ROOT}/cosmos-predict2.5"

# ---- Arguments ------------------------------------------------------------
EX_TYPE="${EX_TYPE:-ex1}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data}"
DATASET_NAME="${DATASET_NAME:-ex1_merged_2026-05-13_12-20-44}"
RUN_DIR="${RUN_DIR:-${REPO_ROOT}/runs/video_inference/${DATASET_NAME}}"
BATCH_JSON="${BATCH_JSON:-${RUN_DIR}/batch.json}"

# Inner run dir for the Cosmos-Predict2.5 training run (contains checkpoints/).
# Mirrors the RUN_DIR layout from upload_cosmos_v25_checkpoints_hf.sh.
V25_RUN_DIR="${V25_RUN_DIR:-${REPO_ROOT}/runs/cosmos_video_v25/predict2_v2w_lora_rank32_ex1_ex2_ex3_merged_2026-05-23_11-20-06/cosmos-video-v25-finetune/video2world_mimic/predict2_v2w_lora_rank32_ex1_ex2_ex3_merged_246784}"

# Checkpoint: path to a DCP iter_XXXXXXXXX directory or a pre-converted .pt
# file.  Defaults to the latest checkpoint recorded in latest_checkpoint.txt.
if [[ -f "${V25_RUN_DIR}/checkpoints/latest_checkpoint.txt" ]]; then
  _LATEST_ITER="$(grep -oP 'iter_\d+' "${V25_RUN_DIR}/checkpoints/latest_checkpoint.txt" | tail -1)"
else
  _LATEST_ITER="iter_000000000"
fi
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${V25_RUN_DIR}/checkpoints/${_LATEST_ITER}}"

# Experiment name registered in cosmos_predict2/experiments/base/mimic_video.py.
# Must match the name used during training (derived from LORA_RANK + DATASET_NAME
# in train_cosmos_video_v25.sh).
EXPERIMENT="${EXPERIMENT:-predict2_v2w_lora_rank32_ex1_ex2_ex3_merged}"

# Human-readable label for the output sub-directory (defaults to checkpoint basename).
MODEL_NAME="${MODEL_NAME:-$(basename "${CHECKPOINT_PATH}")}"
OUTPUT_DIR="${OUTPUT_DIR:-${RUN_DIR}/outputs_v25/${MODEL_NAME}}"

# Number of diffusion steps (35 = default quality; reduce for faster iteration).
NUM_STEPS="${NUM_STEPS:-35}"
# CFG guidance scale (0–7; author default is 7).
GUIDANCE="${GUIDANCE:-7}"
SEED="${SEED:-0}"
# Run only every EPISODE_STRIDE-th episode from the batch (1 = all episodes).
EPISODE_STRIDE="${EPISODE_STRIDE:-5}"
# Set to false to skip DCP -> .pt conversion (when CHECKPOINT_PATH is already
# a .pt file).
CONVERT_CKPT="${CONVERT_CKPT:-true}"
# Disable the safety guardrail — robot lab footage is never unsafe and the
# guardrail adds ~30 s overhead per generated video.
DISABLE_GUARDRAILS="${DISABLE_GUARDRAILS:-true}"
# ---------------------------------------------------------------------------

MODEL_PYTHON="${MODEL_DIR}/.venv/bin/python"

export PATH="/sbin:/usr/sbin:${PATH}"
export CUDA_HOME="${MODEL_DIR}/.venv/lib/python3.10/site-packages/nvidia/cuda_nvrtc"
export CUDA_PATH="${CUDA_HOME}"
_CUDA_RUNTIME_LIB="${MODEL_DIR}/.venv/lib/python3.10/site-packages/nvidia/cuda_runtime/lib"
if [[ -f "${_CUDA_RUNTIME_LIB}/libcudart.so.12" && ! -e "${_CUDA_RUNTIME_LIB}/libcudart.so" ]]; then
    ln -sf libcudart.so.12 "${_CUDA_RUNTIME_LIB}/libcudart.so"
fi
export LD_LIBRARY_PATH="${_CUDA_RUNTIME_LIB}:${MODEL_DIR}/.venv/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib:${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${MODEL_DIR}/.venv/lib/python3.10/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"
export TOKENIZERS_PARALLELISM="false"

echo "=== Cosmos-Predict2.5 Video2World Inference ==="
echo "Ex type:       ${EX_TYPE}"
echo "Dataset:       ${DATASET_NAME}"
echo "Run dir:       ${RUN_DIR}"
echo "Batch JSON:    ${BATCH_JSON}"
echo "Checkpoint:    ${CHECKPOINT_PATH}"
echo "Experiment:    ${EXPERIMENT}"
echo "Output dir:    ${OUTPUT_DIR}"
echo "Steps:         ${NUM_STEPS}"
echo "Guidance:      ${GUIDANCE}"
echo "Seed:          ${SEED}"
echo "Episode stride: ${EPISODE_STRIDE}"
echo

if [[ ! -f "${BATCH_JSON}" ]]; then
  echo "Batch JSON not found: ${BATCH_JSON}" >&2
  echo "Run ./scripts/export_video_inputs.sh first." >&2
  exit 1
fi

# ---- Checkpoint: DCP -> .pt conversion ------------------------------------
# Cosmos-Predict2.5 saves checkpoints in PyTorch DCP (Distributed Checkpoint)
# format.  examples/inference.py requires a single .pt file.
PT_CHECKPOINT="${CHECKPOINT_PATH}"
if [[ "${CONVERT_CKPT}" == "true" && -d "${CHECKPOINT_PATH}" ]]; then
  PT_CHECKPOINT="${CHECKPOINT_PATH}/model_ema_bf16.pt"
  if [[ -f "${PT_CHECKPOINT}" ]]; then
    echo "==> Converted checkpoint already exists: ${PT_CHECKPOINT}"
  else
    echo "==> Converting DCP checkpoint to .pt ..."
    echo "    Source: ${CHECKPOINT_PATH}/model"
    echo "    Target: ${CHECKPOINT_PATH}/"
    cd "${MODEL_DIR}"
    "${MODEL_PYTHON}" scripts/convert_distcp_to_pt.py \
      "${CHECKPOINT_PATH}/model" \
      "${CHECKPOINT_PATH}"
    echo "==> Converted: ${PT_CHECKPOINT}"
  fi
elif [[ ! -f "${PT_CHECKPOINT}" ]]; then
  echo "ERROR: checkpoint not found: ${PT_CHECKPOINT}" >&2
  echo "  Set CHECKPOINT_PATH to a DCP iter_XXXXXXXXX directory or a .pt file." >&2
  exit 2
fi

# ---- Build Cosmos-Predict2.5 batch JSON -----------------------------------
# export_video_inputs.sh produces a Cosmos 2.0 batch JSON with fields:
#   {"input_video": "...", "prompt": "...", "output_video": "..."}
#
# examples/inference.py (Cosmos 2.5) expects InferenceArguments fields:
#   {"name": "...", "inference_type": "video2world", "input_path": "...", "prompt": "..."}
#
# inference_type=video2world uses num_latent_conditional_frames=2, which reads
# the last 5 pixel frames from each input video — matching obs_history=5 from
# the mimic_video.py training config.
V25_BATCH_JSON="$(mktemp --suffix=.json)"
trap 'rm -f "${V25_BATCH_JSON}"' EXIT

echo "==> Building Cosmos 2.5 batch JSON (episode_stride=${EPISODE_STRIDE})"
"${MODEL_PYTHON}" - <<PYEOF
import json, pathlib, sys

with open("${BATCH_JSON}") as f:
    items = json.load(f)

items = items[::${EPISODE_STRIDE}]

out = []
for item in items:
    input_video = item["input_video"]
    # Derive a stable sample name from the video filename (e.g. "episode_000042").
    name = pathlib.Path(input_video).stem
    out.append({
        "name": name,
        "inference_type": "video2world",
        "input_path": input_video,
        "prompt": item["prompt"],
        "guidance": ${GUIDANCE},
        "seed": ${SEED},
        "num_steps": ${NUM_STEPS},
    })

with open("${V25_BATCH_JSON}", "w") as f:
    json.dump(out, f, indent=2)

print(f"Episodes selected: {len(out)} (stride=${EPISODE_STRIDE})", file=sys.stderr)
PYEOF

mkdir -p "${OUTPUT_DIR}"

# ---- Run inference ---------------------------------------------------------
# Must be run from MODEL_DIR so that cosmos_predict2.* imports resolve and
# the config system can discover cosmos_predict2/experiments/base/mimic_video.py.
source "${MODEL_DIR}/.venv/bin/activate"

EXTRA_ARGS=()
if [[ "${DISABLE_GUARDRAILS}" == "true" ]]; then
  EXTRA_ARGS+=(--disable-guardrails)
fi

echo "==> Starting Cosmos-Predict2.5 inference"
echo "    Checkpoint: ${PT_CHECKPOINT}"
echo "    Output dir: ${OUTPUT_DIR}"
echo
cd "${MODEL_DIR}"
torchrun \
  --standalone \
  --nproc_per_node=1 \
  examples/inference.py \
  -i "${V25_BATCH_JSON}" \
  -o "${OUTPUT_DIR}" \
  --checkpoint-path "${PT_CHECKPOINT}" \
  --experiment "${EXPERIMENT}" \
  "${EXTRA_ARGS[@]}"

echo
echo "=== Inference complete ==="
echo "Outputs: ${OUTPUT_DIR}"
