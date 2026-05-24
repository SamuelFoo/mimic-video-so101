#!/bin/bash
#
# Run Video2World inference using the Cosmos-Predict2.5 pipeline.
#
# Uses the dataset layout produced by:
#   1. scripts/prepare_video_finetune_data.sh  →  <DATASET_DIR>/video/ + metas/
#   2. scripts/precompute_reason1_embeddings.py →  <DATASET_DIR>/reason1_proj/
#
# USAGE
#   ./scripts/infer_cosmos_video_v25.sh
#
#   # Override per-run settings via env vars, e.g.:
#   DATASET_NAME=ex2_all_v4 \
#   CHECKPOINT_PATH=/path/to/runs/cosmos_video_v25/.../checkpoints/iter_000001100 \
#   ./scripts/infer_cosmos_video_v25.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODEL_DIR="${REPO_ROOT}/cosmos-predict2.5"

# ---- Arguments ------------------------------------------------------------
DATASET_NAME="${DATASET_NAME:-ex1_all_v4}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/staging/mimic-video}"
DATASET_DIR="${DATASET_DIR:-${DATA_ROOT}/${DATASET_NAME}-cosmos-video}"

# Reason1 embeddings — produced by precompute_reason1_embeddings.py inside DATASET_DIR
REASON1_PROJ_DIR="${REASON1_PROJ_DIR:-${DATASET_DIR}/reason1_proj}"

# Inner run dir for the Cosmos-Predict2.5 training run (contains checkpoints/).
V25_RUN_DIR="${V25_RUN_DIR:-${REPO_ROOT}/runs/cosmos_video_v25/predict2_v2w_lora_rank32_ex1_ex2_ex3_merged_82008}"

# Checkpoint: path to a DCP iter_XXXXXXXXX directory or a pre-converted .pt file.
if [[ -f "${V25_RUN_DIR}/checkpoints/latest_checkpoint.txt" ]]; then
  _LATEST_ITER="$(grep -oP 'iter_\d+' "${V25_RUN_DIR}/checkpoints/latest_checkpoint.txt" | tail -1)"
else
  _LATEST_ITER="iter_000000000"
fi
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${V25_RUN_DIR}/checkpoints/${_LATEST_ITER}}"

# Experiment name registered in cosmos_predict2/experiments/base/mimic_video.py.
EXPERIMENT="${EXPERIMENT:-predict2_v2w_lora_rank32_ex1_ex2_ex3_merged}"

# Human-readable label for the output sub-directory (defaults to checkpoint basename).
MODEL_NAME="${MODEL_NAME:-$(basename "${CHECKPOINT_PATH}")}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/runs/video_inference/${DATASET_NAME}/outputs_v25/${MODEL_NAME}}"

# Number of diffusion steps (35 = default quality; reduce for faster iteration).
NUM_STEPS="${NUM_STEPS:-35}"
# CFG guidance scale.
GUIDANCE="${GUIDANCE:-7}"
SEED="${SEED:-0}"
# Run only every EPISODE_STRIDE-th episode (1 = all episodes).
EPISODE_STRIDE="${EPISODE_STRIDE:-1}"
# Set to false to skip DCP -> .pt conversion (when CHECKPOINT_PATH is already a .pt file).
CONVERT_CKPT="${CONVERT_CKPT:-true}"
# Disable the safety guardrail — robot lab footage is never unsafe.
DISABLE_GUARDRAILS="${DISABLE_GUARDRAILS:-true}"
# ---------------------------------------------------------------------------

MODEL_PYTHON="${MODEL_DIR}/.venv/bin/python"

export PATH="/sbin:/usr/sbin:${PATH}"
_CUDA_RUNTIME_LIB="${MODEL_DIR}/.venv/lib/python3.10/site-packages/nvidia/cuda_runtime/lib"
if [[ -f "${_CUDA_RUNTIME_LIB}/libcudart.so.12" && ! -e "${_CUDA_RUNTIME_LIB}/libcudart.so" ]]; then
    ln -sf libcudart.so.12 "${_CUDA_RUNTIME_LIB}/libcudart.so"
fi
export LD_LIBRARY_PATH="${_CUDA_RUNTIME_LIB}:${MODEL_DIR}/.venv/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib:${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${MODEL_DIR}/.venv/lib/python3.10/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"
export TOKENIZERS_PARALLELISM="false"

echo "=== Cosmos-Predict2.5 Video2World Inference ==="
echo "Dataset name:   ${DATASET_NAME}"
echo "Dataset dir:    ${DATASET_DIR}"
echo "Reason1 dir:    ${REASON1_PROJ_DIR}"
echo "Checkpoint:     ${CHECKPOINT_PATH}"
echo "Experiment:     ${EXPERIMENT}"
echo "Output dir:     ${OUTPUT_DIR}"
echo "Steps:          ${NUM_STEPS}"
echo "Guidance:       ${GUIDANCE}"
echo "Seed:           ${SEED}"
echo "Episode stride: ${EPISODE_STRIDE}"
echo

# ---- Validate inputs -------------------------------------------------------
if [[ ! -d "${DATASET_DIR}/video" ]]; then
  echo "ERROR: video/ not found in ${DATASET_DIR}" >&2
  echo "Run ./scripts/prepare_video_finetune_data.sh first." >&2
  exit 1
fi

if [[ ! -d "${REASON1_PROJ_DIR}" ]]; then
  echo "ERROR: reason1_proj/ not found: ${REASON1_PROJ_DIR}" >&2
  echo "Run scripts/precompute_reason1_embeddings.py first." >&2
  exit 1
fi

# ---- Checkpoint: DCP -> .pt conversion ------------------------------------
# Resolve to absolute path so it survives the `cd "${MODEL_DIR}"` before torchrun.
CHECKPOINT_PATH="$(realpath "${CHECKPOINT_PATH}")"
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

# ---- Build Cosmos-Predict2.5 batch JSONL ----------------------------------
# Reads video/*.mp4 and metas/*.txt directly from the training data layout.
# The sample name matches the pickle filename in reason1_proj/ so embeddings
# are loaded automatically by the inference pipeline.
V25_BATCH_JSON="$(mktemp --suffix=.jsonl)"
trap 'rm -f "${V25_BATCH_JSON}"' EXIT

echo "==> Building batch JSONL from ${DATASET_DIR}/video/ (stride=${EPISODE_STRIDE})"
"${MODEL_PYTHON}" - <<PYEOF
import json, pathlib, sys

video_dir = pathlib.Path("${DATASET_DIR}/video")
metas_dir = pathlib.Path("${DATASET_DIR}/metas")

videos = sorted(video_dir.glob("*.mp4"))[::${EPISODE_STRIDE}]

out = []
for video_path in videos:
    name = video_path.stem
    meta_path = metas_dir / f"{name}.txt"
    if not meta_path.exists():
        print(f"  WARN: no meta for {name}, skipping.", file=sys.stderr)
        continue
    prompt = meta_path.read_text(encoding="utf-8").strip()
    out.append({
        "name": name,
        "inference_type": "video2world",
        "input_path": str(video_path),
        "prompt": prompt,
        # Must match num_frames=93 used during training (default is 77, wrong).
        "num_output_frames": 93,
        # Empty string disables negative-prompt CFG. The CFG unconditional pass
        # then uses zero dropout embeddings, matching the training setup.
        "negative_prompt": "",
    })

with open("${V25_BATCH_JSON}", "w") as f:
    for item in out:
        f.write(json.dumps(item) + "\n")

print(f"Episodes selected: {len(out)} (stride=${EPISODE_STRIDE})", file=sys.stderr)
PYEOF

mkdir -p "${OUTPUT_DIR}"

# ---- Run inference ---------------------------------------------------------
export REASON1_PROJ_DIR

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
