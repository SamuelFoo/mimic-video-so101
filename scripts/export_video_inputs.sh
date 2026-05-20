#!/bin/bash
#
# Export existing zarr episodes to MP4 inputs plus a Video2World batch JSON.
# Iterates over all DATASET_PAIRS defined below.
#
# Run:
#   ./scripts/export_video_inputs.sh
#
# Per-dataset overrides via env (prefix with DATASET_NAME):
#   OVERWRITE=true ./scripts/export_video_inputs.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODEL_DIR="${REPO_ROOT}/mimic-video/model"

# ---- Datasets to export ---------------------------------------------------
# Each entry is "<dataset_name>=<ex_type>". The zarr is read from
# ${DATA_ROOT}/<dataset_name>-zarr and outputs go to
# ${REPO_ROOT}/runs/video_inference/<dataset_name>_<TIMESTAMP>/
DATASET_PAIRS=(
    "ex3-1-blue_all=ex3-1-blue"
    "ex3-1-orange_all=ex3-1-orange"
    "ex3-2-blue_all=ex3-2-blue"
    "ex3-2-orange_all=ex3-2-orange"
)
# ---------------------------------------------------------------------------

DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y-%m-%d_%H-%M-%S)}"
INSTRUCTIONS_JSON="${INSTRUCTIONS_JSON:-${REPO_ROOT}/config/language_instructions.json}"
OVERWRITE="${OVERWRITE:-false}"
MAX_EPISODES="${MAX_EPISODES:-}"   # empty means all episodes
EPISODES=()                        # e.g. (0 4 7) — applies to every dataset
FPS="${FPS:-10}"
FRAME_FRACTION="${FRAME_FRACTION:-0.5}"

MODEL_PYTHON="${MODEL_DIR}/.venv/bin/python"

export PATH="/sbin:/usr/sbin:${PATH}"
export CUDA_HOME="${MODEL_DIR}/.venv/lib/python3.10/site-packages/nvidia/cuda_nvrtc"
export CUDA_PATH="${CUDA_HOME}"
export LD_LIBRARY_PATH="${MODEL_DIR}/.venv/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib:${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${MODEL_DIR}/.venv/lib/python3.10/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"

if [[ "${OVERWRITE}" != "true" && "${OVERWRITE}" != "false" ]]; then
  echo "OVERWRITE must be true or false" >&2
  exit 2
fi

if (( ${#DATASET_PAIRS[@]} == 0 )); then
    echo "ERROR: DATASET_PAIRS is empty — edit the list at the top of $0" >&2
    exit 2
fi

echo "=== Export Video2World inputs ==="
echo "Timestamp:     ${TIMESTAMP}"
echo "Datasets:      ${#DATASET_PAIRS[@]}"
echo "Overwrite:     ${OVERWRITE}"
echo "Frame frac:    ${FRAME_FRACTION}"
echo

for pair in "${DATASET_PAIRS[@]}"; do
    if [[ "${pair}" != *"="* ]]; then
        echo "ERROR: expected <dataset>=<ex_type>, got '${pair}'" >&2
        exit 2
    fi
    ds="${pair%%=*}"
    ex_type="${pair#*=}"

    zarr_dir="${DATA_ROOT}/${ds}-zarr"
    run_dir="${REPO_ROOT}/runs/video_inference/${ds}_${TIMESTAMP}"
    video_dir="${run_dir}/inputs"
    output_dir="${run_dir}/outputs"
    batch_json="${run_dir}/batch.json"

    echo "--- ${ds} (${ex_type}) ---"

    if [[ ! -d "${zarr_dir}" ]]; then
        echo "  [skip] zarr not found: ${zarr_dir}" >&2
        continue
    fi

    language_instruction="$(python3 "${REPO_ROOT}/helpers/utils/language_instructions.py" "${ex_type}" --instructions "${INSTRUCTIONS_JSON}")"

    echo "  zarr:        ${zarr_dir}"
    echo "  run dir:     ${run_dir}"
    echo "  instruction: ${language_instruction:0:80}..."

    export_args=(
      --zarr-dir "${zarr_dir}"
      --video-dir "${video_dir}"
      --output-dir "${output_dir}"
      --batch-json "${batch_json}"
      --ex-type "${ex_type}"
      --instructions "${INSTRUCTIONS_JSON}"
      --prompt "${language_instruction}"
      --fps "${FPS}"
      --frame-fraction "${FRAME_FRACTION}"
    )

    if [[ -n "${MAX_EPISODES}" ]]; then
        export_args+=(--max-episodes "${MAX_EPISODES}")
    fi

    if [[ "${#EPISODES[@]}" -gt 0 ]]; then
        export_args+=(--episodes "${EPISODES[@]}")
    fi

    if [[ "${OVERWRITE}" == "true" ]]; then
        export_args+=(--overwrite)
    fi

    cd "${REPO_ROOT}"
    "${MODEL_PYTHON}" "${REPO_ROOT}/helpers/export_zarr_videos.py" "${export_args[@]}"

    echo "  [done] batch JSON: ${batch_json}"
    echo
done

echo "=== Export complete ==="
