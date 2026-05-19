#!/bin/bash
#
# Standalone script for converting one or more LeRobot v3 datasets to
# mimic-video's per-episode .zarr layout, then precomputing T5 language
# embeddings. Can also be invoked by slurm_scripts/process_lerobot.sbatch.
#
# Edit DATASET_PAIRS below to choose which datasets to process.
#
# NOTE: The lerobot conda env needs zarr<3 / numcodecs<0.16 installed.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODEL_DIR="${REPO_ROOT}/mimic-video/model"

# ---- Datasets to process --------------------------------------------------
# Each entry is "<dataset_name>=<ex_type>". The dataset_name resolves to
# ${DATA_ROOT}/<dataset_name> (input) and ${DATA_ROOT}/<dataset_name>-zarr
# (output). The ex_type selects the language instruction from
# config/language_instructions.json.
DATASET_PAIRS=(
    "ex1_all_v4=ex1"
    "ex2_all_v4=ex2"
    "ex3-1-blue_all=ex3-1-blue"
    "ex3-1-orange_all=ex3-1-orange"
    "ex3-2-blue_all=ex3-2-blue"
    "ex3-2-orange_all=ex3-2-orange"
)
# ---------------------------------------------------------------------------

CONDA_ENV="${CONDA_ENV:-lerobot}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data}"
INSTRUCTIONS_JSON="${INSTRUCTIONS_JSON:-${REPO_ROOT}/config/language_instructions.json}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${REPO_ROOT}/mimic-video/model/checkpoints}"
CONDA_BASE="${CONDA_BASE:-$(conda info --base 2>/dev/null || echo "${HOME}/miniconda3")}"

if (( ${#DATASET_PAIRS[@]} == 0 )); then
    echo "ERROR: DATASET_PAIRS is empty — edit the list at the top of $0" >&2
    exit 2
fi

declare -a DATASETS EX_TYPES IN_DIRS OUT_DIRS
for pair in "${DATASET_PAIRS[@]}"; do
    if [[ "${pair}" != *"="* ]]; then
        echo "ERROR: expected <dataset>=<ex_type>, got '${pair}'" >&2
        exit 2
    fi
    ds="${pair%%=*}"
    DATASETS+=("${ds}")
    EX_TYPES+=("${pair#*=}")
    IN_DIRS+=("${DATA_ROOT}/${ds}")
    OUT_DIRS+=("${DATA_ROOT}/${ds}-zarr")
done

# Resolve language instructions up front using whichever python is on PATH —
# avoids re-activating envs for each dataset.
declare -a LANGUAGE_INSTRUCTIONS
for ex in "${EX_TYPES[@]}"; do
    LANGUAGE_INSTRUCTIONS+=("$(python3 "${REPO_ROOT}/helpers/utils/language_instructions.py" "${ex}" --instructions "${INSTRUCTIONS_JSON}")")
done

source "${CONDA_BASE}/bin/activate"
conda activate "${CONDA_ENV}"

export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

cd "${REPO_ROOT}"

echo "=== LeRobot -> zarr conversion ==="
echo "Node:        $(hostname)"
echo "Conda env:   ${CONDA_ENV}"
echo "Datasets:    ${#DATASETS[@]}"
echo "Checkpoints: ${CHECKPOINT_DIR}"
echo

for i in "${!DATASETS[@]}"; do
    ds="${DATASETS[$i]}"
    ex="${EX_TYPES[$i]}"
    in_dir="${IN_DIRS[$i]}"
    out_dir="${OUT_DIRS[$i]}"
    lang="${LANGUAGE_INSTRUCTIONS[$i]}"

    echo "--- [$((i+1))/${#DATASETS[@]}] ${ds} (ex_type=${ex}) ---"
    echo "Input dir:   ${in_dir}"
    echo "Output dir:  ${out_dir}"
    echo "Instruction: ${lang}"

    python mimic-video/data_preprocessing/action/process_lerobot.py \
      --repo-id "${ds}" \
      --root "${in_dir}" \
      --output-dir "${out_dir}" \
      --language-instruction "${lang}" \
      --overwrite
done

echo "=== Precomputing T5 language embeddings ==="

conda deactivate
source "${MODEL_DIR}/.venv/bin/activate"

export PATH="/sbin:/usr/sbin:${PATH}"
export CUDA_HOME="${MODEL_DIR}/.venv/lib/python3.10/site-packages/nvidia/cuda_nvrtc"
export CUDA_PATH="${CUDA_HOME}"
export LD_LIBRARY_PATH="${MODEL_DIR}/.venv/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib:${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${MODEL_DIR}/.venv/lib/python3.10/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"
export COSMOS_PREDICT2_ARGS="${COSMOS_PREDICT2_ARGS:---checkpoints ${CHECKPOINT_DIR}}"

for i in "${!DATASETS[@]}"; do
    ds="${DATASETS[$i]}"
    out_dir="${OUT_DIRS[$i]}"
    lang="${LANGUAGE_INSTRUCTIONS[$i]}"

    echo "--- T5 for ${ds} ---"
    python mimic-video/data_preprocessing/action/precompute_t5.py \
      --dataset-path "${out_dir}" \
      --prompt "${lang}"
done

echo "=== Conversion + T5 Precompute Complete ==="
