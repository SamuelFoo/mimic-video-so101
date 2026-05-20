#!/bin/bash
#
# Standalone script for converting one or more LeRobot v3 datasets to
# mimic-video's per-episode .zarr layout, then precomputing T5 language
# embeddings. Can also be invoked by slurm_scripts/process_lerobot.sbatch.
#
# Edit DATASET_PAIRS below to choose which datasets to process.
#
# Parallelism:
#   - Phase 1 (zarr conversion) runs all datasets simultaneously (CPU/disk-bound).
#   - Phase 2 (T5 embeddings)  runs up to MAX_T5_JOBS at once (GPU-bound).
#     Override: MAX_T5_JOBS=4 ./process_lerobot.sh
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

# Datasets whose zarr already exists — skip Phase 1 but still run T5 (Phase 2).
ZARR_SKIP=(
    
)
# ---------------------------------------------------------------------------

CONDA_ENV="${CONDA_ENV:-lerobot}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data}"
MAX_T5_JOBS="${MAX_T5_JOBS:-2}"
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

# source "${CONDA_BASE}/bin/activate"
# conda activate "${CONDA_ENV}"

export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

cd "${REPO_ROOT}"

echo "=== LeRobot -> zarr conversion (all ${#DATASETS[@]} datasets in parallel) ==="
echo "Node:        $(hostname)"
echo "Conda env:   ${CONDA_ENV}"
echo "Datasets:    ${#DATASETS[@]}"
echo "Checkpoints: ${CHECKPOINT_DIR}"
echo

_LOG_DIR="$(mktemp -d)"
trap 'rm -rf "${_LOG_DIR}"' EXIT

declare -a ZARR_PIDS=()
declare -a ZARR_IDX=()
for i in "${!DATASETS[@]}"; do
    ds="${DATASETS[$i]}"

    _skip=0
    for _s in "${ZARR_SKIP[@]+"${ZARR_SKIP[@]}"}"; do
        [[ "$ds" == "$_s" ]] && _skip=1 && break
    done
    if (( _skip )); then
        echo "  [skip]  [$((i+1))/${#DATASETS[@]}] ${ds} (zarr already done)"
        continue
    fi

    echo "  [start] [$((i+1))/${#DATASETS[@]}] ${ds}"
    (
        python mimic-video/data_preprocessing/action/process_lerobot.py \
          --repo-id "${ds}" \
          --root "${IN_DIRS[$i]}" \
          --output-dir "${OUT_DIRS[$i]}" \
          --language-instruction "${LANGUAGE_INSTRUCTIONS[$i]}" \
          --overwrite
    ) > "${_LOG_DIR}/zarr_${ds}.log" 2>&1 &
    ZARR_PIDS+=($!)
    ZARR_IDX+=("$i")
done

_ZARR_FAILED=0
for j in "${!ZARR_PIDS[@]}"; do
    ds="${DATASETS[${ZARR_IDX[$j]}]}"
    if wait "${ZARR_PIDS[$j]}"; then
        echo "  [done]  ${ds}"
    else
        echo "  [FAIL]  ${ds} — see log below" >&2
        cat "${_LOG_DIR}/zarr_${ds}.log" >&2
        _ZARR_FAILED=1
    fi
done
(( _ZARR_FAILED )) && exit 1

echo "=== Precomputing T5 language embeddings (max ${MAX_T5_JOBS} parallel) ==="

conda deactivate
source "${MODEL_DIR}/.venv/bin/activate"

export PATH="/sbin:/usr/sbin:${PATH}"
export CUDA_HOME="${MODEL_DIR}/.venv/lib/python3.10/site-packages/nvidia/cuda_nvrtc"
export CUDA_PATH="${CUDA_HOME}"
export LD_LIBRARY_PATH="${MODEL_DIR}/.venv/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib:${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${MODEL_DIR}/.venv/lib/python3.10/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"
export COSMOS_PREDICT2_ARGS="${COSMOS_PREDICT2_ARGS:---checkpoints ${CHECKPOINT_DIR}}"

declare -a T5_PIDS=()
declare -a T5_IDX=()
for i in "${!DATASETS[@]}"; do
    # Semaphore: wait for a slot when at capacity
    while (( ${#T5_PIDS[@]} >= MAX_T5_JOBS )); do
        wait -n
        new_pids=(); new_idx=()
        for j in "${!T5_PIDS[@]}"; do
            if kill -0 "${T5_PIDS[$j]}" 2>/dev/null; then
                new_pids+=("${T5_PIDS[$j]}")
                new_idx+=("${T5_IDX[$j]}")
            else
                wait "${T5_PIDS[$j]}" || { echo "  [FAIL] T5 for ${DATASETS[${T5_IDX[$j]}]}" >&2; exit 1; }
                echo "  [done] T5 ${DATASETS[${T5_IDX[$j]}]}"
            fi
        done
        T5_PIDS=("${new_pids[@]+"${new_pids[@]}"}")
        T5_IDX=("${new_idx[@]+"${new_idx[@]}"}")
    done

    ds="${DATASETS[$i]}"
    echo "  [start] T5 ${ds}"
    (
        python mimic-video/data_preprocessing/action/precompute_t5.py \
          --dataset-path "${OUT_DIRS[$i]}" \
          --prompt "${LANGUAGE_INSTRUCTIONS[$i]}"
    ) > "${_LOG_DIR}/t5_${ds}.log" 2>&1 &
    T5_PIDS+=($!)
    T5_IDX+=("$i")
done

# Drain remaining
for j in "${!T5_PIDS[@]}"; do
    if wait "${T5_PIDS[$j]}"; then
        echo "  [done] T5 ${DATASETS[${T5_IDX[$j]}]}"
    else
        echo "  [FAIL] T5 ${DATASETS[${T5_IDX[$j]}]} — see log below" >&2
        cat "${_LOG_DIR}/t5_${DATASETS[${T5_IDX[$j]}]}.log" >&2
        exit 1
    fi
done

echo "=== Conversion + T5 Precompute Complete ==="
