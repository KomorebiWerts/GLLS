#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

LOCAL_PATHS_FILE="${GLLS_LOCAL_PATHS_FILE:-${SCRIPT_DIR}/local_paths.sh}"
if [[ -f "${LOCAL_PATHS_FILE}" ]]; then
  # shellcheck source=/dev/null
  source "${LOCAL_PATHS_FILE}"
fi

export GLLS_PROJECT_ROOT="${GLLS_PROJECT_ROOT:-${REPO_ROOT}}"
export GLLS_DATA_ROOT="${GLLS_DATA_ROOT:-${HOME}/data/GLLS}"
export GLLS_DATASET_ROOT="${GLLS_DATASET_ROOT:-${GLLS_DATA_ROOT}/datasets/MMAD}"
export GLLS_QA_ROOT="${GLLS_QA_ROOT:-${GLLS_PROJECT_ROOT}/qa_collection}"
export GLLS_DATABASE_ROOT="${GLLS_DATABASE_ROOT:-${GLLS_DATA_ROOT}/databases}"
export GLLS_GRAPH_CACHE_ROOT="${GLLS_GRAPH_CACHE_ROOT:-${GLLS_DATABASE_ROOT}/graph_index}"

export GLLS_VLM_MODEL_PATH="${GLLS_VLM_MODEL_PATH:-${GLLS_DATA_ROOT}/models/qwen3-vl-8B}"
export GLLS_SAM3_PATH="${GLLS_SAM3_PATH:-${GLLS_DATA_ROOT}/models/sam3/sam3.pt}"
export GLLS_ADAPTCLIP_ROOT="${GLLS_ADAPTCLIP_ROOT:-${GLLS_DATA_ROOT}/models/AdaptCLIP}"
export GLLS_ABOUND_MODEL_PATH="${GLLS_ABOUND_MODEL_PATH:-${GLLS_DATA_ROOT}/models/ABounD/model}"
export GLLS_ABOUND_SAVE_PATH="${GLLS_ABOUND_SAVE_PATH:-${GLLS_DATA_ROOT}/models/ABounD/vit336/336/shot4_CL}"
export GLLS_EMBEDDING_MODEL_PATH="${GLLS_EMBEDDING_MODEL_PATH:-${GLLS_DATA_ROOT}/models/bge-base-en-v1.5}"

export GLLS_MPDD_ROOT="${GLLS_MPDD_ROOT:-${GLLS_DATA_ROOT}/datasets/MPDD}"
export GLLS_DTD_ROOT="${GLLS_DTD_ROOT:-${GLLS_DATA_ROOT}/datasets/DTD}"
export GLLS_DAGM_ROOT="${GLLS_DAGM_ROOT:-${GLLS_DATA_ROOT}/datasets/DAGM_KaggleUpload}"

export GLLS_VENV="${GLLS_VENV:-${GLLS_DATA_ROOT}/envs/GLLS}"
export PYTHONPATH="${GLLS_PROJECT_ROOT}/src:${GLLS_PROJECT_ROOT}/src/glls/models/AdaptCLIP:${GLLS_DATA_ROOT}/external/sam3:${PYTHONPATH:-}"
export HF_HOME="${GLLS_DATA_ROOT}/hf_home"
export TORCH_HOME="${GLLS_DATA_ROOT}/torch_cache"
export XDG_CACHE_HOME="${GLLS_DATA_ROOT}/cache"

if [[ -f "${GLLS_VENV}/bin/activate" ]]; then
  source "${GLLS_VENV}/bin/activate"
else
  echo "Warning: virtual environment not found at ${GLLS_VENV}; continuing without activation." >&2
fi

if [[ -z "${GLLS_PYTHON:-}" ]]; then
  if command -v python >/dev/null 2>&1; then
    export GLLS_PYTHON="python"
  elif command -v python3 >/dev/null 2>&1; then
    export GLLS_PYTHON="python3"
  else
    echo "Warning: neither python nor python3 was found on PATH." >&2
  fi
fi

cd "${GLLS_PROJECT_ROOT}"
