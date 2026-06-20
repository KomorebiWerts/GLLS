#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export KRAGAD_PROJECT_ROOT="${KRAGAD_PROJECT_ROOT:-${REPO_ROOT}}"
export KRAGAD_DATA_ROOT="${KRAGAD_DATA_ROOT:-${HOME}/data/kragad}"
export KRAGAD_DATASET_ROOT="${KRAGAD_DATASET_ROOT:-${KRAGAD_DATA_ROOT}/datasets/MMAD}"
export KRAGAD_QA_ROOT="${KRAGAD_QA_ROOT:-${KRAGAD_DATA_ROOT}/datasets/QA_collection}"
export KRAGAD_DATABASE_ROOT="${KRAGAD_DATABASE_ROOT:-${KRAGAD_DATA_ROOT}/datasets/KRagAD/databases}"
export KRAGAD_SAM3_PATH="${KRAGAD_SAM3_PATH:-${KRAGAD_DATA_ROOT}/models/sam3/sam3.pt}"
export KRAGAD_ADAPTCLIP_ROOT="${KRAGAD_ADAPTCLIP_ROOT:-${KRAGAD_DATA_ROOT}/models/AdaptCLIP}"
export KRAGAD_VENV="${KRAGAD_VENV:-${KRAGAD_DATA_ROOT}/envs/kragad}"

export PYTHONPATH="${KRAGAD_PROJECT_ROOT}/src:${KRAGAD_PROJECT_ROOT}/src/kragad/models/AdaptCLIP:${KRAGAD_DATA_ROOT}/external/sam3:${PYTHONPATH:-}"
export HF_HOME="${KRAGAD_DATA_ROOT}/hf_home"
export TORCH_HOME="${KRAGAD_DATA_ROOT}/torch_cache"
export XDG_CACHE_HOME="${KRAGAD_DATA_ROOT}/cache"

if [[ -f "${KRAGAD_VENV}/bin/activate" ]]; then
  source "${KRAGAD_VENV}/bin/activate"
else
  echo "Warning: virtual environment not found at ${KRAGAD_VENV}; continuing without activation." >&2
fi

cd "${KRAGAD_PROJECT_ROOT}"
