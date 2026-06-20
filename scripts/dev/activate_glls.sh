#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

DEFAULT_DATA_ROOT="${HOME}/data/GLLS"
LEGACY_DATA_ROOT="${HOME}/data/kragad"
if [[ -z "${GLLS_DATA_ROOT:-}" && -z "${KRAGAD_DATA_ROOT:-}" && ! -e "${DEFAULT_DATA_ROOT}" && -e "${LEGACY_DATA_ROOT}" ]]; then
  DEFAULT_DATA_ROOT="${LEGACY_DATA_ROOT}"
fi

export GLLS_PROJECT_ROOT="${GLLS_PROJECT_ROOT:-${KRAGAD_PROJECT_ROOT:-${REPO_ROOT}}}"
export GLLS_DATA_ROOT="${GLLS_DATA_ROOT:-${KRAGAD_DATA_ROOT:-${DEFAULT_DATA_ROOT}}}"
export GLLS_DATASET_ROOT="${GLLS_DATASET_ROOT:-${KRAGAD_DATASET_ROOT:-${GLLS_DATA_ROOT}/datasets/MMAD}}"
export GLLS_QA_ROOT="${GLLS_QA_ROOT:-${KRAGAD_QA_ROOT:-${GLLS_DATA_ROOT}/datasets/QA_collection}}"

DEFAULT_DATABASE_ROOT="${GLLS_DATA_ROOT}/datasets/GLLS/databases"
LEGACY_DATABASE_ROOT="${GLLS_DATA_ROOT}/datasets/KRagAD/databases"
if [[ -z "${GLLS_DATABASE_ROOT:-}" && -z "${KRAGAD_DATABASE_ROOT:-}" && ! -e "${DEFAULT_DATABASE_ROOT}" && -e "${LEGACY_DATABASE_ROOT}" ]]; then
  DEFAULT_DATABASE_ROOT="${LEGACY_DATABASE_ROOT}"
fi
export GLLS_DATABASE_ROOT="${GLLS_DATABASE_ROOT:-${KRAGAD_DATABASE_ROOT:-${DEFAULT_DATABASE_ROOT}}}"

export GLLS_SAM3_PATH="${GLLS_SAM3_PATH:-${KRAGAD_SAM3_PATH:-${GLLS_DATA_ROOT}/models/sam3/sam3.pt}}"
export GLLS_ADAPTCLIP_ROOT="${GLLS_ADAPTCLIP_ROOT:-${KRAGAD_ADAPTCLIP_ROOT:-${GLLS_DATA_ROOT}/models/AdaptCLIP}}"

DEFAULT_VENV="${GLLS_DATA_ROOT}/envs/GLLS"
LEGACY_VENV="${GLLS_DATA_ROOT}/envs/kragad"
if [[ -z "${GLLS_VENV:-}" && -z "${KRAGAD_VENV:-}" && ! -f "${DEFAULT_VENV}/bin/activate" && -f "${LEGACY_VENV}/bin/activate" ]]; then
  DEFAULT_VENV="${LEGACY_VENV}"
fi
export GLLS_VENV="${GLLS_VENV:-${KRAGAD_VENV:-${DEFAULT_VENV}}}"

# Compatibility aliases for the existing Python package and older local shells.
export KRAGAD_PROJECT_ROOT="${KRAGAD_PROJECT_ROOT:-${GLLS_PROJECT_ROOT}}"
export KRAGAD_DATA_ROOT="${KRAGAD_DATA_ROOT:-${GLLS_DATA_ROOT}}"
export KRAGAD_DATASET_ROOT="${KRAGAD_DATASET_ROOT:-${GLLS_DATASET_ROOT}}"
export KRAGAD_QA_ROOT="${KRAGAD_QA_ROOT:-${GLLS_QA_ROOT}}"
export KRAGAD_DATABASE_ROOT="${KRAGAD_DATABASE_ROOT:-${GLLS_DATABASE_ROOT}}"
export KRAGAD_SAM3_PATH="${KRAGAD_SAM3_PATH:-${GLLS_SAM3_PATH}}"
export KRAGAD_ADAPTCLIP_ROOT="${KRAGAD_ADAPTCLIP_ROOT:-${GLLS_ADAPTCLIP_ROOT}}"
export KRAGAD_VENV="${KRAGAD_VENV:-${GLLS_VENV}}"

export PYTHONPATH="${GLLS_PROJECT_ROOT}/src:${GLLS_PROJECT_ROOT}/src/kragad/models/AdaptCLIP:${GLLS_DATA_ROOT}/external/sam3:${PYTHONPATH:-}"
export HF_HOME="${GLLS_DATA_ROOT}/hf_home"
export TORCH_HOME="${GLLS_DATA_ROOT}/torch_cache"
export XDG_CACHE_HOME="${GLLS_DATA_ROOT}/cache"

if [[ -f "${GLLS_VENV}/bin/activate" ]]; then
  source "${GLLS_VENV}/bin/activate"
else
  echo "Warning: virtual environment not found at ${GLLS_VENV}; continuing without activation." >&2
fi

cd "${GLLS_PROJECT_ROOT}"
