#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

source "${REPO_ROOT}/scripts/dev/activate_glls.sh"

"${GLLS_PYTHON}" scripts/data/prepare_binary_ad_datasets.py --quiet

args=("$@")
joined=" $* "
if [[ "${joined}" != *" --dataset "* ]]; then
  args=(--dataset all "${args[@]}")
fi
if [[ "${joined}" != *" --output_dir "* && "${joined}" != *" --discovery_only "* ]]; then
  args+=(--output_dir outputs/binary_ad/mpdd_dtd_dagm)
fi

"${GLLS_PYTHON}" -m glls.cli.binary_ad "${args[@]}"
