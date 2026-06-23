#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${REPO_ROOT}/scripts/dev/activate_glls.sh"
"${GLLS_PYTHON}" -m glls.cli.run "$@"
