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

if [[ "${joined}" != *" --discovery_only "* && "${GLLS_BINARY_AD_LIGHTWEIGHT:-0}" != "1" ]]; then
  shot="1"
  for ((i = 0; i < ${#args[@]}; i++)); do
    if [[ "${args[$i]}" == "--k_shot" && $((i + 1)) -lt ${#args[@]} ]]; then
      shot="${args[$((i + 1))]}"
    fi
  done

  if [[ "${joined}" != *" --threshold_policy "* ]]; then
    adaptclip_config="src/glls/models/AdaptCLIP/adaptcliplib/model_config.json"
    if [[ ! -f "${adaptclip_config}" ]]; then
      cat >&2 <<EOF
Missing AdaptCLIP model_config.json: ${adaptclip_config}

This wrapper defaults to the paper-comparable full method. The previous
normal_robust fallback produces the low split accuracies and is now treated as
an explicit ablation only.

Add MPDD/DTD/DAGM binary-AD thresholds to:
  src/glls/models/AdaptCLIP/adaptcliplib/model_config.json

Or intentionally run the lightweight ablation with:
  GLLS_BINARY_AD_LIGHTWEIGHT=1 bash scripts/run/run_binary_ad.sh ...
EOF
      exit 2
    fi
    args+=(--threshold_policy table)
  fi

  if [[ "${joined}" != *" --calibration_shots "* ]]; then
    args+=(--calibration_shots "${shot}")
  fi
  if [[ "${joined}" != *" --binary_score_source "* ]]; then
    args+=(--binary_score_source localizer_image)
  fi
  if [[ "${joined}" != *" --reuse_offline_assets "* && "${joined}" != *" --skip_offline_assets "* ]]; then
    args+=(--reuse_offline_assets)
  fi
  if [[ "${joined}" != *" --final_verifier "* ]]; then
    qwen3_path="${GLLS_QWEN3_VL_MODEL_PATH:-${GLLS_VLM_MODEL_PATH}}"
    if [[ ! -d "${qwen3_path}" ]]; then
      cat >&2 <<EOF
Missing Qwen3-VL model directory: ${qwen3_path}

Set GLLS_QWEN3_VL_MODEL_PATH or pass:
  --final_verifier qwen3 --model_path <qwen3-vl-dir>
EOF
      exit 2
    fi
    args+=(
      --final_verifier qwen3
      --final_verifier_policy anomaly_or
      --model_path "${qwen3_path}"
      --model_type qwen3
      --qwen3_max_crops 3
    )
  fi
fi

"${GLLS_PYTHON}" -m glls.cli.binary_ad "${args[@]}"
