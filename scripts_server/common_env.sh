#!/usr/bin/env bash

CODE_DIR=${CODE_DIR:-/root/autodl-tmp/RevealMAR-revealmar-dev}
DATA_ROOT=${DATA_ROOT:-/root/autodl-tmp/imagenet/imagenet1k_imagefolder_full}
PRETRAIN_ROOT=${PRETRAIN_ROOT:-/root/autodl-tmp/pretrained_models}
OUTPUT_ROOT=${OUTPUT_ROOT:-/root/autodl-tmp/outputs/planmar_main}
VAE_PATH=${VAE_PATH:-${PRETRAIN_ROOT}/vae/kl16.ckpt}

MAR_BASE_CKPT=${MAR_BASE_CKPT:-${PRETRAIN_ROOT}/mar/mar_base}
MAR_LARGE_CKPT=${MAR_LARGE_CKPT:-${PRETRAIN_ROOT}/mar/mar_large}
MAR_HUGE_CKPT=${MAR_HUGE_CKPT:-${PRETRAIN_ROOT}/mar/mar_huge}

CONDA_SH=${CONDA_SH:-/root/miniconda3/etc/profile.d/conda.sh}
CONDA_ENV=${CONDA_ENV:-revealmar}

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

mkdir -p "${OUTPUT_ROOT}"

activate_planmar_env() {
  if [[ ! -f "${CONDA_SH}" ]]; then
    echo "[ERROR] Missing conda init script: ${CONDA_SH}" >&2
    exit 1
  fi
  source "${CONDA_SH}"
  conda activate "${CONDA_ENV}"
}

check_path() {
  local path="$1"
  local label="$2"
  if [[ ! -e "${path}" ]]; then
    echo "[ERROR] Missing ${label}: ${path}" >&2
    exit 1
  fi
}

print_gpu_info() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
  else
    echo "[WARN] nvidia-smi not found"
  fi
}

model_cfg() {
  local model_name="$1"
  case "${model_name}" in
    base)
      MODEL=revealmar_base
      DIFFLOSS_D=6
      DIFFLOSS_W=1024
      PRETRAIN_CKPT=${MAR_BASE_CKPT}
      ;;
    large)
      MODEL=revealmar_large
      DIFFLOSS_D=8
      DIFFLOSS_W=1280
      PRETRAIN_CKPT=${MAR_LARGE_CKPT}
      ;;
    huge)
      MODEL=revealmar_huge
      DIFFLOSS_D=12
      DIFFLOSS_W=1536
      PRETRAIN_CKPT=${MAR_HUGE_CKPT}
      ;;
    *)
      echo "[ERROR] Unknown model name: ${model_name}" >&2
      exit 1
      ;;
  esac
  export MODEL DIFFLOSS_D DIFFLOSS_W PRETRAIN_CKPT
}

write_json_config() {
  local path="$1"
  shift
  python - "$path" "$@" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
items = sys.argv[2:]
config = {}
for item in items:
    key, value = item.split("=", 1)
    config[key] = value
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(config, indent=2), encoding="utf-8")
PY
}
