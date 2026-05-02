#!/usr/bin/env bash

CODE_DIR=${CODE_DIR:-/root/autodl-tmp/RevealMAR-revealmar-dev}
DATA_ROOT=${DATA_ROOT:-/root/autodl-tmp/imagenet/imagenet1k_imagefolder_full}
TINY_DATA_ROOT=${TINY_DATA_ROOT:-/root/autodl-tmp/tiny-imagenet-200-1percent}
PRETRAIN_ROOT=${PRETRAIN_ROOT:-/root/autodl-tmp/pretrained_models}
VAE_PATH=${VAE_PATH:-${PRETRAIN_ROOT}/vae/kl16.ckpt}
MAR_BASE_CKPT=${MAR_BASE_CKPT:-${PRETRAIN_ROOT}/mar/mar_base}
MAR_HUGE_CKPT=${MAR_HUGE_CKPT:-${PRETRAIN_ROOT}/mar/mar_huge}
OUTPUT_ROOT=${OUTPUT_ROOT:-/root/autodl-tmp/outputs/planmar_p0}
SOURCE_CHECKPOINT=${SOURCE_CHECKPOINT:-${CODE_DIR}/checkpoint-last.pth}
RESUME=${RESUME:-${CODE_DIR}}

activate_env() {
  source /root/miniconda3/etc/profile.d/conda.sh
  conda activate revealmar
  export PYTHONUNBUFFERED=1
  export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
}

check_path() {
  local path="$1"
  local label="$2"
  if [[ ! -e "${path}" ]]; then
    echo "[ERROR] Missing ${label}: ${path}" >&2
    exit 1
  fi
}

write_json_config() {
  local path="$1"
  shift
  python - "$path" "$@" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
pairs = sys.argv[2:]
cfg = {}
for item in pairs:
    key, value = item.split("=", 1)
    cfg[key] = value
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
PY
}
