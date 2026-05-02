#!/usr/bin/env bash
set -e
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_env.sh"

MODEL_NAME=${1:-}
if [[ -z "${MODEL_NAME}" ]]; then
  echo "Usage: bash scripts_server/check_server_ready.sh base|large|huge" >&2
  exit 1
fi

activate_planmar_env
model_cfg "${MODEL_NAME}"

check_path "${CODE_DIR}" "CODE_DIR"
cd "${CODE_DIR}"

git branch --show-current || true
CURRENT_BRANCH="$(git branch --show-current || true)"
if [[ "${CURRENT_BRANCH}" != "revealmar-dev" ]]; then
  echo "[WARN] Expected branch revealmar-dev, got: ${CURRENT_BRANCH}"
fi
git log -1 --oneline || true

check_path "${DATA_ROOT}/train" "ImageNet train directory"
check_path "${VAE_PATH}" "VAE checkpoint"
check_path "${PRETRAIN_CKPT}" "${MODEL_NAME} checkpoint directory"
if [[ -d "${PRETRAIN_CKPT}" ]]; then
  check_path "${PRETRAIN_CKPT}/checkpoint-last.pth" "${MODEL_NAME} checkpoint-last.pth"
fi

print_gpu_info

python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
PY

python -m py_compile main_revealmar.py engine_mar.py models/revealmar.py util/revealmar_utils.py

echo "[OK] Server ready for ${MODEL_NAME}"
