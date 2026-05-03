#!/usr/bin/env bash
set -e
set -x

AUTO_SHUTDOWN=${AUTO_SHUTDOWN:-0}
SHUTDOWN_ON_ERROR=${SHUTDOWN_ON_ERROR:-0}

shutdown_if_requested_success() {
  if [[ "${AUTO_SHUTDOWN}" = "1" ]]; then
    echo "Training and main-paper evaluation finished successfully. Shutting down..."
    sync
    /usr/bin/shutdown -h now
  fi
}

shutdown_if_requested_error() {
  if [[ "${SHUTDOWN_ON_ERROR}" = "1" ]]; then
    echo "Run failed and SHUTDOWN_ON_ERROR=1. Shutting down..."
    sync
    /usr/bin/shutdown -h now
  fi
}

trap 'rc=$?; if [[ $rc -ne 0 ]]; then shutdown_if_requested_error; fi; exit $rc' EXIT

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_env.sh"
MODEL_NAME=large

echo "CODE_DIR=${CODE_DIR}"
echo "DATA_ROOT=${DATA_ROOT}"
echo "PRETRAIN_ROOT=${PRETRAIN_ROOT}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "MODEL_NAME=${MODEL_NAME}"
echo "TRAIN_RUN_NAME=${TRAIN_RUN_NAME}"
echo "TRAIN_EPOCHS=${TRAIN_EPOCHS}"
echo "WARMUP_EPOCHS=${WARMUP_EPOCHS}"
echo "TRAIN_BSZ=${TRAIN_BSZ}"
echo "TRAIN_MAX_STEPS=${TRAIN_MAX_STEPS}"
echo "USE_TORCHRUN=${USE_TORCHRUN}"
echo "NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "MIXED_POLICY_RATIO=${MIXED_POLICY_RATIO}"
echo "EVAL_RUN_NAME=${EVAL_RUN_NAME:-eval_mainpaper}"
echo "EVAL_NUM_IMAGES=${EVAL_NUM_IMAGES:-50000}"
echo "EVAL_CLASS_NUM=${EVAL_CLASS_NUM:-1000}"
echo "EVAL_BSZ=${EVAL_BSZ:-128}"
echo "EVAL_POLICIES=baseline,planner"
echo "EVAL_NUM_ITERS=128,256"
echo "EVAL_GPU_IDS=${EVAL_GPU_IDS:-0,1,2,3}"

bash "${SCRIPT_DIR}/check_server_ready.sh" "${MODEL_NAME}"
bash "${SCRIPT_DIR}/train_large_ref_mixed.sh"
check_path "${OUTPUT_ROOT}/${MODEL_NAME}/${TRAIN_RUN_NAME}/checkpoint-last.pth" "trained ${MODEL_NAME} checkpoint"
bash "${SCRIPT_DIR}/eval_large_parallel_mainpaper_4points.sh"

echo "[OK] Final output root: ${OUTPUT_ROOT}/${MODEL_NAME}"
shutdown_if_requested_success
