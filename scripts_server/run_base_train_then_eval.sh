#!/usr/bin/env bash
set -e
set -x

AUTO_SHUTDOWN=${AUTO_SHUTDOWN:-0}
SHUTDOWN_ON_ERROR=${SHUTDOWN_ON_ERROR:-0}

shutdown_if_requested_success() {
  if [[ "${AUTO_SHUTDOWN}" = "1" ]]; then
    echo "Training and evaluation finished successfully. Shutting down..."
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
MODEL_NAME=base

echo "TRAIN_RUN_NAME=${TRAIN_RUN_NAME}"
echo "TRAIN_EPOCHS=${TRAIN_EPOCHS}"
echo "WARMUP_EPOCHS=${WARMUP_EPOCHS}"
echo "EVAL_RUN_NAME=${EVAL_RUN_NAME}"
echo "EVAL_NUM_IMAGES=${EVAL_NUM_IMAGES}"
echo "EVAL_CLASS_NUM=${EVAL_CLASS_NUM}"
echo "EVAL_BSZ=${EVAL_BSZ}"
echo "EVAL_NUM_ITERS=${EVAL_NUM_ITERS}"
echo "EVAL_POLICIES=${EVAL_POLICIES}"

bash "${SCRIPT_DIR}/check_server_ready.sh" "${MODEL_NAME}"
bash "${SCRIPT_DIR}/train_base_ref_mixed.sh"
check_path "${OUTPUT_ROOT}/${MODEL_NAME}/${TRAIN_RUN_NAME}/checkpoint-last.pth" "trained ${MODEL_NAME} checkpoint"
bash "${SCRIPT_DIR}/eval_base_main.sh"

echo "[OK] Final output root: ${OUTPUT_ROOT}/${MODEL_NAME}"
shutdown_if_requested_success
