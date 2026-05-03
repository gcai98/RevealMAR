#!/usr/bin/env bash
set -e
set -o pipefail
set -x

export EVAL_RUN_NAME=${EVAL_RUN_NAME:-eval_mainpaper}
export EVAL_NUM_IMAGES=${EVAL_NUM_IMAGES:-50000}
export EVAL_CLASS_NUM=${EVAL_CLASS_NUM:-1000}
export EVAL_BSZ=${EVAL_BSZ:-128}
export EVAL_POLICIES=baseline,planner
export EVAL_NUM_ITERS=128,256
export EVAL_GPU_IDS=${EVAL_GPU_IDS:-0,1,2,3}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_env.sh"

activate_planmar_env
check_path "${CODE_DIR}" "CODE_DIR"
check_path "${DATA_ROOT}" "DATA_ROOT"
check_path "${VAE_PATH}" "VAE checkpoint"
cd "${CODE_DIR}"
model_cfg large

MODEL_NAME=large
TRAIN_DIR="${OUTPUT_ROOT}/${MODEL_NAME}/${TRAIN_RUN_NAME}"
EVAL_ROOT="${OUTPUT_ROOT}/${MODEL_NAME}/${EVAL_RUN_NAME}"
LOG_DIR="${OUTPUT_ROOT}/${MODEL_NAME}/logs"
mkdir -p "${EVAL_ROOT}" "${LOG_DIR}"

check_path "${PRETRAIN_CKPT}" "pretrained baseline checkpoint"
if [[ -d "${PRETRAIN_CKPT}" ]]; then
  check_path "${PRETRAIN_CKPT}/checkpoint-last.pth" "pretrained baseline checkpoint-last.pth"
fi
check_path "${TRAIN_DIR}/checkpoint-last.pth" "trained planner checkpoint"

if (( EVAL_NUM_IMAGES % EVAL_CLASS_NUM != 0 )); then
  echo "[ERROR] EVAL_NUM_IMAGES must be divisible by EVAL_CLASS_NUM" >&2
  exit 1
fi

split_csv "${EVAL_GPU_IDS}" GPU_IDS
if [[ "${#GPU_IDS[@]}" -eq 0 ]]; then
  echo "[ERROR] EVAL_GPU_IDS is empty" >&2
  exit 1
fi

POLICIES=(baseline planner)
NUM_ITERS=(128 256)
TOTAL_JOBS=4
CONCURRENCY="${#GPU_IDS[@]}"
if [[ "${CONCURRENCY}" -gt "${TOTAL_JOBS}" ]]; then
  CONCURRENCY="${TOTAL_JOBS}"
fi

echo "CODE_DIR=${CODE_DIR}"
echo "DATA_ROOT=${DATA_ROOT}"
echo "PRETRAIN_ROOT=${PRETRAIN_ROOT}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "MODEL_NAME=${MODEL_NAME}"
echo "TRAIN_RUN_NAME=${TRAIN_RUN_NAME}"
echo "EVAL_RUN_NAME=${EVAL_RUN_NAME}"
echo "EVAL_NUM_IMAGES=${EVAL_NUM_IMAGES}"
echo "EVAL_BSZ=${EVAL_BSZ}"
echo "EVAL_POLICIES=${EVAL_POLICIES}"
echo "EVAL_NUM_ITERS=${EVAL_NUM_ITERS}"
echo "EVAL_GPU_IDS=${EVAL_GPU_IDS}"
echo "TOTAL_EVAL_JOBS=${TOTAL_JOBS}"
echo "CONCURRENCY=${CONCURRENCY}"
echo "MIXED_POLICY_RATIO=${MIXED_POLICY_RATIO}"

PIDS=()
LABELS=()
FAIL=0

wait_batch() {
  local i
  for i in "${!PIDS[@]}"; do
    if wait "${PIDS[$i]}"; then
      echo "[OK] ${LABELS[$i]}"
    else
      echo "[WARN] Eval failed: ${LABELS[$i]}" >&2
      FAIL=1
    fi
  done
  PIDS=()
  LABELS=()
}

JOB_INDEX=0
for POLICY in "${POLICIES[@]}"; do
  for NUM_ITER in "${NUM_ITERS[@]}"; do
    GPU_ID="${GPU_IDS[$((JOB_INDEX % CONCURRENCY))]}"
    RUN_DIR="${EVAL_ROOT}/${POLICY}_iter${NUM_ITER}"
    LOG_PATH="${LOG_DIR}/${EVAL_RUN_NAME}_${POLICY}_iter${NUM_ITER}.log"
    mkdir -p "${RUN_DIR}"
    if [[ "${POLICY}" = "planner" ]]; then
      RESUME="${TRAIN_DIR}"
      BUDGET_MODE=soft
      PSEUDO_TARGET_TYPE=ref_mixed_reveal
      DEBUG_ARGS=(--log_planner_sampling_debug --log_planner_sampling_steps 8)
    else
      RESUME="${PRETRAIN_CKPT}"
      BUDGET_MODE=hard
      PSEUDO_TARGET_TYPE=none
      DEBUG_ARGS=()
    fi
    CMD=(python main_revealmar.py
      --evaluate
      --model "${MODEL}"
      --img_size 256
      --vae_path "${VAE_PATH}"
      --vae_embed_dim 16
      --vae_stride 16
      --patch_size 1
      --data_path "${DATA_ROOT}"
      --resume "${RESUME}"
      --output_dir "${RUN_DIR}"
      --class_num "${EVAL_CLASS_NUM}"
      --num_images "${EVAL_NUM_IMAGES}"
      --eval_bsz "${EVAL_BSZ}"
      --num_iter "${NUM_ITER}"
      --num_sampling_steps 100
      --cfg 2.9
      --cfg_schedule linear
      --temperature 1.0
      --num_workers 8
      --diffloss_d "${DIFFLOSS_D}"
      --diffloss_w "${DIFFLOSS_W}"
      --diffusion_batch_mul 1
      --sampling_policy "${POLICY}"
      --pseudo_target_type "${PSEUDO_TARGET_TYPE}"
      --budget_mode "${BUDGET_MODE}"
      --candidate_pool_size 8
      --candidate_selection_mode mixed
      "${DEBUG_ARGS[@]}"
      --dist_url env://)
    printf 'CUDA_VISIBLE_DEVICES=%q ' "${GPU_ID}" > "${RUN_DIR}/run_args.txt"
    printf '%q ' "${CMD[@]}" >> "${RUN_DIR}/run_args.txt"
    write_json_config "${RUN_DIR}/config.json" model_name="${MODEL_NAME}" model="${MODEL}" train_run_name="${TRAIN_RUN_NAME}" eval_run_name="${EVAL_RUN_NAME}" policy="${POLICY}" num_iter="${NUM_ITER}" gpu_id="${GPU_ID}" data_path="${DATA_ROOT}" resume="${RESUME}" output_dir="${RUN_DIR}" sampling_policy="${POLICY}" pseudo_target_type="${PSEUDO_TARGET_TYPE}" budget_mode="${BUDGET_MODE}" cfg=2.9 num_images="${EVAL_NUM_IMAGES}" class_num="${EVAL_CLASS_NUM}" eval_bsz="${EVAL_BSZ}"
    echo "[LAUNCH] policy=${POLICY} iter=${NUM_ITER} gpu=${GPU_ID} output=${RUN_DIR} log=${LOG_PATH}"
    (
      set -o pipefail
      CUDA_VISIBLE_DEVICES="${GPU_ID}" "${CMD[@]}" 2>&1 | tee "${RUN_DIR}/eval.log" "${LOG_PATH}"
    ) &
    PIDS+=("$!")
    LABELS+=("${POLICY}_iter${NUM_ITER}_gpu${GPU_ID}")
    JOB_INDEX=$((JOB_INDEX + 1))
    if [[ "${#PIDS[@]}" -ge "${CONCURRENCY}" ]]; then
      wait_batch
    fi
  done
done

if [[ "${#PIDS[@]}" -gt 0 ]]; then
  wait_batch
fi

python scripts_server/collect_main_results.py \
  --root "${OUTPUT_ROOT}" \
  --models large \
  --skip_missing_models \
  --output_prefix "${MODEL_NAME}_mainpaper" \
  --eval_name "${EVAL_RUN_NAME}" \
  --policies baseline,planner

exit "${FAIL}"
