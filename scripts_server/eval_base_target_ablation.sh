#!/usr/bin/env bash
set -e
set -o pipefail
set -x

EVAL_RUN_NAME=${EVAL_RUN_NAME:-eval_table5_target_ablation}
EVAL_GPU_IDS=${EVAL_GPU_IDS:-0,1,2}
EVAL_TABLE5_NUM_ITER=${EVAL_TABLE5_NUM_ITER:-128}
TARGET_TYPES=${TARGET_TYPES:-ref_gt_reveal,ref_pred_reveal,ref_mixed_reveal}

REF_GT_TRAIN_RUN_NAME=${REF_GT_TRAIN_RUN_NAME:-train_ref_gt_reveal_target_ablation}
REF_PRED_TRAIN_RUN_NAME=${REF_PRED_TRAIN_RUN_NAME:-train_ref_pred_reveal_target_ablation}
REF_MIXED_TRAIN_RUN_NAME=${REF_MIXED_TRAIN_RUN_NAME:-train_ref_mixed_reveal_target_ablation}

REF_TARGET_MIX_ALPHA=${REF_TARGET_MIX_ALPHA:-0.5}
REF_TARGET_HORIZON=${REF_TARGET_HORIZON:-1}
REF_TARGET_LOCAL_RADIUS=${REF_TARGET_LOCAL_RADIUS:-1}
REF_TARGET_MAX_CANDIDATES=${REF_TARGET_MAX_CANDIDATES:-8}
CANDIDATE_POOL_SIZE=${CANDIDATE_POOL_SIZE:-8}
CANDIDATE_SELECTION_MODE=${CANDIDATE_SELECTION_MODE:-mixed}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_env.sh"

activate_planmar_env

check_path "${CODE_DIR}" "CODE_DIR"
check_path "${DATA_ROOT}" "DATA_ROOT"
check_path "${VAE_PATH}" "VAE checkpoint"

cd "${CODE_DIR}"

model_cfg base

MODEL_NAME=base
EVAL_ROOT="${OUTPUT_ROOT}/${MODEL_NAME}/${EVAL_RUN_NAME}"
LOG_DIR="${OUTPUT_ROOT}/${MODEL_NAME}/logs"

mkdir -p "${EVAL_ROOT}" "${LOG_DIR}"

split_csv "${TARGET_TYPES}" TARGET_LIST
split_csv "${EVAL_GPU_IDS}" GPU_IDS

echo "CODE_DIR=${CODE_DIR}"
echo "DATA_ROOT=${DATA_ROOT}"
echo "PRETRAIN_ROOT=${PRETRAIN_ROOT}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "MODEL_NAME=${MODEL_NAME}"
echo "MODEL=${MODEL}"
echo "EVAL_RUN_NAME=${EVAL_RUN_NAME}"
echo "EVAL_NUM_IMAGES=${EVAL_NUM_IMAGES}"
echo "EVAL_BSZ=${EVAL_BSZ}"
echo "EVAL_TABLE5_NUM_ITER=${EVAL_TABLE5_NUM_ITER}"
echo "EVAL_GPU_IDS=${EVAL_GPU_IDS}"
echo "TARGET_TYPES=${TARGET_TYPES}"
echo "REF_GT_TRAIN_RUN_NAME=${REF_GT_TRAIN_RUN_NAME}"
echo "REF_PRED_TRAIN_RUN_NAME=${REF_PRED_TRAIN_RUN_NAME}"
echo "REF_MIXED_TRAIN_RUN_NAME=${REF_MIXED_TRAIN_RUN_NAME}"
echo "REF_TARGET_MIX_ALPHA=${REF_TARGET_MIX_ALPHA}"
echo "REF_TARGET_HORIZON=${REF_TARGET_HORIZON}"
echo "REF_TARGET_LOCAL_RADIUS=${REF_TARGET_LOCAL_RADIUS}"
echo "REF_TARGET_MAX_CANDIDATES=${REF_TARGET_MAX_CANDIDATES}"
echo "CANDIDATE_POOL_SIZE=${CANDIDATE_POOL_SIZE}"
echo "CANDIDATE_SELECTION_MODE=${CANDIDATE_SELECTION_MODE}"
echo "MIXED_POLICY_RATIO=${MIXED_POLICY_RATIO}"

PIDS=()
LABELS=()
JOB_INDEX=0

for TARGET_TYPE in "${TARGET_LIST[@]}"; do
  case "${TARGET_TYPE}" in
    ref_gt_reveal)
      TRAIN_NAME="${REF_GT_TRAIN_RUN_NAME}"
      TARGET_LABEL=ref_gt
      ;;
    ref_pred_reveal)
      TRAIN_NAME="${REF_PRED_TRAIN_RUN_NAME}"
      TARGET_LABEL=ref_pred
      ;;
    ref_mixed_reveal)
      TRAIN_NAME="${REF_MIXED_TRAIN_RUN_NAME}"
      TARGET_LABEL=ref_mixed
      ;;
    *)
      echo "[ERROR] Unsupported TARGET_TYPE: ${TARGET_TYPE}" >&2
      echo "[ERROR] Expected one of: ref_gt_reveal, ref_pred_reveal, ref_mixed_reveal" >&2
      exit 1
      ;;
  esac

  TRAIN_DIR="${OUTPUT_ROOT}/${MODEL_NAME}/${TRAIN_NAME}"
  check_path "${TRAIN_DIR}" "trained target ablation directory for ${TARGET_TYPE}"
  check_path "${TRAIN_DIR}/checkpoint-last.pth" "trained target ablation checkpoint for ${TARGET_TYPE}"

  GPU_ID="${GPU_IDS[$((JOB_INDEX % ${#GPU_IDS[@]}))]}"
  RUN_DIR="${EVAL_ROOT}/${TARGET_LABEL}_iter${EVAL_TABLE5_NUM_ITER}"
  LOG_PATH="${LOG_DIR}/${EVAL_RUN_NAME}_${TARGET_LABEL}_iter${EVAL_TABLE5_NUM_ITER}.log"

  mkdir -p "${RUN_DIR}"

  CMD=(
    python
    main_revealmar.py
    --evaluate
    --model "${MODEL}"
    --img_size 256
    --vae_path "${VAE_PATH}"
    --vae_embed_dim 16
    --vae_stride 16
    --patch_size 1
    --data_path "${DATA_ROOT}"
    --resume "${TRAIN_DIR}"
    --output_dir "${RUN_DIR}"
    --class_num "${EVAL_CLASS_NUM}"
    --num_images "${EVAL_NUM_IMAGES}"
    --eval_bsz "${EVAL_BSZ}"
    --num_iter "${EVAL_TABLE5_NUM_ITER}"
    --num_sampling_steps 100
    --cfg 2.9
    --cfg_schedule linear
    --temperature 1.0
    --num_workers 8
    --diffloss_d "${DIFFLOSS_D}"
    --diffloss_w "${DIFFLOSS_W}"
    --diffusion_batch_mul 1
    --sampling_policy planner
    --pseudo_target_type "${TARGET_TYPE}"
    --budget_mode soft
    --candidate_pool_size "${CANDIDATE_POOL_SIZE}"
    --candidate_selection_mode "${CANDIDATE_SELECTION_MODE}"
    --ref_target_horizon "${REF_TARGET_HORIZON}"
    --ref_target_local_radius "${REF_TARGET_LOCAL_RADIUS}"
    --ref_target_mix_alpha "${REF_TARGET_MIX_ALPHA}"
    --ref_target_loss feature_mse
    --ref_target_max_candidates "${REF_TARGET_MAX_CANDIDATES}"
    --log_planner_sampling_debug
    --log_planner_sampling_steps 8
    --dist_url env://
  )

  printf 'CUDA_VISIBLE_DEVICES=%q ' "${GPU_ID}" > "${RUN_DIR}/run_args.txt"
  printf '%q ' "${CMD[@]}" >> "${RUN_DIR}/run_args.txt"

  write_json_config "${RUN_DIR}/config.json" \
    table=table5 \
    row="${TARGET_LABEL}" \
    model_name="${MODEL_NAME}" \
    model="${MODEL}" \
    train_run_name="${TRAIN_NAME}" \
    eval_run_name="${EVAL_RUN_NAME}" \
    gpu_id="${GPU_ID}" \
    data_path="${DATA_ROOT}" \
    resume="${TRAIN_DIR}" \
    output_dir="${RUN_DIR}" \
    sampling_policy=planner \
    pseudo_target_type="${TARGET_TYPE}" \
    budget_mode=soft \
    num_iter="${EVAL_TABLE5_NUM_ITER}" \
    cfg=2.9 \
    num_images="${EVAL_NUM_IMAGES}" \
    class_num="${EVAL_CLASS_NUM}" \
    eval_bsz="${EVAL_BSZ}" \
    ref_target_horizon="${REF_TARGET_HORIZON}" \
    ref_target_local_radius="${REF_TARGET_LOCAL_RADIUS}" \
    ref_target_mix_alpha="${REF_TARGET_MIX_ALPHA}" \
    ref_target_loss=feature_mse \
    ref_target_max_candidates="${REF_TARGET_MAX_CANDIDATES}" \
    candidate_pool_size="${CANDIDATE_POOL_SIZE}" \
    candidate_selection_mode="${CANDIDATE_SELECTION_MODE}"

  (
    set -o pipefail
    CUDA_VISIBLE_DEVICES="${GPU_ID}" "${CMD[@]}" 2>&1 | tee "${RUN_DIR}/eval.log" "${LOG_PATH}"
  ) &

  PIDS+=("$!")
  LABELS+=("${TARGET_LABEL}_iter${EVAL_TABLE5_NUM_ITER}_gpu${GPU_ID}")
  JOB_INDEX=$((JOB_INDEX + 1))
done

FAIL=0

for i in "${!PIDS[@]}"; do
  if wait "${PIDS[$i]}"; then
    echo "[OK] ${LABELS[$i]}"
  else
    echo "[WARN] Eval failed: ${LABELS[$i]}" >&2
    FAIL=1
  fi
done

exit "${FAIL}"