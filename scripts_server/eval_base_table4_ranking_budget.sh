#!/usr/bin/env bash
set -e
set -o pipefail
set -x

EVAL_RUN_NAME=${EVAL_RUN_NAME:-eval_table4_ranking_budget}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_env.sh"

EVAL_GPU_IDS=${EVAL_GPU_IDS:-0,1,2,3,4}
EVAL_TABLE4_NUM_ITER=${EVAL_TABLE4_NUM_ITER:-128}

activate_planmar_env

check_path "${CODE_DIR}" "CODE_DIR"
check_path "${DATA_ROOT}" "DATA_ROOT"
check_path "${VAE_PATH}" "VAE checkpoint"

cd "${CODE_DIR}"

model_cfg base

MODEL_NAME=base
TRAIN_DIR="${OUTPUT_ROOT}/${MODEL_NAME}/${TRAIN_RUN_NAME}"
EVAL_ROOT="${OUTPUT_ROOT}/${MODEL_NAME}/${EVAL_RUN_NAME}"
LOG_DIR="${OUTPUT_ROOT}/${MODEL_NAME}/logs"

mkdir -p "${EVAL_ROOT}" "${LOG_DIR}"

check_path "${PRETRAIN_CKPT}" "pretrained baseline checkpoint"
if [[ -d "${PRETRAIN_CKPT}" ]]; then
  check_path "${PRETRAIN_CKPT}/checkpoint-last.pth" "pretrained baseline checkpoint-last.pth"
fi

check_path "${TRAIN_DIR}" "trained planner directory"
check_path "${TRAIN_DIR}/checkpoint-last.pth" "trained planner checkpoint"

split_csv "${EVAL_GPU_IDS}" GPU_IDS

echo "CODE_DIR=${CODE_DIR}"
echo "DATA_ROOT=${DATA_ROOT}"
echo "PRETRAIN_ROOT=${PRETRAIN_ROOT}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "MODEL_NAME=${MODEL_NAME}"
echo "TRAIN_RUN_NAME=${TRAIN_RUN_NAME}"
echo "EVAL_RUN_NAME=${EVAL_RUN_NAME}"
echo "EVAL_NUM_IMAGES=${EVAL_NUM_IMAGES}"
echo "EVAL_BSZ=${EVAL_BSZ}"
echo "EVAL_TABLE4_NUM_ITER=${EVAL_TABLE4_NUM_ITER}"
echo "EVAL_GPU_IDS=${EVAL_GPU_IDS}"
echo "MIXED_POLICY_RATIO=${MIXED_POLICY_RATIO}"

ROWS=(
  "cosine_schedule_cosine_budget"
  "entropy_ranking_cosine_budget"
  "entropy_ranking_score_budget"
  "planner_ranking_cosine_budget"
  "planner_ranking_score_budget"
)

PIDS=()
LABELS=()
JOB_INDEX=0

for ROW in "${ROWS[@]}"; do
  GPU_ID="${GPU_IDS[$((JOB_INDEX % ${#GPU_IDS[@]}))]}"
  RUN_DIR="${EVAL_ROOT}/${ROW}_iter${EVAL_TABLE4_NUM_ITER}"
  LOG_PATH="${LOG_DIR}/${EVAL_RUN_NAME}_${ROW}_iter${EVAL_TABLE4_NUM_ITER}.log"

  mkdir -p "${RUN_DIR}"

  case "${ROW}" in
    cosine_schedule_cosine_budget)
      RESUME="${PRETRAIN_CKPT}"
      SAMPLING_POLICY=baseline
      BUDGET_MODE=hard
      PSEUDO_TARGET_TYPE=none
      CHECKPOINT_TYPE=original_mar
      RANKING_RULE=cosine_schedule
      BUDGET_RULE=cosine_budget
      DEBUG_ARGS=()
      ;;
    entropy_ranking_cosine_budget)
      RESUME="${TRAIN_DIR}"
      SAMPLING_POLICY=entropy
      BUDGET_MODE=hard
      PSEUDO_TARGET_TYPE=ref_mixed_reveal
      CHECKPOINT_TYPE=planmar_finetuned
      RANKING_RULE=entropy_ranking
      BUDGET_RULE=cosine_budget
      DEBUG_ARGS=()
      ;;
    entropy_ranking_score_budget)
      RESUME="${TRAIN_DIR}"
      SAMPLING_POLICY=entropy
      BUDGET_MODE=soft
      PSEUDO_TARGET_TYPE=ref_mixed_reveal
      CHECKPOINT_TYPE=planmar_finetuned
      RANKING_RULE=entropy_ranking
      BUDGET_RULE=score_derived_budget
      DEBUG_ARGS=()
      ;;
    planner_ranking_cosine_budget)
      RESUME="${TRAIN_DIR}"
      SAMPLING_POLICY=planner
      BUDGET_MODE=hard
      PSEUDO_TARGET_TYPE=ref_mixed_reveal
      CHECKPOINT_TYPE=planmar_finetuned
      RANKING_RULE=planner_ranking
      BUDGET_RULE=cosine_budget
      DEBUG_ARGS=(--log_planner_sampling_debug --log_planner_sampling_steps 8)
      ;;
    planner_ranking_score_budget)
      RESUME="${TRAIN_DIR}"
      SAMPLING_POLICY=planner
      BUDGET_MODE=soft
      PSEUDO_TARGET_TYPE=ref_mixed_reveal
      CHECKPOINT_TYPE=planmar_finetuned
      RANKING_RULE=planner_ranking
      BUDGET_RULE=score_derived_budget
      DEBUG_ARGS=(--log_planner_sampling_debug --log_planner_sampling_steps 8)
      ;;
    *)
      echo "[ERROR] Unknown row: ${ROW}" >&2
      exit 1
      ;;
  esac

  CMD=(
    python main_revealmar.py
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
    --num_iter "${EVAL_TABLE4_NUM_ITER}"
    --num_sampling_steps 100
    --cfg 2.9
    --cfg_schedule linear
    --temperature 1.0
    --num_workers 8
    --diffloss_d "${DIFFLOSS_D}"
    --diffloss_w "${DIFFLOSS_W}"
    --diffusion_batch_mul 1
    --sampling_policy "${SAMPLING_POLICY}"
    --pseudo_target_type "${PSEUDO_TARGET_TYPE}"
    --budget_mode "${BUDGET_MODE}"
    --candidate_pool_size 8
    --candidate_selection_mode mixed
    "${DEBUG_ARGS[@]}"
    --dist_url env://
  )

  printf 'CUDA_VISIBLE_DEVICES=%q ' "${GPU_ID}" > "${RUN_DIR}/run_args.txt"
  printf '%q ' "${CMD[@]}" >> "${RUN_DIR}/run_args.txt"

  write_json_config "${RUN_DIR}/config.json" \
    table=table4 \
    row="${ROW}" \
    model_name="${MODEL_NAME}" \
    model="${MODEL}" \
    checkpoint_type="${CHECKPOINT_TYPE}" \
    train_run_name="${TRAIN_RUN_NAME}" \
    eval_run_name="${EVAL_RUN_NAME}" \
    gpu_id="${GPU_ID}" \
    data_path="${DATA_ROOT}" \
    resume="${RESUME}" \
    output_dir="${RUN_DIR}" \
    ranking_rule="${RANKING_RULE}" \
    budget_rule="${BUDGET_RULE}" \
    sampling_policy="${SAMPLING_POLICY}" \
    pseudo_target_type="${PSEUDO_TARGET_TYPE}" \
    budget_mode="${BUDGET_MODE}" \
    num_iter="${EVAL_TABLE4_NUM_ITER}" \
    cfg=2.9 \
    num_images="${EVAL_NUM_IMAGES}" \
    class_num="${EVAL_CLASS_NUM}" \
    eval_bsz="${EVAL_BSZ}" \
    candidate_pool_size=8 \
    candidate_selection_mode=mixed

  (
    set -o pipefail
    CUDA_VISIBLE_DEVICES="${GPU_ID}" "${CMD[@]}" 2>&1 | tee "${RUN_DIR}/eval.log" "${LOG_PATH}"
  ) &

  PIDS+=("$!")
  LABELS+=("${ROW}_iter${EVAL_TABLE4_NUM_ITER}_gpu${GPU_ID}")
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