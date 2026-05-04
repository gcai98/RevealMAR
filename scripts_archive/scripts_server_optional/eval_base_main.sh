#!/usr/bin/env bash
set -e
set -o pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_env.sh"
activate_planmar_env
cd "${CODE_DIR}"
model_cfg base

MODEL_NAME=base
TRAIN_DIR="${OUTPUT_ROOT}/${MODEL_NAME}/${TRAIN_RUN_NAME}"
EVAL_ROOT="${OUTPUT_ROOT}/${MODEL_NAME}/${EVAL_RUN_NAME}"
LOG_DIR="${OUTPUT_ROOT}/${MODEL_NAME}/logs"
mkdir -p "${EVAL_ROOT}" "${LOG_DIR}"

split_csv "${EVAL_POLICIES}" POLICIES
split_csv "${EVAL_NUM_ITERS}" NUM_ITERS

echo "MODEL_NAME=${MODEL_NAME}"
echo "TRAIN_RUN_NAME=${TRAIN_RUN_NAME}"
echo "EVAL_RUN_NAME=${EVAL_RUN_NAME}"
echo "EVAL_NUM_IMAGES=${EVAL_NUM_IMAGES}"
echo "EVAL_CLASS_NUM=${EVAL_CLASS_NUM}"
echo "EVAL_BSZ=${EVAL_BSZ}"
echo "EVAL_NUM_ITERS=${EVAL_NUM_ITERS}"
echo "EVAL_POLICIES=${EVAL_POLICIES}"

for POLICY in "${POLICIES[@]}"; do
  for NUM_ITER in "${NUM_ITERS[@]}"; do
    RUN_DIR="${EVAL_ROOT}/${POLICY}_iter${NUM_ITER}"
    mkdir -p "${RUN_DIR}"
    if [[ "${POLICY}" = "planner" ]]; then
      RESUME="${TRAIN_DIR}"
      BUDGET_MODE=soft
      PSEUDO_TARGET_TYPE=ref_mixed_reveal
    else
      RESUME="${PRETRAIN_CKPT}"
      BUDGET_MODE=hard
      PSEUDO_TARGET_TYPE=none
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
      --log_planner_sampling_debug
      --log_planner_sampling_steps 8
      --dist_url env://)
    printf '%q ' "${CMD[@]}" > "${RUN_DIR}/run_args.txt"
    write_json_config "${RUN_DIR}/config.json" model_name="${MODEL_NAME}" model="${MODEL}" train_run_name="${TRAIN_RUN_NAME}" eval_run_name="${EVAL_RUN_NAME}" policy="${POLICY}" num_iter="${NUM_ITER}" data_path="${DATA_ROOT}" resume="${RESUME}" output_dir="${RUN_DIR}" sampling_policy="${POLICY}" pseudo_target_type="${PSEUDO_TARGET_TYPE}" budget_mode="${BUDGET_MODE}" cfg=2.9 num_images="${EVAL_NUM_IMAGES}" class_num="${EVAL_CLASS_NUM}" eval_bsz="${EVAL_BSZ}"
    if "${CMD[@]}" 2>&1 | tee "${RUN_DIR}/eval.log" "${LOG_DIR}/${EVAL_RUN_NAME}_${POLICY}_iter${NUM_ITER}.log"; then
      echo "[OK] ${MODEL_NAME} ${POLICY} iter ${NUM_ITER}"
    else
      echo "[WARN] Eval failed: ${MODEL_NAME} ${POLICY} iter ${NUM_ITER}" >&2
    fi
  done
done

echo "To collect base results:"
echo "python scripts_server/collect_main_results.py --root ${OUTPUT_ROOT} --models base --skip_missing_models --output_prefix base --eval_name ${EVAL_RUN_NAME}"
