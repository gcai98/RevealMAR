#!/usr/bin/env bash
set -e
set -o pipefail
set -x

TARGET_TYPE=${TARGET_TYPE:-ref_mixed_reveal}
REF_TARGET_MIX_ALPHA=${REF_TARGET_MIX_ALPHA:-0.5}
REF_TARGET_HORIZON=${REF_TARGET_HORIZON:-1}
REF_TARGET_LOCAL_RADIUS=${REF_TARGET_LOCAL_RADIUS:-1}
REF_TARGET_MAX_CANDIDATES=${REF_TARGET_MAX_CANDIDATES:-8}
CANDIDATE_POOL_SIZE=${CANDIDATE_POOL_SIZE:-8}
CANDIDATE_SELECTION_MODE=${CANDIDATE_SELECTION_MODE:-mixed}
PLANNER_LOSS_WEIGHT=${PLANNER_LOSS_WEIGHT:-1.0}

case "${TARGET_TYPE}" in
  ref_gt_reveal)
    DEFAULT_TRAIN_RUN_NAME=train_ref_gt_reveal_target_ablation
    ;;
  ref_pred_reveal)
    DEFAULT_TRAIN_RUN_NAME=train_ref_pred_reveal_target_ablation
    ;;
  ref_mixed_reveal)
    DEFAULT_TRAIN_RUN_NAME=train_ref_mixed_reveal_target_ablation
    ;;
  *)
    echo "[ERROR] Unsupported TARGET_TYPE: ${TARGET_TYPE}" >&2
    echo "[ERROR] Expected one of: ref_gt_reveal, ref_pred_reveal, ref_mixed_reveal" >&2
    exit 1
    ;;
esac

TRAIN_RUN_NAME=${TRAIN_RUN_NAME:-${DEFAULT_TRAIN_RUN_NAME}}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_env.sh"

activate_planmar_env

check_path "${CODE_DIR}" "CODE_DIR"
check_path "${DATA_ROOT}" "DATA_ROOT"
check_path "${VAE_PATH}" "VAE checkpoint"

cd "${CODE_DIR}"

model_cfg base

MODEL_NAME=base
TRAIN_DIR="${OUTPUT_ROOT}/${MODEL_NAME}/${TRAIN_RUN_NAME}"
LOG_DIR="${OUTPUT_ROOT}/${MODEL_NAME}/logs"

mkdir -p "${TRAIN_DIR}" "${LOG_DIR}"

check_path "${PRETRAIN_CKPT}" "pretrained baseline checkpoint"
if [[ -d "${PRETRAIN_CKPT}" ]]; then
  check_path "${PRETRAIN_CKPT}/checkpoint-last.pth" "pretrained baseline checkpoint-last.pth"
fi

echo "CODE_DIR=${CODE_DIR}"
echo "DATA_ROOT=${DATA_ROOT}"
echo "PRETRAIN_ROOT=${PRETRAIN_ROOT}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "MODEL_NAME=${MODEL_NAME}"
echo "MODEL=${MODEL}"
echo "TRAIN_RUN_NAME=${TRAIN_RUN_NAME}"
echo "TRAIN_EPOCHS=${TRAIN_EPOCHS}"
echo "WARMUP_EPOCHS=${WARMUP_EPOCHS}"
echo "TRAIN_BSZ=${TRAIN_BSZ}"
echo "TRAIN_MAX_STEPS=${TRAIN_MAX_STEPS}"
echo "USE_TORCHRUN=${USE_TORCHRUN}"
echo "NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "MIXED_POLICY_RATIO=${MIXED_POLICY_RATIO}"
echo "TARGET_TYPE=${TARGET_TYPE}"
echo "REF_TARGET_MIX_ALPHA=${REF_TARGET_MIX_ALPHA}"
echo "REF_TARGET_HORIZON=${REF_TARGET_HORIZON}"
echo "REF_TARGET_LOCAL_RADIUS=${REF_TARGET_LOCAL_RADIUS}"
echo "REF_TARGET_MAX_CANDIDATES=${REF_TARGET_MAX_CANDIDATES}"
echo "CANDIDATE_POOL_SIZE=${CANDIDATE_POOL_SIZE}"
echo "CANDIDATE_SELECTION_MODE=${CANDIDATE_SELECTION_MODE}"
echo "PLANNER_LOSS_WEIGHT=${PLANNER_LOSS_WEIGHT}"
echo "TRAIN_DIR=${TRAIN_DIR}"

if [ "${USE_TORCHRUN}" = "1" ]; then
  LAUNCHER=(torchrun "--nproc_per_node=${NPROC_PER_NODE}")
else
  LAUNCHER=(python)
fi

CMD=(
  "${LAUNCHER[@]}"
  main_revealmar.py
  --model "${MODEL}"
  --img_size 256
  --vae_path "${VAE_PATH}"
  --vae_embed_dim 16
  --vae_stride 16
  --patch_size 1
  --data_path "${DATA_ROOT}"
  --resume "${PRETRAIN_CKPT}"
  --output_dir "${TRAIN_DIR}"
  --diffloss_d "${DIFFLOSS_D}"
  --diffloss_w "${DIFFLOSS_W}"
  --diffusion_batch_mul 1
  --epochs "${TRAIN_EPOCHS}"
  --warmup_epochs "${WARMUP_EPOCHS}"
  --max_train_steps "${TRAIN_MAX_STEPS}"
  --batch_size "${TRAIN_BSZ}"
  --blr 1.0e-4
  --num_workers 8
  --save_last_freq 1
  --eval_freq 999999
  --pseudo_target_type "${TARGET_TYPE}"
  --ref_target_horizon "${REF_TARGET_HORIZON}"
  --ref_target_local_radius "${REF_TARGET_LOCAL_RADIUS}"
  --ref_target_mix_alpha "${REF_TARGET_MIX_ALPHA}"
  --ref_target_loss feature_mse
  --ref_target_max_candidates "${REF_TARGET_MAX_CANDIDATES}"
  --candidate_pool_size "${CANDIDATE_POOL_SIZE}"
  --candidate_selection_mode "${CANDIDATE_SELECTION_MODE}"
  --planner_loss_weight "${PLANNER_LOSS_WEIGHT}"
  --mixed_policy_ratio "${MIXED_POLICY_RATIO}"
  --log_ref_target_debug
  --log_ref_target_freq 50
  --log_revealmar_losses
  --log_revealmar_loss_freq 50
  --dist_url env://
)

printf '%q ' "${CMD[@]}" > "${TRAIN_DIR}/run_args.txt"

write_json_config "${TRAIN_DIR}/config.json" \
  table=table5 \
  model_name="${MODEL_NAME}" \
  model="${MODEL}" \
  train_run_name="${TRAIN_RUN_NAME}" \
  train_epochs="${TRAIN_EPOCHS}" \
  warmup_epochs="${WARMUP_EPOCHS}" \
  train_bsz="${TRAIN_BSZ}" \
  train_max_steps="${TRAIN_MAX_STEPS}" \
  use_torchrun="${USE_TORCHRUN}" \
  nproc_per_node="${NPROC_PER_NODE}" \
  mixed_policy_ratio="${MIXED_POLICY_RATIO}" \
  data_path="${DATA_ROOT}" \
  resume="${PRETRAIN_CKPT}" \
  output_dir="${TRAIN_DIR}" \
  pseudo_target_type="${TARGET_TYPE}" \
  ref_target_horizon="${REF_TARGET_HORIZON}" \
  ref_target_local_radius="${REF_TARGET_LOCAL_RADIUS}" \
  ref_target_mix_alpha="${REF_TARGET_MIX_ALPHA}" \
  ref_target_loss=feature_mse \
  ref_target_max_candidates="${REF_TARGET_MAX_CANDIDATES}" \
  candidate_pool_size="${CANDIDATE_POOL_SIZE}" \
  candidate_selection_mode="${CANDIDATE_SELECTION_MODE}" \
  planner_loss_weight="${PLANNER_LOSS_WEIGHT}"

"${CMD[@]}" 2>&1 | tee "${TRAIN_DIR}/train.log" "${LOG_DIR}/${TRAIN_RUN_NAME}.log"

check_path "${TRAIN_DIR}/checkpoint-last.pth" "trained checkpoint"

echo "[OK] Trained checkpoint: ${TRAIN_DIR}/checkpoint-last.pth"