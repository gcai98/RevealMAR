#!/usr/bin/env bash
set -e
set -o pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_env.sh"
activate_planmar_env
cd "${CODE_DIR}"
model_cfg huge

MODEL_NAME=huge
TRAIN_DIR="${OUTPUT_ROOT}/${MODEL_NAME}/${TRAIN_RUN_NAME}"
LOG_DIR="${OUTPUT_ROOT}/${MODEL_NAME}/logs"
mkdir -p "${TRAIN_DIR}" "${LOG_DIR}"

echo "MODEL_NAME=${MODEL_NAME}"
echo "TRAIN_RUN_NAME=${TRAIN_RUN_NAME}"
echo "TRAIN_EPOCHS=${TRAIN_EPOCHS}"
echo "WARMUP_EPOCHS=${WARMUP_EPOCHS}"
echo "TRAIN_BSZ=${TRAIN_BSZ}"
echo "TRAIN_MAX_STEPS=${TRAIN_MAX_STEPS}"
echo "USE_TORCHRUN=${USE_TORCHRUN}"
echo "NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "MIXED_POLICY_RATIO=${MIXED_POLICY_RATIO}"
echo "EVAL_POLICIES=${EVAL_POLICIES}"
echo "TRAIN_DIR=${TRAIN_DIR}"

if [ "${USE_TORCHRUN}" = "1" ]; then
  LAUNCHER=(torchrun "--nproc_per_node=${NPROC_PER_NODE}")
else
  LAUNCHER=(python)
fi

CMD=("${LAUNCHER[@]}" main_revealmar.py
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
  --pseudo_target_type ref_mixed_reveal
  --ref_target_horizon 1
  --ref_target_local_radius 1
  --ref_target_mix_alpha 0.5
  --ref_target_loss feature_mse
  --ref_target_max_candidates 8
  --candidate_pool_size 8
  --candidate_selection_mode mixed
  --planner_loss_weight 1.0
  --mixed_policy_ratio "${MIXED_POLICY_RATIO}"
  --log_ref_target_debug
  --log_ref_target_freq 50
  --log_revealmar_losses
  --log_revealmar_loss_freq 50
  --dist_url env://)

printf '%q ' "${CMD[@]}" > "${TRAIN_DIR}/run_args.txt"
write_json_config "${TRAIN_DIR}/config.json" model_name="${MODEL_NAME}" model="${MODEL}" train_run_name="${TRAIN_RUN_NAME}" train_epochs="${TRAIN_EPOCHS}" warmup_epochs="${WARMUP_EPOCHS}" train_bsz="${TRAIN_BSZ}" train_max_steps="${TRAIN_MAX_STEPS}" use_torchrun="${USE_TORCHRUN}" nproc_per_node="${NPROC_PER_NODE}" mixed_policy_ratio="${MIXED_POLICY_RATIO}" data_path="${DATA_ROOT}" resume="${PRETRAIN_CKPT}" output_dir="${TRAIN_DIR}" pseudo_target_type=ref_mixed_reveal candidate_selection_mode=mixed planner_loss_weight=1.0
"${CMD[@]}" 2>&1 | tee "${TRAIN_DIR}/train.log" "${LOG_DIR}/${TRAIN_RUN_NAME}.log"

check_path "${TRAIN_DIR}/checkpoint-last.pth" "trained checkpoint"
echo "[OK] Trained checkpoint: ${TRAIN_DIR}/checkpoint-last.pth"
