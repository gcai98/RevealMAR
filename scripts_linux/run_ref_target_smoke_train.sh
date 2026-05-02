#!/usr/bin/env bash
set -e
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"
activate_env
check_path "${CODE_DIR}" "CODE_DIR"
check_path "${TINY_DATA_ROOT}" "TINY_DATA_ROOT"
check_path "${VAE_PATH}" "VAE_PATH"
check_path "${SOURCE_CHECKPOINT}" "SOURCE_CHECKPOINT"
cd "${CODE_DIR}"

RUN_DIR="${OUTPUT_ROOT}/ref_target_smoke_train"
MODEL_ONLY_INIT="${RUN_DIR}/model_only_init"
mkdir -p "${RUN_DIR}" "${MODEL_ONLY_INIT}"
python tools/make_model_only_checkpoint.py --src "${SOURCE_CHECKPOINT}" --dst_dir "${MODEL_ONLY_INIT}"

CMD=(python main_revealmar.py --model revealmar_base --img_size 256 --vae_path "${VAE_PATH}" --vae_embed_dim 16 --vae_stride 16 --patch_size 1 --data_path "${TINY_DATA_ROOT}" --resume "${MODEL_ONLY_INIT}" --output_dir "${RUN_DIR}" --batch_size 4 --epochs 1 --save_last_freq 1 --eval_freq 999999 --num_workers 0 --diffloss_d 6 --diffloss_w 1024 --num_sampling_steps 100 --diffusion_batch_mul 1 --pseudo_target_type ref_mixed_reveal --ref_target_horizon 1 --ref_target_local_radius 1 --ref_target_mix_alpha 0.5 --ref_target_loss feature_mse --ref_target_max_candidates 8 --log_ref_target_debug --log_ref_target_freq 1 --log_revealmar_losses --log_revealmar_loss_freq 1 --dist_url env://)
printf '%q ' "${CMD[@]}" > "${RUN_DIR}/run_args.txt"
write_json_config "${RUN_DIR}/config.json" script=run_ref_target_smoke_train data_path="${TINY_DATA_ROOT}" resume="${MODEL_ONLY_INIT}" output_dir="${RUN_DIR}"
"${CMD[@]}" 2>&1 | tee "${RUN_DIR}/train.log"
