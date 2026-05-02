#!/usr/bin/env bash
set -e
set -x
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"
activate_env
check_path "${CODE_DIR}" "CODE_DIR"
check_path "${SOURCE_CHECKPOINT}" "SOURCE_CHECKPOINT"
cd "${CODE_DIR}"
ROOT="${OUTPUT_ROOT}/oracle_mismatch_ref_targets"
MODEL_ONLY_INIT="${ROOT}/model_only_init"
mkdir -p "${MODEL_ONLY_INIT}"
python tools/make_model_only_checkpoint.py --src "${SOURCE_CHECKPOINT}" --dst_dir "${MODEL_ONLY_INIT}"
for target in ref_gt_reveal ref_pred_reveal ref_mixed_reveal; do
  TRAIN_DIR="${ROOT}/train_${target}"
  mkdir -p "${TRAIN_DIR}"
  CMD=(python main_revealmar.py --model revealmar_base --img_size 256 --vae_path "${VAE_PATH}" --vae_embed_dim 16 --vae_stride 16 --patch_size 1 --data_path "${TINY_DATA_ROOT}" --resume "${MODEL_ONLY_INIT}" --output_dir "${TRAIN_DIR}" --batch_size 4 --epochs 1 --save_last_freq 1 --eval_freq 999999 --num_workers 0 --diffloss_d 6 --diffloss_w 1024 --num_sampling_steps 100 --diffusion_batch_mul 1 --pseudo_target_type "${target}" --ref_target_horizon 1 --ref_target_local_radius 1 --ref_target_mix_alpha 0.5 --ref_target_loss feature_mse --ref_target_max_candidates 8 --log_ref_target_debug --log_ref_target_freq 20 --log_revealmar_losses --log_revealmar_loss_freq 20 --dist_url env://)
  printf '%q ' "${CMD[@]}" > "${TRAIN_DIR}/run_args.txt"
  write_json_config "${TRAIN_DIR}/config.json" phase=train target_type="${target}" output_dir="${TRAIN_DIR}"
  "${CMD[@]}" 2>&1 | tee "${TRAIN_DIR}/train.log" || true
  for spec in baseline:baseline:none:hard planner_hard:planner:"${target}":hard planner_soft:planner:"${target}":soft; do
    IFS=: read -r variant policy pseudo budget <<<"${spec}"
    RUN_DIR="${ROOT}/eval_${target}_${variant}"
    mkdir -p "${RUN_DIR}"
    ECMD=(python main_revealmar.py --evaluate --model revealmar_base --img_size 256 --vae_path "${VAE_PATH}" --vae_embed_dim 16 --vae_stride 16 --patch_size 1 --data_path "${TINY_DATA_ROOT}" --resume "${TRAIN_DIR}" --class_num 1000 --diffloss_d 6 --diffloss_w 1024 --diffusion_batch_mul 1 --num_images 1000 --eval_bsz 32 --num_iter 64 --num_sampling_steps 100 --cfg 1.0 --cfg_schedule linear --temperature 1.0 --num_workers 0 --output_dir "${RUN_DIR}" --sampling_policy "${policy}" --pseudo_target_type "${pseudo}" --budget_mode "${budget}" --planner_loss_weight 0.3 --candidate_pool_size 8 --dist_url env://)
    [[ "${policy}" == "planner" ]] && ECMD+=(--log_planner_sampling_debug --log_planner_sampling_steps 8)
    printf '%q ' "${ECMD[@]}" > "${RUN_DIR}/run_args.txt"
    write_json_config "${RUN_DIR}/config.json" phase=eval target_type="${target}" eval_variant="${variant}" budget_mode="${budget}" output_dir="${RUN_DIR}"
    "${ECMD[@]}" 2>&1 | tee "${RUN_DIR}/eval.log" || true
  done
done
python collect_oracle_mismatch_results.py --root "${ROOT}"
