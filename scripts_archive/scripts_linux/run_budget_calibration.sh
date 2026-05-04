#!/usr/bin/env bash
set -e
set -x
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"
activate_env
check_path "${CODE_DIR}" "CODE_DIR"
check_path "${RESUME}/checkpoint-last.pth" "RESUME checkpoint"
cd "${CODE_DIR}"
ROOT="${OUTPUT_ROOT}/budget_calibration"
mkdir -p "${ROOT}"
for spec in tau1_scale1_beta0:1.0:1.0:0.0 tau0p5_scale1_beta0:0.5:1.0:0.0 tau0p25_scale1_beta0:0.25:1.0:0.0 tau1_scale5_beta0:1.0:5.0:0.0 tau1_scale10_beta0:1.0:10.0:0.0 tau0p5_scale5_beta0p8:0.5:5.0:0.8; do
  IFS=: read -r name tau scale beta <<<"${spec}"
  RUN_DIR="${ROOT}/${name}"
  mkdir -p "${RUN_DIR}"
  CMD=(python main_revealmar.py --evaluate --model revealmar_base --img_size 256 --vae_path "${VAE_PATH}" --vae_embed_dim 16 --vae_stride 16 --patch_size 1 --data_path "${TINY_DATA_ROOT}" --resume "${RESUME}" --class_num 1000 --diffloss_d 6 --diffloss_w 1024 --diffusion_batch_mul 1 --num_images 1000 --eval_bsz 32 --num_iter 64 --num_sampling_steps 100 --cfg 1.0 --cfg_schedule linear --temperature 1.0 --num_workers 0 --output_dir "${RUN_DIR}" --sampling_policy planner --pseudo_target_type mixed_reveal --budget_mode soft --budget_temperature "${tau}" --budget_score_scale "${scale}" --budget_min 1 --budget_max -1 --budget_ema_beta "${beta}" --budget_calibration_debug --planner_loss_weight 0.3 --candidate_pool_size 8 --log_planner_sampling_debug --log_planner_sampling_steps 8 --dist_url env://)
  printf '%q ' "${CMD[@]}" > "${RUN_DIR}/run_args.txt"
  write_json_config "${RUN_DIR}/config.json" run_name="${name}" budget_temperature="${tau}" budget_score_scale="${scale}" budget_ema_beta="${beta}" output_dir="${RUN_DIR}"
  "${CMD[@]}" 2>&1 | tee "${RUN_DIR}/eval.log"
done
