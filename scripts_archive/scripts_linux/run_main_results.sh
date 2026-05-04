#!/usr/bin/env bash
set -e
set -x
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"
activate_env
check_path "${CODE_DIR}" "CODE_DIR"
check_path "${RESUME}/checkpoint-last.pth" "RESUME checkpoint"
cd "${CODE_DIR}"
ROOT="${OUTPUT_ROOT}/main_results_trained_ckpt_full_policies"
mkdir -p "${ROOT}"
for spec in mar_baseline:baseline:none:hard random_reveal:random:none:hard confidence_reveal:confidence:none:hard entropy_reveal:entropy:none:hard planmar_fixed_budget:planner:mixed_reveal:hard planmar_score_budget:planner:mixed_reveal:soft; do
  IFS=: read -r name policy pseudo budget <<<"${spec}"
  RUN_DIR="${ROOT}/${name}"
  mkdir -p "${RUN_DIR}"
  CMD=(python main_revealmar.py --evaluate --model revealmar_base --img_size 256 --vae_path "${VAE_PATH}" --vae_embed_dim 16 --vae_stride 16 --patch_size 1 --data_path "${TINY_DATA_ROOT}" --resume "${RESUME}" --class_num 1000 --diffloss_d 6 --diffloss_w 1024 --diffusion_batch_mul 1 --num_images 1000 --eval_bsz 32 --num_iter 64 --num_sampling_steps 100 --cfg 1.0 --cfg_schedule linear --temperature 1.0 --num_workers 0 --output_dir "${RUN_DIR}" --sampling_policy "${policy}" --pseudo_target_type "${pseudo}" --budget_mode "${budget}" --planner_loss_weight 0.3 --candidate_pool_size 8 --uncertainty_mc_samples 2 --uncertainty_policy_temperature 1.0 --dist_url env://)
  if [[ "${policy}" != "baseline" ]]; then
    CMD+=(--log_planner_sampling_debug --log_planner_sampling_steps 8)
  fi
  printf '%q ' "${CMD[@]}" > "${RUN_DIR}/run_args.txt"
  write_json_config "${RUN_DIR}/config.json" variant="${name}" sampling_policy="${policy}" pseudo_target_type="${pseudo}" budget_mode="${budget}" output_dir="${RUN_DIR}"
  "${CMD[@]}" 2>&1 | tee "${RUN_DIR}/eval.log"
done
python collect_p0_results.py --root "${ROOT}"
