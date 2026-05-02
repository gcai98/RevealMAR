#!/usr/bin/env bash
set -e
set -x
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"
activate_env
check_path "${CODE_DIR}" "CODE_DIR"
check_path "${RESUME}/checkpoint-last.pth" "RESUME checkpoint"
cd "${CODE_DIR}"
ROOT="${OUTPUT_ROOT}/reveal_trajectory"
mkdir -p "${ROOT}"
for spec in planner_hard:planner:hard planner_soft:planner:soft random_hard:random:hard confidence_hard:confidence:hard entropy_hard:entropy:hard; do
  IFS=: read -r name policy budget <<<"${spec}"
  RUN_DIR="${ROOT}/${name}"
  mkdir -p "${RUN_DIR}"
  CMD=(python eval_reveal_trajectory.py --data_path "${TINY_DATA_ROOT}" --vae_path "${VAE_PATH}" --resume "${RESUME}" --output_dir "${RUN_DIR}" --model revealmar_base --num_images 16 --class_num 16 --eval_bsz 4 --num_iter 64 --num_sampling_steps 100 --cfg 1.0 --cfg_schedule linear --temperature 1.0 --num_workers 0 --sampling_policy "${policy}" --budget_mode "${budget}" --pseudo_target_type mixed_reveal --candidate_pool_size 8 --save_heatmaps --save_json)
  printf '%q ' "${CMD[@]}" > "${RUN_DIR}/run_args.txt"
  write_json_config "${RUN_DIR}/config.json" run_name="${name}" sampling_policy="${policy}" budget_mode="${budget}" output_dir="${RUN_DIR}"
  "${CMD[@]}" 2>&1 | tee "${RUN_DIR}/eval.log"
done
