#!/usr/bin/env bash
set -e
set -x
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"
activate_env
check_path "${CODE_DIR}" "CODE_DIR"
check_path "${RESUME}/checkpoint-last.pth" "RESUME checkpoint"
cd "${CODE_DIR}"
ROOT="${OUTPUT_ROOT}/early_step_intervention"
mkdir -p "${ROOT}"
for spec in planner_no_intervention:none:0 planner_intervene_random8:random:8 planner_intervene_confidence8:confidence:8 planner_intervene_entropy8:entropy:8; do
  IFS=: read -r name policy steps <<<"${spec}"
  RUN_DIR="${ROOT}/${name}"
  mkdir -p "${RUN_DIR}"
  CMD=(python eval_early_step_intervention.py --data_path "${TINY_DATA_ROOT}" --vae_path "${VAE_PATH}" --resume "${RESUME}" --output_dir "${RUN_DIR}" --model revealmar_base --num_images 1000 --class_num 1000 --eval_bsz 32 --num_iter 64 --num_sampling_steps 100 --cfg 1.0 --cfg_schedule linear --temperature 1.0 --num_workers 0 --base_policy planner --intervention_policy "${policy}" --intervention_steps "${steps}" --budget_mode hard --pseudo_target_type mixed_reveal --candidate_pool_size 8 --log_planner_sampling_steps 8)
  printf '%q ' "${CMD[@]}" > "${RUN_DIR}/run_args.txt"
  write_json_config "${RUN_DIR}/config.json" run_name="${name}" intervention_policy="${policy}" intervention_steps="${steps}" output_dir="${RUN_DIR}"
  "${CMD[@]}" 2>&1 | tee "${RUN_DIR}/eval.log"
done
