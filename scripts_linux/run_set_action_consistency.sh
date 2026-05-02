#!/usr/bin/env bash
set -e
set -x
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"
activate_env
check_path "${CODE_DIR}" "CODE_DIR"
check_path "${TINY_DATA_ROOT}" "TINY_DATA_ROOT"
check_path "${VAE_PATH}" "VAE_PATH"
check_path "${RESUME}/checkpoint-last.pth" "RESUME checkpoint"
cd "${CODE_DIR}"
RUN_DIR="${OUTPUT_ROOT}/set_action_consistency"
mkdir -p "${RUN_DIR}"
CMD=(python eval_set_action_consistency.py --data_path "${TINY_DATA_ROOT}" --vae_path "${VAE_PATH}" --resume "${RESUME}" --output_dir "${RUN_DIR}" --model revealmar_base --num_eval_images 16 --eval_bsz 4 --num_states_per_image 1 --num_iter 64 --state_step_min 0 --state_step_max 16 --candidate_pool_size 8 --set_k 4 --local_radius 1 --rollout_horizon 1 --deredundancy_lambda 0.2 --pseudo_target_type mixed_reveal --cfg 1.0 --num_workers 0)
printf '%q ' "${CMD[@]}" > "${RUN_DIR}/run_args.txt"
write_json_config "${RUN_DIR}/config.json" script=run_set_action_consistency output_dir="${RUN_DIR}" data_path="${TINY_DATA_ROOT}"
"${CMD[@]}" 2>&1 | tee "${RUN_DIR}/eval.log"
