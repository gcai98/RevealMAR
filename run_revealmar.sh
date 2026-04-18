#!/bin/bash
set -e

# RevealMAR launcher: set server paths and experiment knobs, then pick MODE.
# Baseline MAR is unchanged; this script only calls main_revealmar.py.

# --- Required server paths (replace with real values) ---
export SERVER_DATA_PATH="${SERVER_DATA_PATH:-<SERVER_DATA_PATH>}"
export SERVER_VAE_PATH="${SERVER_VAE_PATH:-<SERVER_VAE_PATH>}"
export SERVER_RESUME_DIR="${SERVER_RESUME_DIR:-<SERVER_MAR_BASE_RESUME_DIR>}"

# --- Experiment control ---
export PSEUDO_TARGET_TYPE="${PSEUDO_TARGET_TYPE:-none}"   # none | gt_reveal | pred_reveal | mixed_reveal
export PLANNER_LOSS_WEIGHT="${PLANNER_LOSS_WEIGHT:-1.0}"
export CANDIDATE_POOL_SIZE="${CANDIDATE_POOL_SIZE:-8}"
export MODE="${MODE:-evaluate}"   # evaluate | train_small

# mar_base-compatible diffloss (required when resuming mar_base)
DIFF_ARGS=(--diffloss_d 6 --diffloss_w 1024)

if [[ "${MODE}" == "evaluate" ]]; then
  python main_revealmar.py \
    --model revealmar_base \
    --data_path "${SERVER_DATA_PATH}" \
    --vae_path "${SERVER_VAE_PATH}" \
    --resume "${SERVER_RESUME_DIR}" \
    "${DIFF_ARGS[@]}" \
    --evaluate \
    --pseudo_target_type "${PSEUDO_TARGET_TYPE}" \
    --planner_loss_weight "${PLANNER_LOSS_WEIGHT}" \
    --candidate_pool_size "${CANDIDATE_POOL_SIZE}" \
    --num_iter 256 \
    --num_sampling_steps 100 \
    --cfg 2.9 \
    --planner_hidden_dim 128 \
    --budget_mode soft \
    --mixed_policy_ratio 0.0
elif [[ "${MODE}" == "train_small" ]]; then
  # Small real training smoke: few epochs, small batch, loss decomposition to stdout.
  python main_revealmar.py \
    --model revealmar_base \
    --data_path "${SERVER_DATA_PATH}" \
    --vae_path "${SERVER_VAE_PATH}" \
    --resume "${SERVER_RESUME_DIR}" \
    "${DIFF_ARGS[@]}" \
    --epochs 2 \
    --batch_size 8 \
    --save_last_freq 1 \
    --eval_freq 999999 \
    --pseudo_target_type "${PSEUDO_TARGET_TYPE}" \
    --planner_loss_weight "${PLANNER_LOSS_WEIGHT}" \
    --candidate_pool_size "${CANDIDATE_POOL_SIZE}" \
    --planner_hidden_dim 128 \
    --budget_mode soft \
    --mixed_policy_ratio 0.0 \
    --log_revealmar_losses \
    --log_revealmar_loss_freq 20
else
  echo "Unknown MODE=${MODE} (use evaluate or train_small)" >&2
  exit 1
fi
