#!/usr/bin/env bash
set -e
set -x

CODE_DIR=${CODE_DIR:-/path/to/revealmar}
DATA_ROOT=${DATA_ROOT:-/path/to/imagenet}
PRETRAIN_ROOT=${PRETRAIN_ROOT:-/path/to/pretrained_models}
VAE_PATH=${VAE_PATH:-${PRETRAIN_ROOT}/vae/kl16.ckpt}
SOURCE_CHECKPOINT=${SOURCE_CHECKPOINT:-${PRETRAIN_ROOT}/mar/mar_base/checkpoint-last.pth}
OUTPUT_DIR=${OUTPUT_DIR:-/path/to/output/planmar_linux_preflight}
MODEL_ONLY_INIT=${MODEL_ONLY_INIT:-${OUTPUT_DIR}/model_only_init}

source "${CONDA_SH:-/path/to/miniconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-revealmar}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

test -d "${CODE_DIR}"
test -d "${DATA_ROOT}"
test -f "${VAE_PATH}"
test -f "${SOURCE_CHECKPOINT}"

mkdir -p "${OUTPUT_DIR}" "${MODEL_ONLY_INIT}"
cd "${CODE_DIR}"

python -m py_compile tools/make_model_only_checkpoint.py
python -m py_compile main_revealmar.py models/revealmar.py util/revealmar_utils.py engine_mar.py
python -m py_compile eval_surrogate_validity.py eval_set_action_consistency.py eval_candidate_subset_stability.py eval_reveal_trajectory.py eval_early_step_intervention.py
python -m py_compile collect_p0_results.py collect_surrogate_validity.py collect_oracle_mismatch_results.py collect_budget_calibration_results.py collect_set_action_consistency_results.py collect_mixed_policy_ablation_results.py collect_same_parameter_control_results.py collect_reveal_trajectory_results.py collect_candidate_subset_ablation_results.py

python tools/make_model_only_checkpoint.py --src "${SOURCE_CHECKPOINT}" --dst_dir "${MODEL_ONLY_INIT}"

TRAIN_DIR="${OUTPUT_DIR}/train_ref_mixed_preflight"
EVAL_DIR="${OUTPUT_DIR}/eval_planner_preflight"
mkdir -p "${TRAIN_DIR}" "${EVAL_DIR}"

TRAIN_CMD=(python main_revealmar.py
  --model revealmar_base
  --img_size 256
  --vae_path "${VAE_PATH}"
  --vae_embed_dim 16
  --vae_stride 16
  --patch_size 1
  --data_path "${DATA_ROOT}"
  --resume "${MODEL_ONLY_INIT}"
  --output_dir "${TRAIN_DIR}"
  --batch_size 4
  --epochs 1
  --max_train_steps 20
  --save_last_freq 1
  --eval_freq 999999
  --num_workers 4
  --diffloss_d 6
  --diffloss_w 1024
  --diffusion_batch_mul 1
  --pseudo_target_type ref_mixed_reveal
  --ref_target_horizon 1
  --ref_target_local_radius 1
  --ref_target_mix_alpha 0.5
  --ref_target_loss feature_mse
  --ref_target_max_candidates 8
  --log_ref_target_debug
  --log_ref_target_freq 5
  --log_revealmar_losses
  --log_revealmar_loss_freq 5
  --preflight_mode
  --dist_url env://)

printf '%q ' "${TRAIN_CMD[@]}" > "${TRAIN_DIR}/run_args.txt"
"${TRAIN_CMD[@]}" 2>&1 | tee "${TRAIN_DIR}/train.log"

EVAL_CMD=(python main_revealmar.py
  --evaluate
  --model revealmar_base
  --img_size 256
  --vae_path "${VAE_PATH}"
  --vae_embed_dim 16
  --vae_stride 16
  --patch_size 1
  --data_path "${DATA_ROOT}"
  --resume "${TRAIN_DIR}"
  --output_dir "${EVAL_DIR}"
  --class_num 16
  --num_images 16
  --eval_bsz 4
  --num_iter 64
  --num_sampling_steps 100
  --cfg 1.0
  --cfg_schedule linear
  --temperature 1.0
  --num_workers 4
  --diffloss_d 6
  --diffloss_w 1024
  --diffusion_batch_mul 1
  --sampling_policy planner
  --pseudo_target_type ref_mixed_reveal
  --budget_mode hard
  --log_planner_sampling_debug
  --log_planner_sampling_steps 8
  --max_eval_batches 4
  --preflight_mode
  --dist_url env://)

printf '%q ' "${EVAL_CMD[@]}" > "${EVAL_DIR}/run_args.txt"
"${EVAL_CMD[@]}" 2>&1 | tee "${EVAL_DIR}/eval.log"

echo "Model-only checkpoint: ${MODEL_ONLY_INIT}/checkpoint-last.pth"
echo "Train log: ${TRAIN_DIR}/train.log"
echo "Eval log: ${EVAL_DIR}/eval.log"

grep -E "\[RevealMAR\]\[ref-target\]|\[RevealMAR\]\[planner-sampling\]|Traceback|RuntimeError|CUDA out of memory|NaN" "${TRAIN_DIR}/train.log" "${EVAL_DIR}/eval.log" || true
