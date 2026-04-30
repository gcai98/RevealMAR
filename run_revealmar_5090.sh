cd /root/autodl-tmp/RevealMAR

cat > run_revealmar_5090.sh <<'EOF'
#!/bin/bash
set -e

# ============================================================
# RevealMAR single-GPU script for RTX 5090
# Smoke run:
#   1) train for 1 epoch
#   2) evaluate and generate 1000 images
#
# Only two differences from final run:
#   EPOCHS=1      (final should be 50)
#   NUM_IMAGES=1000 (final should be 50000)
# ============================================================

export CUDA_VISIBLE_DEVICES=0

# -------------------------
# Paths
# -------------------------
PROJECT_DIR="/root/autodl-tmp/RevealMAR"
DATA_PATH="/root/autodl-tmp/imagenet/imagenet1k_imagefolder_full"
VAE_PATH="/root/autodl-tmp/pretrained_models/vae/kl16.ckpt"
BASE_RESUME_DIR="/root/autodl-tmp/pretrained_models/mar/mar_base"

OUTPUT_ROOT="/root/autodl-tmp/RevealMAR_outputs"
EXP_NAME="revealmar_base_5090_smoke"
OUTPUT_DIR="${OUTPUT_ROOT}/${EXP_NAME}"

mkdir -p "${OUTPUT_DIR}"

cd "${PROJECT_DIR}"

# -------------------------
# Core model settings
# -------------------------
MODEL="revealmar_base"

# Keep consistent with mar_base
DIFFLOSS_D="6"
DIFFLOSS_W="1024"

# -------------------------
# Final RevealMAR settings
# (keep these same as final experiment)
# -------------------------
PSEUDO_TARGET_TYPE="mixed_reveal"
PLANNER_LOSS_WEIGHT="0.3"
CANDIDATE_POOL_SIZE="8"

SAMPLING_POLICY="planner"
CANDIDATE_SELECTION_MODE="mixed"
MIXED_POLICY_RATIO="0.5"

PLANNER_HIDDEN_DIM="128"
BUDGET_MODE="soft"

# -------------------------
# Smoke-only differences
# -------------------------
EPOCHS="1"
NUM_IMAGES="1000"

# -------------------------
# Batch / eval settings
# -------------------------
TRAIN_BATCH_SIZE="8"
EVAL_BSZ="64"

# -------------------------
# Generation settings
# -------------------------
NUM_ITER="256"
NUM_SAMPLING_STEPS="100"
CFG="2.9"

echo "============================================================"
echo "RevealMAR 5090 smoke run"
echo "PROJECT_DIR=${PROJECT_DIR}"
echo "DATA_PATH=${DATA_PATH}"
echo "VAE_PATH=${VAE_PATH}"
echo "BASE_RESUME_DIR=${BASE_RESUME_DIR}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "MODEL=${MODEL}"
echo "EPOCHS=${EPOCHS}"
echo "NUM_IMAGES=${NUM_IMAGES}"
echo "============================================================"

# -------------------------
# Basic checks
# -------------------------
if [ ! -d "${DATA_PATH}/train" ]; then
  echo "[ERROR] train folder not found: ${DATA_PATH}/train"
  exit 1
fi

if [ ! -d "${DATA_PATH}/test" ]; then
  echo "[ERROR] test folder not found: ${DATA_PATH}/test"
  exit 1
fi

if [ ! -f "${VAE_PATH}" ]; then
  echo "[ERROR] VAE checkpoint not found: ${VAE_PATH}"
  exit 1
fi

if [ ! -f "${BASE_RESUME_DIR}/checkpoint-last.pth" ]; then
  echo "[ERROR] MAR checkpoint not found: ${BASE_RESUME_DIR}/checkpoint-last.pth"
  exit 1
fi

# ============================================================
# 1) Train
# ============================================================
echo ""
echo "============================================================"
echo "[1/2] Start training"
echo "============================================================"

python main_revealmar.py \
  --model "${MODEL}" \
  --data_path "${DATA_PATH}" \
  --vae_path "${VAE_PATH}" \
  --resume "${BASE_RESUME_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --diffloss_d "${DIFFLOSS_D}" \
  --diffloss_w "${DIFFLOSS_W}" \
  --epochs "${EPOCHS}" \
  --batch_size "${TRAIN_BATCH_SIZE}" \
  --save_last_freq 1 \
  --eval_freq 999999 \
  --pseudo_target_type "${PSEUDO_TARGET_TYPE}" \
  --planner_loss_weight "${PLANNER_LOSS_WEIGHT}" \
  --candidate_pool_size "${CANDIDATE_POOL_SIZE}" \
  --planner_hidden_dim "${PLANNER_HIDDEN_DIM}" \
  --budget_mode "${BUDGET_MODE}" \
  --mixed_policy_ratio "${MIXED_POLICY_RATIO}" \
  --log_revealmar_losses \
  --log_revealmar_loss_freq 20

echo ""
echo "============================================================"
echo "[1/2] Training finished"
echo "============================================================"

# ============================================================
# 2) Evaluate
# ============================================================
# Use the checkpoint saved in OUTPUT_DIR after training
echo ""
echo "============================================================"
echo "[2/2] Start evaluation"
echo "============================================================"

python main_revealmar.py \
  --model "${MODEL}" \
  --data_path "${DATA_PATH}" \
  --vae_path "${VAE_PATH}" \
  --resume "${OUTPUT_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --diffloss_d "${DIFFLOSS_D}" \
  --diffloss_w "${DIFFLOSS_W}" \
  --evaluate \
  --num_images "${NUM_IMAGES}" \
  --eval_bsz "${EVAL_BSZ}" \
  --num_iter "${NUM_ITER}" \
  --num_sampling_steps "${NUM_SAMPLING_STEPS}" \
  --cfg "${CFG}" \
  --pseudo_target_type "${PSEUDO_TARGET_TYPE}" \
  --planner_loss_weight "${PLANNER_LOSS_WEIGHT}" \
  --candidate_pool_size "${CANDIDATE_POOL_SIZE}" \
  --sampling_policy "${SAMPLING_POLICY}" \
  --candidate_selection_mode "${CANDIDATE_SELECTION_MODE}" \
  --planner_hidden_dim "${PLANNER_HIDDEN_DIM}" \
  --budget_mode "${BUDGET_MODE}" \
  --mixed_policy_ratio "${MIXED_POLICY_RATIO}"

echo ""
echo "============================================================"
echo "[DONE] train + evaluate finished"
echo "Output dir: ${OUTPUT_DIR}"
echo "============================================================"
EOF