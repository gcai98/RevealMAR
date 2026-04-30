#!/bin/bash
set -x
set -e

# =========================
# 0. Basic paths
# =========================

CODE_DIR=/root/autodl-tmp/RevealMAR-revealmar-dev

# 必须是包含 train/ 的上一级目录
DATA_ROOT=/root/autodl-tmp/imagenet/imagenet1k_imagefolder_full

PRETRAIN_ROOT=/root/autodl-tmp/pretrained_models
OUTPUT_DIR=/root/autodl-tmp/outputs/revealmar_train1_eval1000_planner_bs64
LOG_DIR=/root/autodl-tmp/outputs
GPU_LOG=${LOG_DIR}/gpu_mem_revealmar_train1_eval1000_planner_bs64.log
NOHUP_LOG=${LOG_DIR}/revealmar_train1_eval1000_planner_bs64.nohup.log

# 为了避免直接 resume 原始 MAR checkpoint 时继承 epoch/optimizer 状态，
# 我们构造一个“仅保留模型权重”的 finetune 初始化目录
FINETUNE_CKPT_DIR=/root/autodl-tmp/pretrained_models/mar/mar_base_finetune_init

mkdir -p ${LOG_DIR}
mkdir -p ${OUTPUT_DIR}
mkdir -p ${FINETUNE_CKPT_DIR}

cd ${CODE_DIR}

# =========================
# 1. Activate conda env
# =========================

source /root/miniconda3/etc/profile.d/conda.sh
conda activate revealmar

echo "PWD=$(pwd)"
echo "Start time:"
date
echo "Python=$(which python)"
python -V
echo "DATA_ROOT=${DATA_ROOT}"
echo "PRETRAIN_ROOT=${PRETRAIN_ROOT}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "GPU_LOG=${GPU_LOG}"

# =========================
# 2. Check paths
# =========================

if [ ! -d "${DATA_ROOT}/train" ]; then
    echo "ERROR: ${DATA_ROOT}/train does not exist."
    echo "Please set DATA_ROOT to the folder that contains train/."
    exit 1
fi

if [ ! -f "${PRETRAIN_ROOT}/vae/kl16.ckpt" ]; then
    echo "ERROR: ${PRETRAIN_ROOT}/vae/kl16.ckpt does not exist."
    exit 1
fi

if [ ! -f "${PRETRAIN_ROOT}/mar/mar_base/checkpoint-last.pth" ]; then
    echo "ERROR: ${PRETRAIN_ROOT}/mar/mar_base/checkpoint-last.pth does not exist."
    exit 1
fi

nvidia-smi || true

# =========================
# 3. Create finetune init checkpoint
# =========================

python - <<PY
import os
import torch

src = "${PRETRAIN_ROOT}/mar/mar_base/checkpoint-last.pth"
dst = "${FINETUNE_CKPT_DIR}/checkpoint-last.pth"

if not os.path.exists(dst):
    print(f"Creating finetune checkpoint: {dst}")
    ckpt = torch.load(src, map_location="cpu")
    slim = {}
    for k in ["model", "model_ema"]:
        if k in ckpt:
            slim[k] = ckpt[k]
    torch.save(slim, dst)
else:
    print(f"Finetune checkpoint already exists: {dst}")
PY

# =========================
# 4. Start nvidia-smi monitor
# =========================

(
  while true; do
    echo "===== time ====="
    date
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
      --format=csv,noheader,nounits
    echo "----------------------------------------"
    sleep 2
  done
) > "${GPU_LOG}" 2>&1 &

MONITOR_PID=$!

cleanup() {
  kill ${MONITOR_PID} 2>/dev/null || true
}
trap cleanup EXIT

# =========================
# 5. Train 1 epoch
# =========================

python main_revealmar.py \
    --model revealmar_base \
    --img_size 256 \
    --vae_path ${PRETRAIN_ROOT}/vae/kl16.ckpt \
    --vae_embed_dim 16 \
    --vae_stride 16 \
    --patch_size 1 \
    --data_path ${DATA_ROOT} \
    --resume ${FINETUNE_CKPT_DIR} \
    --diffloss_d 6 \
    --diffloss_w 1024 \
    --diffusion_batch_mul 1 \
    --epochs 1 \
    --warmup_epochs 1 \
    --batch_size 96 \
    --blr 1.0e-4 \
    --num_workers 8 \
    --save_last_freq 1 \
    --eval_freq 999999 \
    --pseudo_target_type mixed_reveal \
    --planner_loss_weight 0.3 \
    --candidate_pool_size 8 \
    --sampling_policy baseline \
    --mixed_policy_ratio 0.0 \
    --log_revealmar_losses \
    --log_revealmar_loss_freq 20 \
    --output_dir ${OUTPUT_DIR} \
    --dist_url env://

echo "Training finished:"
date

if [ ! -f "${OUTPUT_DIR}/checkpoint-last.pth" ]; then
    echo "ERROR: training did not produce ${OUTPUT_DIR}/checkpoint-last.pth"
    exit 1
fi

# =========================
# 6. Generate 1000 images with planner decoding
# =========================

python main_revealmar.py \
    --model revealmar_base \
    --img_size 256 \
    --vae_path ${PRETRAIN_ROOT}/vae/kl16.ckpt \
    --vae_embed_dim 16 \
    --vae_stride 16 \
    --patch_size 1 \
    --data_path ${DATA_ROOT} \
    --resume ${OUTPUT_DIR} \
    --diffloss_d 6 \
    --diffloss_w 1024 \
    --diffusion_batch_mul 1 \
    --evaluate \
    --eval_bsz 96 \
    --num_images 1000 \
    --num_iter 256 \
    --num_sampling_steps 100 \
    --cfg 2.9 \
    --cfg_schedule linear \
    --temperature 1.0 \
    --pseudo_target_type mixed_reveal \
    --planner_loss_weight 0.3 \
    --candidate_pool_size 8 \
    --sampling_policy planner \
    --mixed_policy_ratio 0.0 \
    --log_planner_sampling_debug \
    --log_planner_sampling_steps 8 \
    --output_dir ${OUTPUT_DIR} \
    --dist_url env://

echo "Evaluation finished:"
date
echo "Output dir: ${OUTPUT_DIR}"
echo "GPU log saved to: ${GPU_LOG}"