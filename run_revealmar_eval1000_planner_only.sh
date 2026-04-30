#!/bin/bash
set -x
set -e

# =========================
# 0. Basic paths
# =========================

CODE_DIR=/root/autodl-tmp/RevealMAR-revealmar-dev
DATA_ROOT=/root/autodl-tmp/imagenet/imagenet1k_imagefolder_full
PRETRAIN_ROOT=/root/autodl-tmp/pretrained_models

# 这里必须是你已经训练完的输出目录
OUTPUT_DIR=/root/autodl-tmp/outputs/revealmar_train1_eval1000_planner_bs64

LOG_DIR=/root/autodl-tmp/outputs
GPU_LOG=${LOG_DIR}/gpu_mem_revealmar_eval1000_planner_only.log

mkdir -p ${LOG_DIR}
mkdir -p ${OUTPUT_DIR}

cd ${CODE_DIR}

# =========================
# 1. Activate conda env
# =========================

source /root/miniconda3/etc/profile.d/conda.sh
conda activate revealmar

# 避免 libgomp: Invalid value for OMP_NUM_THREADS
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1

echo "PWD=$(pwd)"
echo "Start eval time:"
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
    exit 1
fi

if [ ! -f "${PRETRAIN_ROOT}/vae/kl16.ckpt" ]; then
    echo "ERROR: ${PRETRAIN_ROOT}/vae/kl16.ckpt does not exist."
    exit 1
fi

if [ ! -f "${OUTPUT_DIR}/checkpoint-last.pth" ]; then
    echo "ERROR: ${OUTPUT_DIR}/checkpoint-last.pth does not exist."
    echo "Your training checkpoint was not found."
    exit 1
fi

python -m py_compile main_revealmar.py

nvidia-smi || true

# =========================
# 3. Start nvidia-smi monitor
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
# 4. Evaluate / Generate 1000 images with planner decoding
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