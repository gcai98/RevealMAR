#!/bin/bash
set -x
set -e

# ============================================================
# One-click launcher for PlanMAR-S print-only evaluation suite.
#
# This launcher:
# - activates conda env
# - starts GPU monitor
# - runs planmar_eval_suite_print_only.py
# - stores all printed summaries in nohup log
# ============================================================

CODE_DIR=/root/autodl-tmp/RevealMAR-revealmar-dev
EVAL_ROOT=/root/autodl-tmp/outputs/planmar_paper_eval_print_only
LOG_ROOT=/root/autodl-tmp/outputs
GPU_LOG=${EVAL_ROOT}/gpu_mem_eval_suite.log

mkdir -p ${EVAL_ROOT}
mkdir -p ${LOG_ROOT}

cd ${CODE_DIR}

source /root/miniconda3/etc/profile.d/conda.sh
conda activate revealmar

export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1

echo "Start PlanMAR-S print-only eval suite:"
date
echo "PWD=$(pwd)"
echo "Python=$(which python)"
python -V

# GPU monitor
(
  while true; do
    echo "===== time ====="
    date
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
      --format=csv,noheader,nounits
    echo "----------------------------------------"
    sleep 5
  done
) > "${GPU_LOG}" 2>&1 &

MONITOR_PID=$!

cleanup() {
  kill ${MONITOR_PID} 2>/dev/null || true
}
trap cleanup EXIT

python planmar_eval_suite_print_only.py \
  --code_dir /root/autodl-tmp/RevealMAR-revealmar-dev \
  --data_root /root/autodl-tmp/imagenet/imagenet1k_imagefolder_full \
  --pretrain_root /root/autodl-tmp/pretrained_models \
  --planmar_ckpt /root/autodl-tmp/outputs/revealmar_train1_eval1000_planner_bs64 \
  --baseline_ckpt /root/autodl-tmp/outputs/revealmar_train1_eval1000_planner_bs64 \
  --eval_root /root/autodl-tmp/outputs/planmar_paper_eval_print_only \
  --num_images 1000 \
  --eval_bsz 96 \
  --num_iters 64,128,256 \
  --methods baseline,planner \
  --continue_on_error

echo "Eval suite finished:"
date
echo "Results root: ${EVAL_ROOT}"
echo "GPU log: ${GPU_LOG}"
