#!/usr/bin/env bash
set -e
set -x

export TRAIN_EPOCHS=${TRAIN_EPOCHS:-1}
export WARMUP_EPOCHS=${WARMUP_EPOCHS:-1}
export TRAIN_MAX_STEPS=${TRAIN_MAX_STEPS:-200}
export TRAIN_RUN_NAME=${TRAIN_RUN_NAME:-train_ref_mixed_ep1_chain}
export EVAL_RUN_NAME=${EVAL_RUN_NAME:-eval_main_ep1_1000}
export EVAL_NUM_IMAGES=${EVAL_NUM_IMAGES:-1000}
export EVAL_CLASS_NUM=${EVAL_CLASS_NUM:-1000}
export EVAL_BSZ=${EVAL_BSZ:-32}
export EVAL_NUM_ITERS=${EVAL_NUM_ITERS:-64,128,256}
export EVAL_POLICIES=${EVAL_POLICIES:-baseline,confidence,entropy,planner}
export AUTO_COLLECT=${AUTO_COLLECT:-1}

bash scripts_server/run_base_train_then_eval.sh

if [ "${AUTO_COLLECT}" = "1" ]; then
  echo "[AUTO_COLLECT] Collecting base ep1 eval1000 results..."
  python scripts_server/collect_main_results.py \
    --root "${OUTPUT_ROOT:-/path/to/output/planmar_main}" \
    --models base \
    --skip_missing_models \
    --output_prefix base_ep1_1000 \
    --eval_name eval_main_ep1_1000
  echo "[AUTO_COLLECT] Done."
fi
