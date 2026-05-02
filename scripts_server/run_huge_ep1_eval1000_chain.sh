#!/usr/bin/env bash
set -e
set -x

export TRAIN_EPOCHS=${TRAIN_EPOCHS:-1}
export WARMUP_EPOCHS=${WARMUP_EPOCHS:-1}
export TRAIN_RUN_NAME=${TRAIN_RUN_NAME:-train_ref_mixed_ep1_chain}
export EVAL_RUN_NAME=${EVAL_RUN_NAME:-eval_main_ep1_1000}
export EVAL_NUM_IMAGES=${EVAL_NUM_IMAGES:-1000}
export EVAL_CLASS_NUM=${EVAL_CLASS_NUM:-1000}
export EVAL_BSZ=${EVAL_BSZ:-32}
export EVAL_NUM_ITERS=${EVAL_NUM_ITERS:-64,128,256}
export EVAL_POLICIES=${EVAL_POLICIES:-baseline,confidence,entropy,planner}

bash scripts_server/run_huge_train_then_eval.sh
