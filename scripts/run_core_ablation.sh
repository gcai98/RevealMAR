#!/usr/bin/env bash
set -euo pipefail

# P0-4 Core Ablation Table runner. Evaluation only; no training commands.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR=${CODE_DIR:-"$(cd "${SCRIPT_DIR}/.." && pwd)"}
DATA_PATH=${DATA_PATH:-"${CODE_DIR}/data/imagenet"}
VAE_PATH=${VAE_PATH:-"${CODE_DIR}/pretrained_models/vae/kl16.ckpt"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"${CODE_DIR}/outputs/p0_tables"}
BASE_RESUME=${BASE_RESUME:-""}
PLANMAR_RESUME=${PLANMAR_RESUME:-""}
NUM_IMAGES=${NUM_IMAGES:-1000}
EVAL_BSZ=${EVAL_BSZ:-32}
NUM_ITER=${NUM_ITER:-64}
NUM_WORKERS=${NUM_WORKERS:-8}

PLANNER_HEAD_ONLY_RESUME=${PLANNER_HEAD_ONLY_RESUME:-""}
GT_RESUME=${GT_RESUME:-"${PLANMAR_RESUME}"}
PRED_RESUME=${PRED_RESUME:-"${PLANMAR_RESUME}"}
MIXED_RESUME=${MIXED_RESUME:-"${PLANMAR_RESUME}"}

MODEL=${MODEL:-revealmar_base}
DIFFLOSS_D=${DIFFLOSS_D:-6}
DIFFLOSS_W=${DIFFLOSS_W:-1024}
NUM_SAMPLING_STEPS=${NUM_SAMPLING_STEPS:-100}
CFG=${CFG:-1.0}
PLANNER_LOSS_WEIGHT=${PLANNER_LOSS_WEIGHT:-0.3}
CANDIDATE_POOL_SIZE=${CANDIDATE_POOL_SIZE:-8}

TABLE_NAME="core"
TABLE_DIR="${OUTPUT_ROOT}/core_ablation"
mkdir -p "${TABLE_DIR}"

write_config() {
  local run_dir="$1"
  local variant="$2"
  local method="$3"
  local reveal_strategy="$4"
  local budget_rule="$5"
  local resume="$6"
  local sampling_policy="$7"
  local pseudo_target_type="$8"
  local budget_mode="$9"
  local timestamp
  timestamp="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

  python - "$run_dir" "$TABLE_NAME" "$variant" "$method" "$reveal_strategy" "$budget_rule" "$resume" "$DATA_PATH" "$NUM_IMAGES" "$EVAL_BSZ" "$NUM_ITER" "$sampling_policy" "$pseudo_target_type" "$budget_mode" "$timestamp" "$MODEL" "$DIFFLOSS_D" "$DIFFLOSS_W" "$NUM_SAMPLING_STEPS" "$CFG" <<'PY'
import json
import sys
from pathlib import Path

(
    run_dir, table, variant, method, reveal_strategy, budget_rule, resume,
    data_path, num_images, eval_bsz, num_iter, sampling_policy,
    pseudo_target_type, budget_mode, timestamp, model, diffloss_d,
    diffloss_w, num_sampling_steps, cfg
) = sys.argv[1:]

payload = {
    "table": table,
    "variant": variant,
    "method": method,
    "reveal_strategy": reveal_strategy,
    "budget_rule": budget_rule,
    "resume": resume,
    "data_path": data_path,
    "num_images": int(num_images),
    "eval_bsz": int(eval_bsz),
    "num_iter": int(num_iter),
    "sampling_policy": sampling_policy,
    "pseudo_target_type": pseudo_target_type,
    "budget_mode": budget_mode,
    "timestamp": timestamp,
    "extra_params": {
        "model": model,
        "diffloss_d": int(diffloss_d),
        "diffloss_w": int(diffloss_w),
        "num_sampling_steps": num_sampling_steps,
        "cfg": float(cfg),
    },
}
Path(run_dir, "config.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
PY
}

run_variant() {
  local variant="$1"
  local method="$2"
  local reveal_strategy="$3"
  local budget_rule="$4"
  local resume="$5"
  local sampling_policy="$6"
  local pseudo_target_type="$7"
  local budget_mode="$8"

  if [[ -z "${resume}" ]]; then
    echo "[WARN] Skipping ${variant}: resume path is empty."
    return 0
  fi

  local run_name
  run_name="$(echo "${variant}" | tr '[:upper:] ' '[:lower:]_' | tr -cd '[:alnum:]_-')"
  local run_dir="${TABLE_DIR}/${run_name}_n${NUM_IMAGES}_iter${NUM_ITER}"
  mkdir -p "${run_dir}"

  local cmd=(
    python main_revealmar.py
    --evaluate
    --model "${MODEL}"
    --data_path "${DATA_PATH}"
    --vae_path "${VAE_PATH}"
    --resume "${resume}"
    --diffloss_d "${DIFFLOSS_D}"
    --diffloss_w "${DIFFLOSS_W}"
    --num_images "${NUM_IMAGES}"
    --eval_bsz "${EVAL_BSZ}"
    --num_iter "${NUM_ITER}"
    --num_sampling_steps "${NUM_SAMPLING_STEPS}"
    --cfg "${CFG}"
    --num_workers "${NUM_WORKERS}"
    --output_dir "${run_dir}"
    --sampling_policy "${sampling_policy}"
    --pseudo_target_type "${pseudo_target_type}"
    --budget_mode "${budget_mode}"
    --planner_loss_weight "${PLANNER_LOSS_WEIGHT}"
    --candidate_pool_size "${CANDIDATE_POOL_SIZE}"
  )

  printf '%q ' "${cmd[@]}" > "${run_dir}/run_args.txt"
  printf '\n' >> "${run_dir}/run_args.txt"
  write_config "${run_dir}" "${variant}" "${method}" "${reveal_strategy}" "${budget_rule}" "${resume}" "${sampling_policy}" "${pseudo_target_type}" "${budget_mode}"

  echo "[RUN] ${variant}"
  (cd "${CODE_DIR}" && "${cmd[@]}") > "${run_dir}/eval.log" 2>&1
}

if [[ -z "${BASE_RESUME}" && -n "${PLANMAR_RESUME}" ]]; then
  echo "[WARN] BASE_RESUME is empty; using PLANMAR_RESUME for MAR Baseline evaluation."
  BASE_RESUME="${PLANMAR_RESUME}"
fi

if [[ -z "${PLANNER_HEAD_ONLY_RESUME}" ]]; then
  echo "[WARN] PLANNER_HEAD_ONLY_RESUME is empty; using PLANMAR_RESUME for Planner Head only."
  PLANNER_HEAD_ONLY_RESUME="${PLANMAR_RESUME}"
fi

run_variant "MAR Baseline" "MAR Baseline" "baseline" "fixed" "${BASE_RESUME}" "baseline" "none" "hard"
run_variant "Planner Head only" "PlanMAR-S" "planner-head-only" "fixed" "${PLANNER_HEAD_ONLY_RESUME}" "planner" "none" "hard"
run_variant "GT future-aware target" "PlanMAR-S" "gt-reveal" "fixed" "${GT_RESUME}" "planner" "gt_reveal" "hard"
run_variant "Pred-reveal" "PlanMAR-S" "pred-reveal" "fixed" "${PRED_RESUME}" "planner" "pred_reveal" "hard"
run_variant "Mixed-reveal" "PlanMAR-S" "mixed-reveal" "fixed" "${MIXED_RESUME}" "planner" "mixed_reveal" "hard"
run_variant "Mixed-reveal + score-derived budget" "PlanMAR-S" "mixed-reveal" "score-derived" "${MIXED_RESUME}" "planner" "mixed_reveal" "soft"
