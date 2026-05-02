#!/usr/bin/env bash
set -e
set -x
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"
activate_env
check_path "${CODE_DIR}" "CODE_DIR"
cd "${CODE_DIR}"

python -m py_compile tools/make_model_only_checkpoint.py
python -m py_compile main_revealmar.py models/revealmar.py util/revealmar_utils.py engine_mar.py
python -m py_compile eval_surrogate_validity.py eval_set_action_consistency.py eval_candidate_subset_stability.py eval_reveal_trajectory.py eval_early_step_intervention.py
python -m py_compile collect_p0_results.py collect_surrogate_validity.py collect_oracle_mismatch_results.py collect_budget_calibration_results.py collect_set_action_consistency_results.py collect_mixed_policy_ablation_results.py collect_same_parameter_control_results.py collect_reveal_trajectory_results.py collect_candidate_subset_ablation_results.py

bash scripts_linux/run_ref_target_smoke_train.sh
bash scripts_linux/run_candidate_subset_ablation.sh
bash scripts_linux/run_set_action_consistency.sh
bash scripts_linux/run_reveal_trajectory.sh
