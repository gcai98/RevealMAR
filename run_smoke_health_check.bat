@echo off
setlocal enabledelayedexpansion

echo ============================================================
echo START SMOKE HEALTH CHECK: %DATE% %TIME%
echo ============================================================

set "CODE_DIR=C:\caogang\RevealMAR\RevealMAR"
cd /d "%CODE_DIR%"

echo ============================================================
echo STEP 1: PY_COMPILE
echo ============================================================
python -m py_compile tools/make_model_only_checkpoint.py
if ERRORLEVEL 1 exit /b 1
python -m py_compile main_revealmar.py models/revealmar.py util/revealmar_utils.py engine_mar.py
if ERRORLEVEL 1 exit /b 1
python -m py_compile eval_surrogate_validity.py eval_set_action_consistency.py eval_candidate_subset_stability.py eval_reveal_trajectory.py eval_early_step_intervention.py
if ERRORLEVEL 1 exit /b 1
python -m py_compile collect_p0_results.py collect_surrogate_validity.py collect_oracle_mismatch_results.py collect_budget_calibration_results.py collect_set_action_consistency_results.py collect_mixed_policy_ablation_results.py collect_same_parameter_control_results.py collect_reveal_trajectory_results.py collect_candidate_subset_ablation_results.py
if ERRORLEVEL 1 exit /b 1

echo ============================================================
echo STEP 2: REF TARGET SMOKE TRAIN
echo ============================================================
call run_ref_target_smoke_train.bat
if ERRORLEVEL 1 echo [WARN] ref target smoke train failed.

echo ============================================================
echo STEP 3: CANDIDATE SUBSET ABLATION
echo ============================================================
call run_candidate_subset_ablation.bat
if ERRORLEVEL 1 echo [WARN] candidate subset ablation failed.

echo ============================================================
echo STEP 4: SET ACTION CONSISTENCY
echo ============================================================
call run_set_action_consistency.bat
if ERRORLEVEL 1 echo [WARN] set action consistency failed.

echo ============================================================
echo STEP 5: REVEAL TRAJECTORY
echo ============================================================
call run_reveal_trajectory.bat
if ERRORLEVEL 1 echo [WARN] reveal trajectory failed.

echo ============================================================
echo END SMOKE HEALTH CHECK: %DATE% %TIME%
echo ============================================================
exit /b 0
