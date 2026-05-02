Cross-Platform PlanMAR-S Runners
================================

Purpose
-------

Windows .bat scripts are intended for local/lab smoke debugging. Linux scripts
under scripts_linux/ are intended for final server runs and are easier to adapt
for paper-scale jobs.

Path Editing
------------

Edit paths at the top of scripts_linux/common.sh or override them with
environment variables:

  CODE_DIR=/root/autodl-tmp/RevealMAR-revealmar-dev
  DATA_ROOT=/root/autodl-tmp/imagenet/imagenet1k_imagefolder_full
  TINY_DATA_ROOT=/root/autodl-tmp/tiny-imagenet-200-1percent
  PRETRAIN_ROOT=/root/autodl-tmp/pretrained_models
  OUTPUT_ROOT=/root/autodl-tmp/outputs/planmar_p0

Windows scripts keep their C:\caogang\... paths near the top.

Model-Only Init
---------------

Scripts that need a model-only initialization checkpoint use:

  tools/make_model_only_checkpoint.py

This avoids fragile batch echo blocks and strips optimizer / epoch state while
keeping model and model_ema.

Recommended Smoke Order
-----------------------

Run the cheap smoke health check first:

  .\run_smoke_health_check.bat

or on Linux:

  bash scripts_linux/run_smoke_health_check.sh

The smoke health check runs:

  py_compile checks
  ref target smoke training
  candidate subset ablation
  set-action consistency
  reveal trajectory

Expensive Runs
--------------

After smoke passes, run heavier scripts such as:

  run_main_results
  run_budget_calibration
  run_oracle_mismatch_ref_targets
  run_mixed_policy_ablation
  run_same_parameter_control
  run_early_step_intervention

These are still configured as smoke-scale by default unless the script name or
edited settings clearly indicate full paper-scale evaluation.
