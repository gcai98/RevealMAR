Final PlanMAR-S Runbook
=======================

Recommended Workflow
--------------------

1. Local Windows smoke:

   run_smoke_health_check.bat

2. Linux cheap preflight:

   bash scripts_linux/run_linux_preflight.sh

3. Only after preflight passes, run paper-scale or extended experiments:

   bash scripts_linux/run_main_results.sh
   bash scripts_linux/run_budget_calibration.sh
   bash scripts_linux/run_oracle_mismatch_ref_targets.sh
   bash scripts_linux/run_mixed_policy_ablation.sh
   bash scripts_linux/run_same_parameter_control.sh
   bash scripts_linux/run_early_step_intervention.sh

Important Notes
---------------

Do not run expensive Linux experiments before preflight passes.

class_num must match the checkpoint class embedding size. For ImageNet-1K
checkpoints, use class_num=1000 even for small evals.

The current eval logic requires num_images to be divisible by class_num.

max_train_steps and max_eval_batches are preflight-only safety controls. Leave
them unset or set to -1 for normal runs.

Preflight truncated eval intentionally skips FID/IS. Do not use preflight
metrics as paper results.

Path Checklist
--------------

Before Linux runs, verify paths at the top of scripts_linux/common.sh or
override them as environment variables:

  CODE_DIR
  DATA_ROOT
  TINY_DATA_ROOT
  PRETRAIN_ROOT
  VAE_PATH
  RESUME
  SOURCE_CHECKPOINT
  OUTPUT_ROOT
