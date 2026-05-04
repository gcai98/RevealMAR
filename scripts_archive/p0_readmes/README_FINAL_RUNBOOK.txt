Final PlanMAR-S Runbook
=======================

Recommended Workflow
--------------------

Do not use root-level run scripts. Final experiments should be launched from
scripts_server/. Linux preflight should be launched from scripts_linux/.
Windows local debugging, if needed, should use scripts_win/.

1. Local Windows smoke:

   scripts_win/run_main_results.bat

2. Linux cheap preflight:

   bash scripts_linux/run_linux_preflight.sh

3. Only after preflight passes, run the server workflow for one model per server:

   bash scripts_server/run_base_train_then_eval.sh
   bash scripts_server/run_large_train_then_eval.sh
   bash scripts_server/run_huge_train_then_eval.sh

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
