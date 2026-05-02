Linux Preflight
===============

Purpose
-------

Linux preflight is a cheap sanity check before renting time for expensive
ImageNet-1K PlanMAR-S experiments. It is not a paper-scale run.

What It Checks
--------------

The preflight script runs:

  py_compile on core files, evaluators, collectors, and checkpoint utility
  model-only checkpoint creation
  20 bounded training steps with ref_mixed_reveal targets
  a tiny planner eval with at most 4 generation batches

This validates the Linux environment, CUDA stack, paths, data loading,
checkpoint loading, reference target construction, and planner sampling.

Bounded Controls
----------------

The following flags default to normal behavior when unset:

  --max_train_steps -1
  --max_eval_batches -1
  --preflight_mode

Only positive max_* values truncate train/eval loops.

Run
---

From the repo root on Linux:

  bash scripts_linux/run_linux_preflight.sh

Default output:

  /root/autodl-tmp/outputs/planmar_linux_preflight

Inspect:

  train_ref_mixed_preflight/train.log
  eval_planner_preflight/eval.log
  model_only_init/checkpoint-last.pth
