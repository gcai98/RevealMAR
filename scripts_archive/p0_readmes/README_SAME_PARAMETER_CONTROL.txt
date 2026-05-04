Same-Parameter Control Experiment
=================================

Purpose
-------

This smoke-scale experiment checks whether PlanMAR-S behavior comes from useful
surrogate supervision rather than merely adding a small planner head or extra
parameters.

Control targets
---------------

control_zero keeps the planner head and planner loss path, but supervises all
candidate pseudo targets with zero utility.

control_random keeps the same planner head and planner loss path, but
supervises candidate pseudo targets with deterministic random noise.

control_shuffle first builds the existing mixed_reveal target, then shuffles
target values across candidates within each sample. This preserves the target
value distribution while breaking candidate-target correspondence.

ref_mixed_reveal is included as a paper-aligned positive control.

Scope
-----

This is a smoke-scale pipeline first, not a final paper-scale ImageNet-1K
result.

How to run
----------

From the RevealMAR repo root:

  .\run_same_parameter_control.bat

Then collect:

  python .\collect_same_parameter_control_results.py --root "<repo_root>\p0_runs\same_parameter_control"

Outputs are written under:

  <repo_root>\p0_runs\same_parameter_control

Inspect:

  same_parameter_control_summary.csv
  same_parameter_control_summary.json
  train_*\train.log
  eval_*\eval.log

