Mixed-Policy Exposure Ablation
==============================

This smoke-scale experiment tests whether mixed-policy exposure helps mitigate
policy-induced state shift during PlanMAR-S training.

What it tests
-------------

mixed_policy_ratio controls the probability of rolling in with the learned
planner instead of the reference/baseline policy, according to the current
RevealMAR implementation.

The ablation trains small runs with:

  ratio0p0
  ratio0p25
  ratio0p5
  ratio0p75

Each run uses ref_mixed_reveal targets, then evaluates:

  baseline decoding
  planner hard budget
  planner soft budget

Scope note
----------

Mixed-policy exposure is for reducing policy-induced state shift. It does not
make the surrogate objective fully equivalent to the true test-time objective.

This is a smoke-scale pipeline first, not a final paper-scale ImageNet-1K
result.

How to run
----------

From the RevealMAR repo root:

  .\run_mixed_policy_ablation.bat

Then collect:

  python .\collect_mixed_policy_ablation_results.py --root "<repo_root>\p0_runs\mixed_policy_ablation"

Outputs are written under:

  <repo_root>\p0_runs\mixed_policy_ablation

Inspect:

  mixed_policy_ablation_summary.csv
  mixed_policy_ablation_summary.json
  train_*\train.log
  eval_*\eval.log

