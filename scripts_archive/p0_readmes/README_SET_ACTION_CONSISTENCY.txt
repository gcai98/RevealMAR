Set-Action Consistency Smoke Evaluator
======================================

This evaluator supports the paper claim that independent token utility ranking
followed by top-k selection is a tractable approximation to set-action
selection.

What is compared
----------------

individual_topk is the deployed PlanMAR-S method: rank tokens independently by
planner score and select the top-k next reveal actions.

greedy_marginal is an expensive local proxy baseline. It greedily builds a set
using rollout-proxy token gains with a spatial redundancy penalty. It is not a
true global optimal set search.

pairwise_deredundancy ranks by planner score while penalizing spatial
redundancy against already selected candidates. This tests whether explicit
spatial de-redundancy matters for small next-action sets.

random_set, confidence_topk, and entropy_topk are lightweight smoke baselines.

Scope
-----

This is a smoke-scale mechanism evaluator, not a final paper-scale ImageNet-1K
result. It uses a small candidate pool and local teacher-forced proxy gains.

How to run
----------

From the RevealMAR repo root:

  .\run_set_action_consistency.bat

Then collect:

  python .\collect_set_action_consistency_results.py --root "<repo_root>\p0_runs\set_action_consistency"

Outputs are written under:

  <repo_root>\p0_runs\set_action_consistency

Inspect:

  set_action_consistency.csv
  set_action_consistency.json
  set_action_consistency_details.json
  set_action_consistency_summary.csv
  set_action_consistency_summary.json
  eval.log

