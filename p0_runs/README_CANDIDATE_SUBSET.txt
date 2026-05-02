Candidate Subset / Proposal Bias Ablation
=========================================

Why candidate subsets exist
---------------------------

PlanMAR-S builds pseudo utility targets only on a candidate subset C_t because
computing counterfactual targets for every masked token is too expensive.
Candidate subset construction is a tractable approximation for surrogate
supervision, not the decision rule itself.

Modes
-----

topk selects the highest scoring masked tokens.

random uniformly proposes masked tokens using a deterministic pseudo-random
order.

uncertainty selects tokens with the largest score-based proxy. This preserves
the old mixed-mode biased component while exposing it as an explicit mode.

spatial uses greedy farthest-point selection on the latent token grid.

mixed combines random, uncertainty-biased, and spatially diverse proposals to
reduce proposal bias. The default ratios are:

  random 0.25
  uncertainty 0.50
  spatial 0.25

Scope
-----

This is smoke-scale infrastructure first, not a final paper-scale result.

How to run
----------

From the RevealMAR repo root:

  .\run_candidate_subset_ablation.bat

Then collect:

  python .\collect_candidate_subset_ablation_results.py --root "C:\caogang\RevealMAR\RevealMAR\p0_runs\candidate_subset_ablation"

Outputs
-------

  candidate_subset_stability.csv
  candidate_subset_stability.json
  candidate_subset_stability_details.json
  candidate_subset_ablation_summary.csv
  candidate_subset_ablation_summary.json
