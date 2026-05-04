Oracle Mismatch: Paper-Aligned ref_* Targets
============================================

This smoke-scale experiment compares three paper-aligned pseudo utility target
families:

  ref_gt_reveal
  ref_pred_reveal
  ref_mixed_reveal

What it tests
-------------

The experiment asks whether planner training is sensitive to the reveal-first
counterfactual target source.

ref_gt_reveal uses a ground-truth latent token for the reveal-first
counterfactual.

ref_pred_reveal uses the model's current predicted latent token for the
reveal-first counterfactual.

ref_mixed_reveal blends the two:

  alpha * ref_gt_reveal + (1 - alpha) * ref_pred_reveal

Pipeline
--------

Run from the RevealMAR repo root:

  .\run_oracle_mismatch_ref_targets.bat

Then collect results with:

  python .\collect_oracle_mismatch_results.py --root "<repo_root>\p0_runs\oracle_mismatch_ref_targets"

Outputs are written under:

  <repo_root>\p0_runs\oracle_mismatch_ref_targets

Scope note
----------

This is a smoke-scale pipeline for validating the oracle mismatch experiment
infrastructure. It is not the final ImageNet-1K paper-scale result.

