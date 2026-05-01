Paper-Aligned Reference-Policy Targets
======================================

The old pseudo target types remain backward-compatible heuristic targets:

  gt_reveal
  pred_reveal
  mixed_reveal

The new paper-aligned target types are:

  ref_gt_reveal
  ref_pred_reveal
  ref_mixed_reveal

What they mean
--------------

ref_gt_reveal estimates the local future utility of revealing a candidate token
first using the ground-truth latent token for that reveal.

ref_pred_reveal estimates the same utility using the model's current diffusion
prediction for the candidate latent token.

ref_mixed_reveal combines the two utilities:

  alpha * ref_gt_reveal + (1 - alpha) * ref_pred_reveal

The utility is a local future loss reduction under a fixed reference
continuation policy:

  default local future feature-MSE loss
  minus
  reveal-candidate-first local future feature-MSE loss

The local neighborhood currently uses Chebyshev distance on the latent token
grid with --ref_target_local_radius.

Supported horizon
-----------------

--ref_target_horizon 1 is currently supported.

If --ref_target_horizon is greater than 1, training raises:

  ref_target_horizon > 1 is not implemented yet for training targets

This avoids silently pretending that multi-step reference rollouts are already
implemented.

Smoke test
----------

Run from the RevealMAR repo root:

  .\run_ref_target_smoke_train.bat

The smoke script is only for checking that ref_mixed_reveal target construction
and planner loss wiring execute. It is not intended for final paper-scale
ImageNet-1K training or evaluation.
