Budget Calibration Smoke Pipeline
=================================

This smoke-scale experiment tests score-derived budget calibration for
planner-soft sampling. It is meant to diagnose whether the learned ranking is
useful once the reveal budget is calibrated, not to tune final paper-scale
ImageNet-1K numbers.

Controls
--------

budget_temperature, or tau, is the softmax temperature used when converting
planner scores into a concentration value. Lower tau sharpens the distribution
and can increase the score-derived budget.

budget_score_scale multiplies planner scores before the concentration softmax.
Larger scale also sharpens the score distribution when score magnitudes are
small.

budget_ema_beta smooths the soft-budget sequence within one sampling call. A
value of 0 disables smoothing and preserves the original no-EMA behavior.

budget_min and budget_max clamp the score-derived reveal count. When
budget_max is -1, the existing cosine schedule reveal count is used as the
upper bound.

How to run
----------

From the RevealMAR repo root:

  .\run_budget_calibration.bat

Then collect the results:

  python .\collect_budget_calibration_results.py --root "C:\caogang\RevealMAR\RevealMAR\p0_runs\budget_calibration"

Outputs are written under:

  C:\caogang\RevealMAR\RevealMAR\p0_runs\budget_calibration

Inspect:

  budget_calibration_summary.csv
  budget_calibration_summary.json
  each run's eval.log and config.json
