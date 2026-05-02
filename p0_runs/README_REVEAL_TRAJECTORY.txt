Reveal Trajectory And Early-Step Intervention
=============================================

Purpose
-------

Reveal trajectory visualization provides structural order evidence for
PlanMAR-S by recording which latent-grid tokens are selected at each decoding
step.

Early-step intervention tests whether early planner decisions matter by
replacing the first few planner-selected next-sets with random, confidence, or
entropy selections while keeping the planner budget schedule.

Scope
-----

This is smoke-scale infrastructure first. The trajectory runner uses 16 images
for compact visualization. The intervention runner uses the existing evaluation
path so FID / IS are computed normally when the run completes.

Commands
--------

From the RevealMAR repo root:

  .\run_reveal_trajectory.bat

  .\run_early_step_intervention.bat

  python .\collect_reveal_trajectory_results.py --trajectory_root "C:\caogang\RevealMAR\RevealMAR\p0_runs\reveal_trajectory" --intervention_root "C:\caogang\RevealMAR\RevealMAR\p0_runs\early_step_intervention"

Outputs
-------

Trajectory outputs:

  C:\caogang\RevealMAR\RevealMAR\p0_runs\reveal_trajectory

Intervention outputs:

  C:\caogang\RevealMAR\RevealMAR\p0_runs\early_step_intervention

Inspect:

  reveal_trajectory_details.json
  reveal_trajectory_summary.csv
  selected_frequency_all_steps.png
  selected_frequency_early_steps.png
  selected_frequency_late_steps.png
  reveal_trajectory_summary_all.csv
  early_step_intervention_summary.csv
