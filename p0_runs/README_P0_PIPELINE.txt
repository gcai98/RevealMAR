P0 Smoke Pipeline
=================

Run from the RevealMAR repo root with:

  .\run_p0_all.bat

What it runs
------------

run_p0_all.bat runs the current P0 smoke pipeline in this order:

1. run_main_results.bat
2. collect_p0_results.py
3. run_surrogate_validity.bat
4. collect_surrogate_validity.py

Main result outputs
-------------------

Main policy evaluation outputs are saved under:

  C:\caogang\RevealMAR\RevealMAR\p0_runs\main_results_trained_ckpt_full_policies

Inspect these files after running:

  p0_summary.csv
  p0_summary.json
  each variant subfolder's eval.log
  each variant subfolder's config.json
  each variant subfolder's run_args.txt

Surrogate validity outputs
--------------------------

Surrogate validity outputs are saved under:

  C:\caogang\RevealMAR\RevealMAR\p0_runs\surrogate_validity_trained_ckpt

Inspect these files after running:

  surrogate_validity.csv
  surrogate_validity.json
  surrogate_validity_summary.csv
  surrogate_validity_summary.json
  eval.log
  config.json
  run_args.txt

Scope note
----------

This is a smoke-test P0 pipeline for checking the current PlanMAR-S workflow,
baselines, logging, checkpoint loading, and surrogate-validity integration.
It is not the final paper-scale ImageNet-1K evaluation.
