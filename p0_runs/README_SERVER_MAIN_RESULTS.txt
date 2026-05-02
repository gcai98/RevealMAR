Server Main Results Workflow
============================

Purpose
-------

These scripts run paper-scale PlanMAR-S main-result experiments on rented Linux
AutoDL servers. This is a multi-server workflow: one server runs one model.

Do not run base, large, and huge together on one server.

Recommended Order
-----------------

1. First run the cheap Linux preflight:

   bash scripts_linux/run_linux_preflight.sh

2. After preflight passes, choose exactly one model for the current server:

   bash scripts_server/run_base_train_then_eval.sh
   bash scripts_server/run_large_train_then_eval.sh
   bash scripts_server/run_huge_train_then_eval.sh

The top-level scripts run readiness checks, train ref_mixed_reveal for one
epoch, then evaluate baseline, confidence, entropy, and planner at 64, 128, and
256 decoding iterations.

AutoDL Auto-Shutdown
--------------------

Shutdown is disabled by default.

Run without shutdown:

   nohup bash scripts_server/run_huge_train_then_eval.sh > /root/autodl-tmp/outputs/planmar_main/huge_full.nohup.log 2>&1 &

Shutdown on success:

   AUTO_SHUTDOWN=1 nohup bash scripts_server/run_huge_train_then_eval.sh > /root/autodl-tmp/outputs/planmar_main/huge_full.nohup.log 2>&1 &

Shutdown on success or error:

   AUTO_SHUTDOWN=1 SHUTDOWN_ON_ERROR=1 nohup bash scripts_server/run_huge_train_then_eval.sh > /root/autodl-tmp/outputs/planmar_main/huge_full.nohup.log 2>&1 &

Only the top-level run_*_train_then_eval.sh scripts can shut down the machine.
The lower-level train and eval scripts never call shutdown.

Outputs
-------

Default output root:

   /root/autodl-tmp/outputs/planmar_main

Per-model outputs:

   /root/autodl-tmp/outputs/planmar_main/base
   /root/autodl-tmp/outputs/planmar_main/large
   /root/autodl-tmp/outputs/planmar_main/huge

Each model directory contains:

   train_ref_mixed/
   eval_main/
   logs/

Merging Results
---------------

Per-server collection can be run before merging:

On Server-Base:

   python scripts_server/collect_main_results.py --root /root/autodl-tmp/outputs/planmar_main --models base --skip_missing_models --output_prefix base

On Server-Large:

   python scripts_server/collect_main_results.py --root /root/autodl-tmp/outputs/planmar_main --models large --skip_missing_models --output_prefix large

On Server-Huge:

   python scripts_server/collect_main_results.py --root /root/autodl-tmp/outputs/planmar_main --models huge --skip_missing_models --output_prefix huge

After all servers finish, copy or rsync the base, large, and huge folders into
the same OUTPUT_ROOT on one machine. Then run the final merged collection:

   python scripts_server/collect_main_results.py --root /root/autodl-tmp/outputs/planmar_main --models all

Collector outputs:

   main_results_summary.csv
   main_results_summary.json
   pareto_data.csv

With --output_prefix base, large, or huge, the collector writes prefixed files
such as base_main_results_summary.csv and base_pareto_data.csv.

Notes
-----

Edit paths in scripts_server/common_env.sh before launching if your AutoDL
layout differs.

For ImageNet-1K checkpoints, class_num remains 1000. The eval scripts use
num_images=50000 so it is divisible by class_num.
