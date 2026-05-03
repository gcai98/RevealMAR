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

The top-level scripts run readiness checks, train ref_mixed_reveal for
TRAIN_EPOCHS epochs, then evaluate the policies listed in EVAL_POLICIES at 64,
128, and 256 decoding iterations. Formal main results default to baseline and
planner only.

Training Run Names
------------------

Server training scripts read these optional environment variables:

   TRAIN_EPOCHS
   WARMUP_EPOCHS
   TRAIN_RUN_NAME
   TRAIN_MAX_STEPS
   USE_TORCHRUN
   NPROC_PER_NODE
   MIXED_POLICY_RATIO

Defaults are safe for a 1-epoch validation:

   TRAIN_EPOCHS=1
   WARMUP_EPOCHS=1
   TRAIN_RUN_NAME=train_ref_mixed
   TRAIN_MAX_STEPS=-1
   USE_TORCHRUN=0
   NPROC_PER_NODE=1
   MIXED_POLICY_RATIO=0.0

Use a separate run name for probes so they do not overwrite final outputs.

1-Epoch Validation Probe
------------------------

Run train_xxx_ref_mixed.sh directly for the 1-epoch validation. Do not use
run_xxx_train_then_eval.sh for the probe, because the top-level script starts
full evaluation after training.

Base probe:

   TRAIN_EPOCHS=1 WARMUP_EPOCHS=1 TRAIN_RUN_NAME=train_ref_mixed_ep1_probe \
   nohup bash scripts_server/train_base_ref_mixed.sh \
   > /root/autodl-tmp/outputs/planmar_main/base_ep1_probe.nohup.log 2>&1 &

Large probe:

   TRAIN_EPOCHS=1 WARMUP_EPOCHS=1 TRAIN_RUN_NAME=train_ref_mixed_ep1_probe \
   nohup bash scripts_server/train_large_ref_mixed.sh \
   > /root/autodl-tmp/outputs/planmar_main/large_ep1_probe.nohup.log 2>&1 &

Huge probe:

   TRAIN_EPOCHS=1 WARMUP_EPOCHS=1 TRAIN_RUN_NAME=train_ref_mixed_ep1_probe \
   nohup bash scripts_server/train_huge_ref_mixed.sh \
   > /root/autodl-tmp/outputs/planmar_main/huge_ep1_probe.nohup.log 2>&1 &

Probe checkpoints are saved to:

   ${OUTPUT_ROOT}/{model}/train_ref_mixed_ep1_probe/

1-Epoch Full-Chain Validation With 1000 Images
----------------------------------------------

This validates the full train+eval chain before expensive final training. It
trains for 1 epoch, then evaluates the policies listed in EVAL_POLICIES at
num_iter=64,128,256 with 1000 generated images per run.

The chain order is:

   training -> evaluation -> auto collection

Huge training batch size defaults to 64 through TRAIN_BSZ. Evaluation batch
size remains controlled separately by EVAL_BSZ and defaults to 32 in the ep1
eval1000 chain to reduce OOM risk.

Override huge training batch size if needed:

   TRAIN_BSZ=32 bash scripts_server/run_huge_ep1_eval1000_chain.sh
   TRAIN_BSZ=64 bash scripts_server/run_huge_ep1_eval1000_chain.sh

The ep1 eval1000 chain defaults to TRAIN_MAX_STEPS=200 to keep validation
bounded. Single-card debug example:

   TRAIN_MAX_STEPS=200 bash scripts_server/run_base_ep1_eval1000_chain.sh

Base:

   nohup bash scripts_server/run_base_ep1_eval1000_chain.sh \
   > /root/autodl-tmp/outputs/planmar_main/base_ep1_eval1000_chain.nohup.log 2>&1 &

Large:

   nohup bash scripts_server/run_large_ep1_eval1000_chain.sh \
   > /root/autodl-tmp/outputs/planmar_main/large_ep1_eval1000_chain.nohup.log 2>&1 &

Huge:

   nohup bash scripts_server/run_huge_ep1_eval1000_chain.sh \
   > /root/autodl-tmp/outputs/planmar_main/huge_ep1_eval1000_chain.nohup.log 2>&1 &

The ep1 eval1000 chain scripts automatically collect per-model results by
default after evaluation finishes. Disable this with:

   AUTO_COLLECT=0 bash scripts_server/run_huge_ep1_eval1000_chain.sh

Collector for base:

   python scripts_server/collect_main_results.py \
     --root /root/autodl-tmp/outputs/planmar_main \
     --models base \
     --skip_missing_models \
     --output_prefix base_ep1_1000 \
     --eval_name eval_main_ep1_1000

Recommended baseline/planner-only main-result collector for base:

   python scripts_server/collect_main_results.py \
     --root /root/autodl-tmp/outputs/planmar_main \
     --models base \
     --skip_missing_models \
     --output_prefix base_ep1_baseline_planner \
     --eval_name eval_main_ep1_1000 \
     --policies baseline,planner

Collector for large:

   python scripts_server/collect_main_results.py \
     --root /root/autodl-tmp/outputs/planmar_main \
     --models large \
     --skip_missing_models \
     --output_prefix large_ep1_1000 \
     --eval_name eval_main_ep1_1000

Collector for huge:

   python scripts_server/collect_main_results.py \
     --root /root/autodl-tmp/outputs/planmar_main \
     --models huge \
     --skip_missing_models \
     --output_prefix huge_ep1_1000 \
     --eval_name eval_main_ep1_1000

1-epoch full-chain outputs are saved to:

   ${OUTPUT_ROOT}/{model}/train_ref_mixed_ep1_chain/
   ${OUTPUT_ROOT}/{model}/eval_main_ep1_1000/

Final formal outputs are saved to:

   ${OUTPUT_ROOT}/{model}/train_ref_mixed/
   ${OUTPUT_ROOT}/{model}/eval_main/

Do not mix ep1 chain outputs with final outputs.

Final Train-Then-Eval
---------------------

After preflight and the 1-epoch probe pass, launch the first formal
train-then-eval with a separate final run name.

Use 10/1 first. The 400/100 setting was the original MAR-style long training
setting and is not recommended for the first PlanMAR-S fine-tuning run.
Consider 20/2 only if 10 epochs shows promising trends but is insufficient.

Formal main results default to EVAL_POLICIES=baseline,planner. Confidence and
entropy are not part of the default main result because they are slow and
currently require cfg=1.0.

Mixed-policy exposure is optional. The default is MIXED_POLICY_RATIO=0.0. To
run an optional mixed-policy ablation, set a ratio such as:

   MIXED_POLICY_RATIO=0.25 bash scripts_server/run_base_train_then_eval.sh

Base final:

   AUTO_SHUTDOWN=1 EVAL_POLICIES=baseline,planner TRAIN_EPOCHS=10 WARMUP_EPOCHS=1 TRAIN_MAX_STEPS=-1 TRAIN_RUN_NAME=train_ref_mixed \
   nohup bash scripts_server/run_base_train_then_eval.sh \
   > /root/autodl-tmp/outputs/planmar_main/base_full.nohup.log 2>&1 &

Large final:

   AUTO_SHUTDOWN=1 EVAL_POLICIES=baseline,planner TRAIN_EPOCHS=10 WARMUP_EPOCHS=1 TRAIN_MAX_STEPS=-1 TRAIN_RUN_NAME=train_ref_mixed \
   nohup bash scripts_server/run_large_train_then_eval.sh \
   > /root/autodl-tmp/outputs/planmar_main/large_full.nohup.log 2>&1 &

Huge final:

   AUTO_SHUTDOWN=1 EVAL_POLICIES=baseline,planner TRAIN_EPOCHS=10 WARMUP_EPOCHS=1 TRAIN_MAX_STEPS=-1 TRAIN_RUN_NAME=train_ref_mixed \
   nohup bash scripts_server/run_huge_train_then_eval.sh \
   > /root/autodl-tmp/outputs/planmar_main/huge_full.nohup.log 2>&1 &

8-card final example:

   USE_TORCHRUN=1 NPROC_PER_NODE=8 EVAL_POLICIES=baseline,planner TRAIN_EPOCHS=10 WARMUP_EPOCHS=1 TRAIN_MAX_STEPS=-1 bash scripts_server/run_huge_train_then_eval.sh

Final checkpoints are saved to:

   ${OUTPUT_ROOT}/{model}/train_ref_mixed/

Do not mix probe and final checkpoints. When evaluating a probe, pass
TRAIN_RUN_NAME=train_ref_mixed_ep1_probe to the eval script; when evaluating
final results, use TRAIN_RUN_NAME=train_ref_mixed.

Scheme A Formal Run: Base + Large, Baseline/Planner Only
--------------------------------------------------------

Scheme A is the current formal main-result plan:

   models: base, large
   policies: baseline, planner
   num_iter: 64,128,256
   num_images: 50000

Do not include huge, confidence, entropy, oracle mismatch, same-parameter, or
mixed-policy full-FID runs in the first Scheme A formal run.

The parallel eval scripts launch one job per policy/step pair:

   baseline_iter64
   baseline_iter128
   baseline_iter256
   planner_iter64
   planner_iter128
   planner_iter256

By default these jobs use EVAL_GPU_IDS=0,1,2,3,4,5. Baseline eval uses the
official pretrained MAR checkpoint, while planner eval uses the trained
PlanMAR-S checkpoint at ${OUTPUT_ROOT}/{model}/${TRAIN_RUN_NAME}.

Base Scheme A:

   AUTO_SHUTDOWN=1 \
   USE_TORCHRUN=1 NPROC_PER_NODE=8 \
   TRAIN_EPOCHS=10 WARMUP_EPOCHS=1 TRAIN_MAX_STEPS=-1 TRAIN_RUN_NAME=train_ref_mixed \
   MIXED_POLICY_RATIO=0.0 \
   EVAL_RUN_NAME=eval_main EVAL_NUM_IMAGES=50000 EVAL_BSZ=256 EVAL_POLICIES=baseline,planner EVAL_NUM_ITERS=64,128,256 EVAL_GPU_IDS=0,1,2,3,4,5 \
   nohup bash scripts_server/run_base_train_then_parallel_eval.sh \
   > /root/autodl-tmp/outputs/planmar_main/base_schemeA.nohup.log 2>&1 &

Large Scheme A:

   AUTO_SHUTDOWN=1 \
   USE_TORCHRUN=1 NPROC_PER_NODE=8 \
   TRAIN_EPOCHS=10 WARMUP_EPOCHS=1 TRAIN_MAX_STEPS=-1 TRAIN_RUN_NAME=train_ref_mixed \
   MIXED_POLICY_RATIO=0.0 \
   EVAL_RUN_NAME=eval_main EVAL_NUM_IMAGES=50000 EVAL_BSZ=256 EVAL_POLICIES=baseline,planner EVAL_NUM_ITERS=64,128,256 EVAL_GPU_IDS=0,1,2,3,4,5 \
   nohup bash scripts_server/run_large_train_then_parallel_eval.sh \
   > /root/autodl-tmp/outputs/planmar_main/large_schemeA.nohup.log 2>&1 &

Eval-only after training already finished:

   bash scripts_server/eval_base_parallel_6points.sh
   bash scripts_server/eval_large_parallel_6points.sh

Manuscript-Priority Run: Minimal Main-Paper Results
---------------------------------------------------

This is the recommended first formal run when time is limited. It produces the
minimal main-paper result package:

   Table 1: Main Results at 256 steps
   Table 2: Quality-Efficiency Trade-off at 128 and 256 steps
   Budget sanity from planner logs

It runs:

   models: base, large
   policies: baseline, planner
   num_iter: 128,256
   num_images: 50000

It skips 64-step eval, confidence/entropy, huge, oracle mismatch,
same-parameter controls, mixed-policy full-FID, and other full-FID ablations.

The main-paper eval scripts use EVAL_BSZ=128 by default because 7x5090
preflight showed VAE decode OOM risk at EVAL_BSZ=256.

7x5090 base main-paper run:

   AUTO_SHUTDOWN=1 \
   USE_TORCHRUN=1 NPROC_PER_NODE=7 \
   TRAIN_BSZ=96 \
   TRAIN_EPOCHS=10 WARMUP_EPOCHS=1 TRAIN_MAX_STEPS=-1 TRAIN_RUN_NAME=train_ref_mixed \
   MIXED_POLICY_RATIO=0.0 \
   EVAL_RUN_NAME=eval_mainpaper EVAL_NUM_IMAGES=50000 EVAL_BSZ=128 EVAL_POLICIES=baseline,planner EVAL_NUM_ITERS=128,256 EVAL_GPU_IDS=0,1,2,3 \
   nohup bash scripts_server/run_base_train_then_mainpaper_eval.sh \
   > /root/autodl-tmp/outputs/planmar_main/base_mainpaper_7gpu.log 2>&1 &

7x5090 large main-paper run:

   AUTO_SHUTDOWN=1 \
   USE_TORCHRUN=1 NPROC_PER_NODE=7 \
   TRAIN_BSZ=96 \
   TRAIN_EPOCHS=10 WARMUP_EPOCHS=1 TRAIN_MAX_STEPS=-1 TRAIN_RUN_NAME=train_ref_mixed \
   MIXED_POLICY_RATIO=0.0 \
   EVAL_RUN_NAME=eval_mainpaper EVAL_NUM_IMAGES=50000 EVAL_BSZ=128 EVAL_POLICIES=baseline,planner EVAL_NUM_ITERS=128,256 EVAL_GPU_IDS=0,1,2,3 \
   nohup bash scripts_server/run_large_train_then_mainpaper_eval.sh \
   > /root/autodl-tmp/outputs/planmar_main/large_mainpaper_7gpu.log 2>&1 &

7x5090 base main-paper preflight:

   USE_TORCHRUN=1 NPROC_PER_NODE=7 \
   TRAIN_BSZ=96 \
   TRAIN_EPOCHS=1 WARMUP_EPOCHS=1 TRAIN_MAX_STEPS=200 TRAIN_RUN_NAME=train_ref_mixed_preflight \
   MIXED_POLICY_RATIO=0.0 \
   EVAL_RUN_NAME=eval_mainpaper_preflight EVAL_NUM_IMAGES=1000 EVAL_BSZ=128 EVAL_POLICIES=baseline,planner EVAL_NUM_ITERS=128,256 EVAL_GPU_IDS=0,1,2,3 \
   nohup bash scripts_server/run_base_train_then_mainpaper_eval.sh \
   > /root/autodl-tmp/outputs/planmar_main/base_mainpaper_preflight_7gpu.log 2>&1 &

Surrogate validity and reveal trajectory are separate small-scale mechanism
experiments. Do not run them inside the 50k main train/eval wrapper.

   Surrogate Validity: run eval_surrogate_validity.py or the corresponding script on Base only.
   Reveal Trajectory: run eval_reveal_trajectory.py or the corresponding script on Base only.

Scheme A on 7-GPU RTX 5090 Server
---------------------------------

If an 8-GPU RTX 5090 server is unavailable, Scheme A can run safely on a
7-GPU server.

Training should use:

   USE_TORCHRUN=1 NPROC_PER_NODE=7

Parallel eval should still use six jobs:

   EVAL_GPU_IDS=0,1,2,3,4,5

GPU 6 is intentionally left idle/reserved during eval. This is expected because
Scheme A has six eval points: baseline/planner x 64/128/256.

Because NPROC_PER_NODE changes from 8 to 7, the effective global training batch
size changes if TRAIN_BSZ is unchanged. For the time-constrained Scheme-A run,
keep TRAIN_BSZ unchanged for stability unless OOM or training instability
occurs. Report the actual GPU count and effective batch size in the
appendix/runtime diagnostics.

7-GPU base preflight:

   USE_TORCHRUN=1 NPROC_PER_NODE=7 \
   TRAIN_EPOCHS=1 WARMUP_EPOCHS=1 TRAIN_MAX_STEPS=200 TRAIN_RUN_NAME=train_ref_mixed_preflight \
   MIXED_POLICY_RATIO=0.0 \
   EVAL_RUN_NAME=eval_preflight EVAL_NUM_IMAGES=1000 EVAL_BSZ=64 EVAL_POLICIES=baseline,planner EVAL_NUM_ITERS=64,128,256 EVAL_GPU_IDS=0,1,2,3,4,5 \
   nohup bash scripts_server/run_base_train_then_parallel_eval.sh \
   > /root/autodl-tmp/outputs/planmar_main/base_schemeA_7gpu_preflight.nohup.log 2>&1 &

7-GPU formal base:

   AUTO_SHUTDOWN=1 \
   USE_TORCHRUN=1 NPROC_PER_NODE=7 \
   TRAIN_EPOCHS=10 WARMUP_EPOCHS=1 TRAIN_MAX_STEPS=-1 TRAIN_RUN_NAME=train_ref_mixed \
   MIXED_POLICY_RATIO=0.0 \
   EVAL_RUN_NAME=eval_main EVAL_NUM_IMAGES=50000 EVAL_BSZ=128 EVAL_POLICIES=baseline,planner EVAL_NUM_ITERS=64,128,256 EVAL_GPU_IDS=0,1,2,3,4,5 \
   nohup bash scripts_server/run_base_train_then_parallel_eval.sh \
   > /root/autodl-tmp/outputs/planmar_main/base_schemeA_7gpu.nohup.log 2>&1 &

7-GPU formal large:

   AUTO_SHUTDOWN=1 \
   USE_TORCHRUN=1 NPROC_PER_NODE=7 \
   TRAIN_EPOCHS=10 WARMUP_EPOCHS=1 TRAIN_MAX_STEPS=-1 TRAIN_RUN_NAME=train_ref_mixed \
   MIXED_POLICY_RATIO=0.0 \
   EVAL_RUN_NAME=eval_main EVAL_NUM_IMAGES=50000 EVAL_BSZ=128 EVAL_POLICIES=baseline,planner EVAL_NUM_ITERS=64,128,256 EVAL_GPU_IDS=0,1,2,3,4,5 \
   nohup bash scripts_server/run_large_train_then_parallel_eval.sh \
   > /root/autodl-tmp/outputs/planmar_main/large_schemeA_7gpu.nohup.log 2>&1 &

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
