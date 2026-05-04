Server Main Results Runbook
===========================

This is the only active runbook for the formal reproduction workflow.

Verified Base Command
---------------------

```bash
AUTO_SHUTDOWN=1 \
USE_TORCHRUN=1 NPROC_PER_NODE=8 \
TRAIN_BSZ=96 \
TRAIN_EPOCHS=7 WARMUP_EPOCHS=1 TRAIN_MAX_STEPS=-1 TRAIN_RUN_NAME=train_ref_mixed_7ep \
MIXED_POLICY_RATIO=0.0 \
EVAL_RUN_NAME=eval_mainpaper EVAL_NUM_IMAGES=50000 EVAL_BSZ=128 EVAL_POLICIES=baseline,planner EVAL_NUM_ITERS=128,256 EVAL_GPU_IDS=0,1,2,3 \
nohup bash scripts_server/run_base_train_then_parallel_eval.sh \
> /path/to/output/base_mainpaper_7ep_8gpu.log 2>&1 &
```

Notes
-----

- This is the verified formal server workflow.
- `EVAL_NUM_ITERS=128,256` restricts evaluation to the main-paper four points.
- The script name still contains `6points` for historical compatibility; environment variables control the active points.
- The minimal formal reproduction excludes 64-step, confidence/entropy, Huge, ablations, and mechanism experiments.
- Large can be run analogously if needed, but is optional and compute expensive.
- Use `NPROC_PER_NODE=7` on a seven-GPU server.

Active Scripts
--------------

- `scripts_server/common_env.sh`
- `scripts_server/check_server_ready.sh`
- `scripts_server/train_base_ref_mixed.sh`
- `scripts_server/train_large_ref_mixed.sh`
- `scripts_server/eval_base_parallel_6points.sh`
- `scripts_server/run_base_train_then_parallel_eval.sh`
- `scripts_server/collect_main_results.py`

Collect Results
---------------

```bash
python scripts_server/collect_main_results.py \
  --root /path/to/output/planmar_main \
  --models base \
  --skip_missing_models \
  --output_prefix base_mainpaper \
  --eval_name eval_mainpaper \
  --policies baseline,planner
```

Archived Material
-----------------

All platform-specific, optional, mechanism, ablation, visualization, debug, probe, and preflight scripts are archived under `scripts_archive/` and are not part of the active formal workflow.
