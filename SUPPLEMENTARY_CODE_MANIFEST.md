# Supplementary Code Manifest

This supplementary package focuses on reproducing the formal PlanMAR-S main-result workflow.

## Included Core Code

- `main_revealmar.py`
- `engine_mar.py`
- `main_mar.py`
- `main_cache.py`
- `models/`
- `diffusion/`
- `util/`
- `fid_stats/adm_in256_stats.npz` if permitted by release policy

## Active Formal Reproduction Scripts

Only these scripts are active in the release workflow:

- `scripts_server/common_env.sh`
- `scripts_server/check_server_ready.sh`
- `scripts_server/train_base_ref_mixed.sh`
- `scripts_server/train_large_ref_mixed.sh`
- `scripts_server/eval_base_parallel_6points.sh`
- `scripts_server/run_base_train_then_parallel_eval.sh`
- `scripts_server/collect_main_results.py`

The verified command sets `EVAL_POLICIES=baseline,planner`, `EVAL_NUM_ITERS=128,256`, `EVAL_BSZ=128`, `USE_TORCHRUN=1`, and `NPROC_PER_NODE` to the available GPU count.

## Excluded Generated Artifacts

Generated outputs, checkpoints, logs, samples, local archives, cache directories, and server temporary files are excluded from the final supplementary package.

## External Assets Not Included

- ImageNet-1K image-folder dataset.
- Pretrained MAR checkpoints.
- KL-16 VAE checkpoint.
- Any metric backend weights not already distributed with the environment.

## Command Template

```bash
export CODE_DIR=/path/to/revealmar
export DATA_ROOT=/path/to/imagenet
export PRETRAIN_ROOT=/path/to/pretrained_models
export OUTPUT_ROOT=/path/to/output/planmar_main
export CONDA_SH=/path/to/miniconda3/etc/profile.d/conda.sh

AUTO_SHUTDOWN=1 \
USE_TORCHRUN=1 NPROC_PER_NODE=8 \
TRAIN_BSZ=96 \
TRAIN_EPOCHS=7 WARMUP_EPOCHS=1 TRAIN_MAX_STEPS=-1 TRAIN_RUN_NAME=train_ref_mixed_7ep \
MIXED_POLICY_RATIO=0.0 \
EVAL_RUN_NAME=eval_mainpaper EVAL_NUM_IMAGES=50000 EVAL_BSZ=128 EVAL_POLICIES=baseline,planner EVAL_NUM_ITERS=128,256 EVAL_GPU_IDS=0,1,2,3 \
nohup bash scripts_server/run_base_train_then_parallel_eval.sh \
> /path/to/output/base_mainpaper_7ep_8gpu.log 2>&1 &
```

Collect results:

```bash
python scripts_server/collect_main_results.py \
  --root /path/to/output/planmar_main \
  --models base \
  --skip_missing_models \
  --output_prefix base_mainpaper \
  --eval_name eval_mainpaper \
  --policies baseline,planner
```
