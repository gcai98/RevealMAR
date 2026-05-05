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
- `scripts_server/eval_large_parallel_6points.sh`
- `scripts_server/run_base_train_then_parallel_eval.sh`
- `scripts_server/run_large_train_then_parallel_eval.sh`
- `scripts_server/collect_main_results.py`

The verified commands set `EVAL_POLICIES=baseline,planner`, `EVAL_NUM_ITERS=128,256`, `USE_TORCHRUN=1`, and `NPROC_PER_NODE` to the available GPU count. For the Base workflow, the recommended setting is `EVAL_BSZ=128`. For the Large workflow, the recommended conservative setting is `EVAL_BSZ=64`.

## Excluded Generated Artifacts

Generated outputs, checkpoints, logs, samples, local archives, cache directories, and server temporary files are excluded from the final supplementary package.

## External Assets Not Included

- ImageNet-1K image-folder dataset.
- Pretrained MAR checkpoints.
- KL-16 VAE checkpoint.
- Any metric backend weights not already distributed with the environment.

## Command Templates

Set paths before running either workflow:

```bash
export CODE_DIR=/path/to/planmar-s
export DATA_ROOT=/path/to/imagenet
export PRETRAIN_ROOT=/path/to/pretrained_models
export OUTPUT_ROOT=/path/to/output/planmar_main
export CONDA_SH=/path/to/miniconda3/etc/profile.d/conda.sh