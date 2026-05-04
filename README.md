# PlanMAR-S / RevealMAR

PlanMAR-S extends masked autoregressive image generation with a lightweight planner for reveal-set selection. During training, the planner receives a one-step reference-policy-conditioned mixed-reveal pseudo utility target. During inference, the learned planner ranks currently masked latent tokens and selects the next reveal set under a score-derived soft budget. The original MAR sampling path remains available as the baseline.

This release focuses on the formal server-side main-result reproduction workflow.

## Installation

```bash
conda env create -f environment.yaml
conda activate revealmar
```

Install a PyTorch/CUDA build appropriate for the target multi-GPU server.

## Required External Assets

External assets are not included:

- ImageNet-1K in image-folder format.
- Pretrained MAR checkpoints, for example `/path/to/pretrained_models/mar/mar_base/checkpoint-last.pth`.
- KL-16 VAE checkpoint, for example `/path/to/pretrained_models/vae/kl16.ckpt`.
- ImageNet FID/Inception statistics. The tracked `fid_stats/adm_in256_stats.npz` file is kept if permitted by the release policy.

Expected ImageNet layout:

```text
/path/to/imagenet/
  train/
  val/
```

## Configure Server Paths

Edit `scripts_server/common_env.sh` or override these variables:

```bash
export CODE_DIR=/path/to/revealmar
export DATA_ROOT=/path/to/imagenet
export PRETRAIN_ROOT=/path/to/pretrained_models
export OUTPUT_ROOT=/path/to/output/planmar_main
export CONDA_SH=/path/to/miniconda3/etc/profile.d/conda.sh
```

## Verified Formal Workflow

The official verified run path is:

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

Notes:

- This is the verified formal server workflow.
- `EVAL_NUM_ITERS=128,256` restricts evaluation to the main-paper four points.
- The evaluation script name contains `6points` for historical compatibility; `EVAL_NUM_ITERS` controls which decoding steps are actually evaluated.
- The formal minimal reproduction uses only Base baseline/planner evaluation at 128 and 256 decoding steps.
- Large can be run analogously using `train_large_ref_mixed.sh` and the same environment, but it is optional and compute expensive.
- Use `NPROC_PER_NODE=7` instead of `8` on a seven-GPU server.

The active server scripts for the release are:

- `scripts_server/common_env.sh`
- `scripts_server/check_server_ready.sh`
- `scripts_server/train_base_ref_mixed.sh`
- `scripts_server/train_large_ref_mixed.sh`
- `scripts_server/eval_base_parallel_6points.sh`
- `scripts_server/run_base_train_then_parallel_eval.sh`
- `scripts_server/collect_main_results.py`

## Collect Results

```bash
python scripts_server/collect_main_results.py \
  --root /path/to/output/planmar_main \
  --models base \
  --skip_missing_models \
  --output_prefix base_mainpaper \
  --eval_name eval_mainpaper \
  --policies baseline,planner
```

Expected outputs:

- `<OUTPUT_ROOT>/base_mainpaper_main_results_summary.csv`
- `<OUTPUT_ROOT>/base_mainpaper_main_results_summary.json`
- `<OUTPUT_ROOT>/base_mainpaper_pareto_data.csv`
- Per-run `eval.log`, `run_args.txt`, and `config.json` files under the selected output root.
