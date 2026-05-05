# PlanMAR-S

PlanMAR-S extends masked autoregressive image generation with a lightweight planner for reveal-set selection. During training, the planner receives a one-step reference-policy-conditioned mixed-reveal pseudo utility target. During inference, the learned planner ranks currently masked latent tokens and selects the next reveal set under a score-derived soft budget. The original MAR sampling path remains available as the baseline.

![PlanMAR-S overview](figs/planmar_overview.jpg)

This release focuses on the formal server-side main-result reproduction workflow.

## Installation

```bash
conda env create -f environment.yaml
conda activate <environment-name>
```

Install a PyTorch/CUDA build appropriate for the target multi-GPU server.

## Required External Assets

External assets are not included in this repository.

### ImageNet-1K

The formal experiments use ImageNet-1K in image-folder format. The expected layout is:

```text
/path/to/imagenet/
  train/
  val/
```

ImageNet access link:

- Official ImageNet download page: https://www.image-net.org/download.php

Please follow the corresponding dataset license and access requirements.

### Pretrained MAR Checkpoints

Pretrained MAR checkpoints should be placed under a directory such as:

```text
/path/to/pretrained_models/
  mar/
    mar_base/
      checkpoint-last.pth
    mar_large/
      checkpoint-last.pth
```

The original MAR repository provides instructions and pretrained model references:

```text
https://github.com/LTH14/mar
```

### KL-16 VAE Checkpoint

The KL-16 VAE checkpoint should be placed under a directory such as:

```text
/path/to/pretrained_models/
  vae/
    kl16.ckpt
```

A KL-16 checkpoint can be downloaded from:

```text
https://www.dropbox.com/scl/fi/hhmuvaiacrarfg28qxhwz/kl16.ckpt?rlkey=l44xipsezc8atcffdp4q7mwmh&dl=0
```

If downloading from the command line, users may need to change `dl=0` to `dl=1` depending on the download tool. Please follow the license and usage terms associated with the checkpoint.

### FID / Inception Statistics

The tracked file

```text
fid_stats/adm_in256_stats.npz
```

is used for ImageNet 256×256 FID evaluation if permitted by the release policy. Otherwise, prepare the corresponding ImageNet 256×256 statistics file and place it under `fid_stats/`.

## Configure Server Paths

Edit `scripts_server/common_env.sh` or override these variables:

```bash
export CODE_DIR=/path/to/planmar-s
export DATA_ROOT=/path/to/imagenet
export PRETRAIN_ROOT=/path/to/pretrained_models
export OUTPUT_ROOT=/path/to/output/planmar_main
export CONDA_SH=/path/to/miniconda3/etc/profile.d/conda.sh
```

## Verified Formal Workflow

### Base Workflow

The verified Base run path is:

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

### Large Workflow

The Large workflow follows the same training/evaluation structure. Because the Large backbone is more memory-intensive, a more conservative batch size is recommended by default:

```bash
AUTO_SHUTDOWN=1 \
USE_TORCHRUN=1 NPROC_PER_NODE=8 \
TRAIN_BSZ=64 \
TRAIN_EPOCHS=7 WARMUP_EPOCHS=1 TRAIN_MAX_STEPS=-1 TRAIN_RUN_NAME=train_ref_mixed_7ep \
MIXED_POLICY_RATIO=0.0 \
EVAL_RUN_NAME=eval_mainpaper EVAL_NUM_IMAGES=50000 EVAL_BSZ=64 EVAL_POLICIES=baseline,planner EVAL_NUM_ITERS=128,256 EVAL_GPU_IDS=0,1,2,3 \
nohup bash scripts_server/run_large_train_then_parallel_eval.sh \
> /path/to/output/large_mainpaper_7ep_8gpu.log 2>&1 &
```

Notes:

- These are the verified formal server workflows.
- `EVAL_NUM_ITERS=128,256` restricts evaluation to the main-paper four points.
- The evaluation script names contain `6points` for historical compatibility; `EVAL_NUM_ITERS` controls which decoding steps are actually evaluated.
- The formal minimal reproduction uses baseline/planner evaluation at 128 and 256 decoding steps.
- Use `NPROC_PER_NODE=7` instead of `8` on a seven-GPU server.
- For the Large workflow, increase `TRAIN_BSZ` or `EVAL_BSZ` only after confirming memory safety on the target hardware.

The active server scripts for the release are:

- `scripts_server/common_env.sh`
- `scripts_server/check_server_ready.sh`
- `scripts_server/train_base_ref_mixed.sh`
- `scripts_server/train_large_ref_mixed.sh`
- `scripts_server/eval_base_parallel_6points.sh`
- `scripts_server/eval_large_parallel_6points.sh`
- `scripts_server/run_base_train_then_parallel_eval.sh`
- `scripts_server/run_large_train_then_parallel_eval.sh`
- `scripts_server/collect_main_results.py`

## Collect Results

For Base results:

```bash
python scripts_server/collect_main_results.py \
  --root /path/to/output/planmar_main \
  --models base \
  --skip_missing_models \
  --output_prefix base_mainpaper \
  --eval_name eval_mainpaper \
  --policies baseline,planner
```

For Large results:

```bash
python scripts_server/collect_main_results.py \
  --root /path/to/output/planmar_main \
  --models large \
  --skip_missing_models \
  --output_prefix large_mainpaper \
  --eval_name eval_mainpaper \
  --policies baseline,planner
```

For collecting both Base and Large results together:

```bash
python scripts_server/collect_main_results.py \
  --root /path/to/output/planmar_main \
  --models base,large \
  --skip_missing_models \
  --output_prefix mainpaper \
  --eval_name eval_mainpaper \
  --policies baseline,planner
```

Expected outputs include:

- `<OUTPUT_ROOT>/<prefix>_main_results_summary.csv`
- `<OUTPUT_ROOT>/<prefix>_main_results_summary.json`
- `<OUTPUT_ROOT>/<prefix>_pareto_data.csv`
- Per-run `eval.log`, `run_args.txt`, and `config.json` files under the selected output root.

## Scope of This Release

This release focuses on reproducing the formal main-result workflow. Optional mechanism experiments, debug/probe scripts, platform-specific runners, and historical ablation scripts are not part of this minimal release package.

## Acknowledgements

This implementation builds on the official MAR codebase:

- `LTH14/mar`: https://github.com/LTH14/mar

We thank the MAR authors for releasing their PyTorch implementation, pretrained checkpoints, and evaluation framework.

## License

This codebase includes modifications built on top of the official MAR implementation, which is released under the MIT License. This repository follows the same open-source license unless otherwise noted. Please see the `LICENSE` file for details.

External assets are subject to their own licenses and terms of use, including ImageNet-1K, pretrained MAR checkpoints, the KL-16 VAE checkpoint, and metric backend weights. Users are responsible for obtaining these assets and complying with their respective terms.