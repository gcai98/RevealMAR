# Cleanup Report

This report was created before repository cleanup for the anonymous NeurIPS 2026
supplementary code package.

## Core Implementation Kept

- `main_revealmar.py`, `engine_mar.py`
- `models/revealmar.py`, `models/mar.py`, `models/diffloss.py`, `models/vae.py`
- `diffusion/`
- `util/`
- `fid_stats/adm_in256_stats.npz`
- Formal result collection is kept in `scripts_server/collect_main_results.py`.
- Small-scale mechanism evaluators and older collectors were archived in Step 2.

## Official Training and Evaluation Scripts Kept

- `scripts_server/common_env.sh`
- `scripts_server/train_base_ref_mixed.sh`
- `scripts_server/train_large_ref_mixed.sh`
- `scripts_server/run_base_train_then_parallel_eval.sh`
- `scripts_server/eval_base_parallel_6points.sh`
- `scripts_server/collect_main_results.py`

These scripts are retained because they cover the current verified workflow:
train PlanMAR-S with `ref_mixed_reveal`, evaluate baseline/planner, and collect
main results.

## Files Removed as Obsolete, Debug, or Temporary

The root-level `run_*.bat` and `run_*.sh` files had already been removed before
this cleanup pass. Local cache directories such as `.pyc_tmp/`,
`.pycache_tmp/`, `.pycache_prefix/`, and `__pycache__/` should not be included
in a release package and are ignored by `.gitignore`.

## Files Moved to Archive

- `scripts/` -> `scripts_archive/scripts/`

The archived scripts were older convenience wrappers superseded by the verified
`scripts_server/` formal workflow. They are not part of the recommended
anonymous release workflow.

## Personal, Machine-Specific, or Non-Anonymous Text Found

- User-specific Windows paths in older collectors, P0 README files, and
  `scripts_win/`.
- Machine-specific Linux server paths in server/Linux scripts and old
  documentation.
- Temporary or debug-oriented text in `README.md`, `revealmar.md`, and P0
  runbook files.
- Chinese/debug comments in `main_revealmar.py` and `models/difficulty.py`.

These are cleaned or replaced with neutral placeholders such as:

- `/path/to/imagenet`
- `/path/to/pretrained_models`
- `/path/to/output`
- `anonymous/revealmar`

## Files Excluded From Anonymous Supplementary Zip

Do not include:

- `.git/`
- Python cache directories and compiled files
- local checkpoints: `*.pth`, `*.pt`, `*.ckpt`, `*.safetensors`
- generated samples, logs, `nohup.out`, TensorBoard or wandb outputs
- local output directories such as `outputs/`, `checkpoints/`, `generated/`
- archived/debug scripts under `scripts_archive/`, unless explicitly needed
- private data, ImageNet, pretrained MAR checkpoints, and VAE checkpoints

External assets are documented in `SUPPLEMENTARY_CODE_MANIFEST.md`.

## Step 2: Final Script-Surface Minimization

Active scripts kept for formal reproduction:

- `scripts_server/common_env.sh`
- `scripts_server/check_server_ready.sh`
- `scripts_server/train_base_ref_mixed.sh`
- `scripts_server/train_large_ref_mixed.sh`
- `scripts_server/eval_base_parallel_6points.sh`
- `scripts_server/run_base_train_then_parallel_eval.sh`
- `scripts_server/collect_main_results.py`

Folders removed from the active workflow:

- `scripts_win/`
- `scripts_linux/`
- legacy `scripts/`

Folders moved to `scripts_archive/`:

- `scripts_archive/scripts/`
- `scripts_archive/scripts_win/`
- `scripts_archive/scripts_linux/`
- `scripts_archive/scripts_server_optional/`
- `scripts_archive/mechanism_experiments/`
- `scripts_archive/p0_readmes/`
- `scripts_archive/tools/`

Optional or unverified scripts archived:

- mainpaper four-point wrappers
- ep1 and preflight chain wrappers
- huge and large convenience eval wrappers
- local Windows and Linux smoke/debug runners
- surrogate validity, reveal trajectory, candidate subset, set-action consistency, early intervention, oracle mismatch, budget calibration, mixed-policy, same-parameter, and P0 result collectors/evaluators
- cleanup and checkpoint-conversion helper tools used by archived smoke workflows

No model, training, sampling, pseudo-target, FID/IS, checkpoint, data, or tracked FID-stat file was modified by this script-surface minimization.
