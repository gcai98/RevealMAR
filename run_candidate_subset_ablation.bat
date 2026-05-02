@echo off
setlocal enabledelayedexpansion

set "CODE_DIR=C:\caogang\RevealMAR\RevealMAR"
set "DATA_PATH=C:\caogang\MAR\tiny-imagenet-200-1percent"
set "VAE_PATH=C:\caogang\RevealMAR\pretrained_models\vae\kl16.ckpt"
set "RESUME=C:\caogang\RevealMAR\RevealMAR"
set "OUTPUT_DIR=C:\caogang\RevealMAR\RevealMAR\p0_runs\candidate_subset_ablation"

if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"
cd /d "%CODE_DIR%"

set "CMD=python eval_candidate_subset_stability.py --data_path "%DATA_PATH%" --vae_path "%VAE_PATH%" --resume "%RESUME%" --output_dir "%OUTPUT_DIR%" --model revealmar_base --num_eval_images 16 --eval_bsz 4 --num_states_per_image 1 --num_iter 64 --state_step_min 0 --state_step_max 16 --candidate_pool_size 8 --candidate_modes topk,random,uncertainty,spatial,mixed --local_radius 1 --rollout_horizon 1 --pseudo_target_type mixed_reveal --cfg 1.0 --num_workers 0"

echo !CMD! > "%OUTPUT_DIR%\run_args.txt"
powershell -NoProfile -ExecutionPolicy Bypass -Command "$cfg = [ordered]@{ data_path='%DATA_PATH%'; vae_path='%VAE_PATH%'; resume='%RESUME%'; output_dir='%OUTPUT_DIR%'; model='revealmar_base'; num_eval_images=16; eval_bsz=4; num_states_per_image=1; num_iter=64; state_step_min=0; state_step_max=16; candidate_pool_size=8; candidate_modes='topk,random,uncertainty,spatial,mixed'; local_radius=1; rollout_horizon=1; pseudo_target_type='mixed_reveal'; cfg=1.0; num_workers=0; timestamp=(Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ') }; $cfg | ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 '%OUTPUT_DIR%\config.json'"

powershell -NoProfile -ExecutionPolicy Bypass -Command "& { !CMD! 2>&1 | Tee-Object -FilePath '%OUTPUT_DIR%\eval.log' }"
if ERRORLEVEL 1 echo [WARN] Candidate subset ablation failed. Collector may report failed/no_metrics.

echo Collector command:
echo python collect_candidate_subset_ablation_results.py --root "%OUTPUT_DIR%"
exit /b 0
