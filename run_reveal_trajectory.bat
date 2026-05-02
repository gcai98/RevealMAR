@echo off
setlocal enabledelayedexpansion

set "CODE_DIR=C:\caogang\RevealMAR\RevealMAR"
set "DATA_PATH=C:\caogang\MAR\tiny-imagenet-200-1percent"
set "VAE_PATH=C:\caogang\RevealMAR\pretrained_models\vae\kl16.ckpt"
set "RESUME=C:\caogang\RevealMAR\RevealMAR"
set "OUTPUT_ROOT=C:\caogang\RevealMAR\RevealMAR\p0_runs\reveal_trajectory"

if not exist "%OUTPUT_ROOT%" mkdir "%OUTPUT_ROOT%"
cd /d "%CODE_DIR%"

call :RUN_TRAJ planner_hard planner hard
call :RUN_TRAJ planner_soft planner soft
call :RUN_TRAJ random_hard random hard
call :RUN_TRAJ confidence_hard confidence hard
call :RUN_TRAJ entropy_hard entropy hard

echo Collector command:
echo python collect_reveal_trajectory_results.py --trajectory_root "%OUTPUT_ROOT%" --intervention_root "C:\caogang\RevealMAR\RevealMAR\p0_runs\early_step_intervention"
exit /b 0

:RUN_TRAJ
set "RUN_NAME=%~1"
set "SAMPLING_POLICY=%~2"
set "BUDGET_MODE=%~3"
set "RUN_DIR=%OUTPUT_ROOT%\%RUN_NAME%"
if not exist "%RUN_DIR%" mkdir "%RUN_DIR%"
set "CMD=python eval_reveal_trajectory.py --data_path "%DATA_PATH%" --vae_path "%VAE_PATH%" --resume "%RESUME%" --output_dir "%RUN_DIR%" --model revealmar_base --num_images 16 --class_num 16 --eval_bsz 4 --num_iter 64 --num_sampling_steps 100 --cfg 1.0 --cfg_schedule linear --temperature 1.0 --num_workers 0 --sampling_policy %SAMPLING_POLICY% --budget_mode %BUDGET_MODE% --pseudo_target_type mixed_reveal --candidate_pool_size 8 --save_heatmaps --save_json"
echo !CMD! > "%RUN_DIR%\run_args.txt"
powershell -NoProfile -ExecutionPolicy Bypass -Command "$cfg = [ordered]@{ run_name='%RUN_NAME%'; sampling_policy='%SAMPLING_POLICY%'; budget_mode='%BUDGET_MODE%'; data_path='%DATA_PATH%'; resume='%RESUME%'; output_dir='%RUN_DIR%'; num_images=16; class_num=16; eval_bsz=4; num_iter=64; pseudo_target_type='mixed_reveal'; candidate_pool_size=8; timestamp=(Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ') }; $cfg | ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 '%RUN_DIR%\config.json'"
echo Running %RUN_NAME%
powershell -NoProfile -ExecutionPolicy Bypass -Command "& { !CMD! 2>&1 | Tee-Object -FilePath '%RUN_DIR%\eval.log' }"
if ERRORLEVEL 1 echo [WARN] %RUN_NAME% failed.
exit /b 0
