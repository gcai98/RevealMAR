@echo off
setlocal enabledelayedexpansion

set "CODE_DIR=C:\caogang\RevealMAR\RevealMAR"
set "DATA_PATH=C:\caogang\MAR\tiny-imagenet-200-1percent"
set "VAE_PATH=C:\caogang\RevealMAR\pretrained_models\vae\kl16.ckpt"
set "RESUME=C:\caogang\RevealMAR\RevealMAR"
set "OUTPUT_ROOT=C:\caogang\RevealMAR\RevealMAR\p0_runs\early_step_intervention"

if not exist "%OUTPUT_ROOT%" mkdir "%OUTPUT_ROOT%"
cd /d "%CODE_DIR%"

call :RUN_INTERVENTION planner_no_intervention none 0
call :RUN_INTERVENTION planner_intervene_random8 random 8
call :RUN_INTERVENTION planner_intervene_confidence8 confidence 8
call :RUN_INTERVENTION planner_intervene_entropy8 entropy 8

echo Collector command:
echo python collect_reveal_trajectory_results.py --trajectory_root "C:\caogang\RevealMAR\RevealMAR\p0_runs\reveal_trajectory" --intervention_root "%OUTPUT_ROOT%"
exit /b 0

:RUN_INTERVENTION
set "RUN_NAME=%~1"
set "INTERVENTION_POLICY=%~2"
set "INTERVENTION_STEPS=%~3"
set "RUN_DIR=%OUTPUT_ROOT%\%RUN_NAME%"
if not exist "%RUN_DIR%" mkdir "%RUN_DIR%"
set "CMD=python eval_early_step_intervention.py --data_path "%DATA_PATH%" --vae_path "%VAE_PATH%" --resume "%RESUME%" --output_dir "%RUN_DIR%" --model revealmar_base --num_images 1000 --class_num 1000 --eval_bsz 32 --num_iter 64 --num_sampling_steps 100 --cfg 1.0 --cfg_schedule linear --temperature 1.0 --num_workers 0 --base_policy planner --intervention_policy %INTERVENTION_POLICY% --intervention_steps %INTERVENTION_STEPS% --budget_mode hard --pseudo_target_type mixed_reveal --candidate_pool_size 8 --log_planner_sampling_steps 8"
echo !CMD! > "%RUN_DIR%\run_args.txt"
powershell -NoProfile -ExecutionPolicy Bypass -Command "$cfg = [ordered]@{ run_name='%RUN_NAME%'; intervention_policy='%INTERVENTION_POLICY%'; intervention_steps=%INTERVENTION_STEPS%; sampling_policy='planner'; budget_mode='hard'; data_path='%DATA_PATH%'; resume='%RESUME%'; output_dir='%RUN_DIR%'; num_images=1000; class_num=1000; eval_bsz=32; num_iter=64; cfg=1.0; pseudo_target_type='mixed_reveal'; candidate_pool_size=8; timestamp=(Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ') }; $cfg | ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 '%RUN_DIR%\config.json'"
echo Running %RUN_NAME%
powershell -NoProfile -ExecutionPolicy Bypass -Command "& { !CMD! 2>&1 | Tee-Object -FilePath '%RUN_DIR%\eval.log' }"
if ERRORLEVEL 1 echo [WARN] %RUN_NAME% failed.
exit /b 0
