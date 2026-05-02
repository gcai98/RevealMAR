@echo off
setlocal enabledelayedexpansion

REM ============================================================
REM P0 budget calibration runner.
REM Smoke-scale score-derived budget sensitivity for planner-soft.
REM ============================================================

set "CODE_DIR=C:\caogang\RevealMAR\RevealMAR"
set "DATA_PATH=C:\caogang\MAR\tiny-imagenet-200-1percent"
set "VAE_PATH=C:\caogang\RevealMAR\pretrained_models\vae\kl16.ckpt"
set "RESUME=C:\caogang\RevealMAR\RevealMAR"
set "OUTPUT_ROOT=C:\caogang\RevealMAR\RevealMAR\p0_runs\budget_calibration"

set "MODEL=revealmar_base"
set "NUM_IMAGES=1000"
set "CLASS_NUM=1000"
set "EVAL_BSZ=32"
set "NUM_ITER=64"
set "NUM_WORKERS=0"
set "CFG=1.0"
set "CFG_SCHEDULE=linear"
set "TEMPERATURE=1.0"
set "DIFFLOSS_D=6"
set "DIFFLOSS_W=1024"
set "NUM_SAMPLING_STEPS=100"
set "DIFFUSION_BATCH_MUL=1"
set "SAMPLING_POLICY=planner"
set "PSEUDO_TARGET_TYPE=mixed_reveal"
set "BUDGET_MODE=soft"
set "BUDGET_MIN=1"
set "BUDGET_MAX=-1"
set "LOG_PLANNER_SAMPLING_STEPS=8"

echo ============================================================
echo START BUDGET CALIBRATION: %DATE% %TIME%
echo CODE_DIR=%CODE_DIR%
echo DATA_PATH=%DATA_PATH%
echo RESUME=%RESUME%
echo OUTPUT_ROOT=%OUTPUT_ROOT%
echo ============================================================

if not exist "%OUTPUT_ROOT%" mkdir "%OUTPUT_ROOT%"
cd /d "%CODE_DIR%"

call :RUN_VARIANT tau1_scale1_beta0 1.0 1.0 0.0
call :RUN_VARIANT tau0p5_scale1_beta0 0.5 1.0 0.0
call :RUN_VARIANT tau0p25_scale1_beta0 0.25 1.0 0.0
call :RUN_VARIANT tau1_scale5_beta0 1.0 5.0 0.0
call :RUN_VARIANT tau1_scale10_beta0 1.0 10.0 0.0
call :RUN_VARIANT tau0p5_scale5_beta0p8 0.5 5.0 0.8

echo ============================================================
echo Collector command
echo ============================================================
echo python collect_budget_calibration_results.py --root "%OUTPUT_ROOT%"
echo ============================================================
echo END BUDGET CALIBRATION: %DATE% %TIME%
echo ============================================================
exit /b 0

:RUN_VARIANT
set "RUN_NAME=%~1"
set "BUDGET_TEMPERATURE=%~2"
set "BUDGET_SCORE_SCALE=%~3"
set "BUDGET_EMA_BETA=%~4"
set "RUN_DIR=%OUTPUT_ROOT%\%RUN_NAME%"
if not exist "%RUN_DIR%" mkdir "%RUN_DIR%"

set "CMD=python main_revealmar.py --evaluate --model %MODEL% --img_size 256 --vae_path "%VAE_PATH%" --vae_embed_dim 16 --vae_stride 16 --patch_size 1 --data_path "%DATA_PATH%" --resume "%RESUME%" --class_num %CLASS_NUM% --diffloss_d %DIFFLOSS_D% --diffloss_w %DIFFLOSS_W% --diffusion_batch_mul %DIFFUSION_BATCH_MUL% --num_images %NUM_IMAGES% --eval_bsz %EVAL_BSZ% --num_iter %NUM_ITER% --num_sampling_steps %NUM_SAMPLING_STEPS% --cfg %CFG% --cfg_schedule %CFG_SCHEDULE% --temperature %TEMPERATURE% --num_workers %NUM_WORKERS% --output_dir "%RUN_DIR%" --sampling_policy %SAMPLING_POLICY% --pseudo_target_type %PSEUDO_TARGET_TYPE% --budget_mode %BUDGET_MODE% --budget_temperature %BUDGET_TEMPERATURE% --budget_score_scale %BUDGET_SCORE_SCALE% --budget_min %BUDGET_MIN% --budget_max %BUDGET_MAX% --budget_ema_beta %BUDGET_EMA_BETA% --budget_calibration_debug --planner_loss_weight 0.3 --candidate_pool_size 8 --log_planner_sampling_debug --log_planner_sampling_steps %LOG_PLANNER_SAMPLING_STEPS% --dist_url env://"

echo !CMD! > "%RUN_DIR%\run_args.txt"
call :WRITE_CONFIG "%RUN_DIR%" "%RUN_NAME%" "%BUDGET_TEMPERATURE%" "%BUDGET_SCORE_SCALE%" "%BUDGET_EMA_BETA%"

echo ------------------------------------------------------------
echo RUN %RUN_NAME% tau=%BUDGET_TEMPERATURE% scale=%BUDGET_SCORE_SCALE% beta=%BUDGET_EMA_BETA%
echo ------------------------------------------------------------
powershell -NoProfile -ExecutionPolicy Bypass -Command "& { !CMD! 2>&1 | Tee-Object -FilePath '%RUN_DIR%\eval.log' }"
if ERRORLEVEL 1 (
    echo [WARN] Budget calibration run failed for %RUN_NAME%. Continuing.
)
exit /b 0

:WRITE_CONFIG
set "CFG_DIR=%~1"
set "CFG_RUN_NAME=%~2"
set "CFG_TAU=%~3"
set "CFG_SCALE=%~4"
set "CFG_BETA=%~5"
powershell -NoProfile -ExecutionPolicy Bypass -Command "$cfg = [ordered]@{ run_name='%CFG_RUN_NAME%'; data_path='%DATA_PATH%'; resume='%RESUME%'; output_dir='%CFG_DIR%'; model='%MODEL%'; sampling_policy='%SAMPLING_POLICY%'; pseudo_target_type='%PSEUDO_TARGET_TYPE%'; budget_mode='%BUDGET_MODE%'; budget_temperature=[double]'%CFG_TAU%'; budget_score_scale=[double]'%CFG_SCALE%'; budget_ema_beta=[double]'%CFG_BETA%'; budget_min=%BUDGET_MIN%; budget_max=%BUDGET_MAX%; num_images=%NUM_IMAGES%; class_num=%CLASS_NUM%; eval_bsz=%EVAL_BSZ%; num_iter=%NUM_ITER%; num_sampling_steps=%NUM_SAMPLING_STEPS%; cfg=%CFG%; cfg_schedule='%CFG_SCHEDULE%'; temperature=%TEMPERATURE%; timestamp=(Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ') }; $cfg | ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 '%CFG_DIR%\config.json'"
exit /b 0
