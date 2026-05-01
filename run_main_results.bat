@echo off
setlocal enabledelayedexpansion

REM ============================================================
REM P0-1 Main Results Table runner for Windows CMD/PowerShell.
REM Evaluation only; no training is run.
REM ============================================================

set "CODE_DIR=C:\caogang\RevealMAR\RevealMAR"
set "DATA_PATH=C:\caogang\MAR\tiny-imagenet-200-1percent"
set "VAE_PATH=C:\caogang\RevealMAR\pretrained_models\vae\kl16.ckpt"
set "TRAINED_RESUME=C:\caogang\RevealMAR\RevealMAR"
set "OUTPUT_ROOT=C:\caogang\RevealMAR\RevealMAR\p0_runs"

set "NUM_IMAGES=1000"
set "CLASS_NUM=1000"
set "EVAL_BSZ=32"
set "NUM_ITER=64"
set "NUM_WORKERS=0"

set "MODEL=revealmar_base"
set "DIFFLOSS_D=6"
set "DIFFLOSS_W=1024"
set "NUM_SAMPLING_STEPS=100"
set "CFG=1.0"
set "PLANNER_LOSS_WEIGHT=0.3"
set "CANDIDATE_POOL_SIZE=8"
set "UNCERTAINTY_MC_SAMPLES=2"
set "UNCERTAINTY_POLICY_TEMPERATURE=1.0"
set "LOG_PLANNER_SAMPLING_STEPS=8"

set "TABLE_NAME=main"
set "TABLE_DIR=%OUTPUT_ROOT%\main_results_trained_ckpt_full_policies"

echo ============================================================
echo START P0-1 Main Results: %DATE% %TIME%
echo CODE_DIR=%CODE_DIR%
echo DATA_PATH=%DATA_PATH%
echo TRAINED_RESUME=%TRAINED_RESUME%
echo OUTPUT_ROOT=%OUTPUT_ROOT%
echo NUM_IMAGES=%NUM_IMAGES% CLASS_NUM=%CLASS_NUM% EVAL_BSZ=%EVAL_BSZ% NUM_ITER=%NUM_ITER%
echo ============================================================

if not exist "%TABLE_DIR%" mkdir "%TABLE_DIR%"
cd /d "%CODE_DIR%"

call :RUN_VARIANT "MAR Baseline" "MAR Baseline" "baseline" "fixed" "%TRAINED_RESUME%" "baseline" "none" "hard" "0"
call :RUN_VARIANT "Random Reveal" "Random Reveal" "random" "fixed" "%TRAINED_RESUME%" "random" "none" "hard" "1"
call :RUN_VARIANT "Confidence Reveal" "Confidence Reveal" "confidence" "fixed" "%TRAINED_RESUME%" "confidence" "none" "hard" "1"
call :RUN_VARIANT "Entropy Reveal" "Entropy Reveal" "entropy" "fixed" "%TRAINED_RESUME%" "entropy" "none" "hard" "1"
call :RUN_VARIANT "PlanMAR-S Fixed Budget" "PlanMAR-S" "planner" "fixed" "%TRAINED_RESUME%" "planner" "mixed_reveal" "hard" "1"
call :RUN_VARIANT "PlanMAR-S Score-derived Budget" "PlanMAR-S" "planner" "score-derived" "%TRAINED_RESUME%" "planner" "mixed_reveal" "soft" "1"

echo ============================================================
echo FID / Inception Score lines
echo ============================================================
powershell -NoProfile -ExecutionPolicy Bypass -Command "Get-ChildItem -LiteralPath '%TABLE_DIR%' -Recurse -Filter 'eval.log' | Select-String -Pattern 'FID:' | ForEach-Object { '{0}: {1}' -f $_.Path, $_.Line }"

echo ============================================================
echo Checkpoint load warnings
echo ============================================================
powershell -NoProfile -ExecutionPolicy Bypass -Command "Get-ChildItem -LiteralPath '%TABLE_DIR%' -Recurse -Filter 'eval.log' | Select-String -Pattern 'Resume missing model keys','EMA missing keys','planner_head' | ForEach-Object { '{0}: {1}' -f $_.Path, $_.Line }"

echo ============================================================
echo END P0-1 Main Results: %DATE% %TIME%
echo Results root: %TABLE_DIR%
echo ============================================================
exit /b 0

:RUN_VARIANT
set "VARIANT=%~1"
set "METHOD=%~2"
set "REVEAL_STRATEGY=%~3"
set "BUDGET_RULE=%~4"
set "RESUME=%~5"
set "SAMPLING_POLICY=%~6"
set "PSEUDO_TARGET_TYPE=%~7"
set "BUDGET_MODE=%~8"
set "ENABLE_DEBUG=%~9"

if "%RESUME%"=="" (
    echo [WARN] Skipping %VARIANT% because RESUME is empty.
    exit /b 0
)

set "RUN_NAME=%VARIANT%"
set "RUN_NAME=%RUN_NAME: =_%"
set "RUN_NAME=%RUN_NAME:-=_%"
set "RUN_NAME=%RUN_NAME:/=_%"
set "RUN_NAME=%RUN_NAME:\=_%"
set "RUN_NAME=%RUN_NAME::=_%"
set "RUN_NAME=%RUN_NAME:(=_%"
set "RUN_NAME=%RUN_NAME:)=_%"
set "RUN_NAME=%RUN_NAME:,=_%"
set "RUN_NAME=%RUN_NAME:.=_%"
set "RUN_DIR=%TABLE_DIR%\%RUN_NAME%_n%NUM_IMAGES%_iter%NUM_ITER%"
if not exist "%RUN_DIR%" mkdir "%RUN_DIR%"

set "CMD=python main_revealmar.py --evaluate --model %MODEL% --img_size 256 --vae_path "%VAE_PATH%" --vae_embed_dim 16 --vae_stride 16 --patch_size 1 --data_path "%DATA_PATH%" --resume "%RESUME%" --class_num %CLASS_NUM% --diffloss_d %DIFFLOSS_D% --diffloss_w %DIFFLOSS_W% --diffusion_batch_mul 1 --num_images %NUM_IMAGES% --eval_bsz %EVAL_BSZ% --num_iter %NUM_ITER% --num_sampling_steps %NUM_SAMPLING_STEPS% --cfg %CFG% --cfg_schedule linear --temperature 1.0 --num_workers %NUM_WORKERS% --output_dir "%RUN_DIR%" --sampling_policy %SAMPLING_POLICY% --pseudo_target_type %PSEUDO_TARGET_TYPE% --budget_mode %BUDGET_MODE% --planner_loss_weight %PLANNER_LOSS_WEIGHT% --candidate_pool_size %CANDIDATE_POOL_SIZE% --dist_url env://"

if /I "%SAMPLING_POLICY%"=="confidence" (
    set "CMD=!CMD! --uncertainty_mc_samples %UNCERTAINTY_MC_SAMPLES% --uncertainty_policy_temperature %UNCERTAINTY_POLICY_TEMPERATURE%"
)
if /I "%SAMPLING_POLICY%"=="entropy" (
    set "CMD=!CMD! --uncertainty_mc_samples %UNCERTAINTY_MC_SAMPLES% --uncertainty_policy_temperature %UNCERTAINTY_POLICY_TEMPERATURE%"
)
if "%ENABLE_DEBUG%"=="1" (
    set "CMD=!CMD! --log_planner_sampling_debug --log_planner_sampling_steps %LOG_PLANNER_SAMPLING_STEPS%"
)

echo !CMD! > "%RUN_DIR%\run_args.txt"
call :WRITE_CONFIG

echo ------------------------------------------------------------
echo RUN %VARIANT%
echo START %DATE% %TIME%
echo RUN_DIR=%RUN_DIR%
echo ------------------------------------------------------------

powershell -NoProfile -ExecutionPolicy Bypass -Command "& { !CMD! 2>&1 | Tee-Object -FilePath '%RUN_DIR%\eval.log' }"
if ERRORLEVEL 1 (
    echo [WARN] Run failed: %VARIANT%
)
echo END %VARIANT%: %DATE% %TIME%
exit /b 0

:WRITE_CONFIG
powershell -NoProfile -ExecutionPolicy Bypass -Command "$cfg = [ordered]@{ table='%TABLE_NAME%'; variant='%VARIANT%'; method='%METHOD%'; reveal_strategy='%REVEAL_STRATEGY%'; budget_rule='%BUDGET_RULE%'; resume='%RESUME%'; data_path='%DATA_PATH%'; num_images=%NUM_IMAGES%; class_num=%CLASS_NUM%; eval_bsz=%EVAL_BSZ%; num_iter=%NUM_ITER%; sampling_policy='%SAMPLING_POLICY%'; pseudo_target_type='%PSEUDO_TARGET_TYPE%'; budget_mode='%BUDGET_MODE%'; debug_enabled=([bool]::Parse(('%ENABLE_DEBUG%' -eq '1').ToString())); timestamp=(Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ'); extra_params=[ordered]@{ model='%MODEL%'; diffloss_d=%DIFFLOSS_D%; diffloss_w=%DIFFLOSS_W%; num_sampling_steps='%NUM_SAMPLING_STEPS%'; cfg=%CFG%; planner_loss_weight=%PLANNER_LOSS_WEIGHT%; candidate_pool_size=%CANDIDATE_POOL_SIZE%; uncertainty_mc_samples=%UNCERTAINTY_MC_SAMPLES%; uncertainty_policy_temperature=%UNCERTAINTY_POLICY_TEMPERATURE% } }; $cfg | ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 '%RUN_DIR%\config.json'"
exit /b 0
