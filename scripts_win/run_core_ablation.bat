@echo off
setlocal enabledelayedexpansion

REM ============================================================
REM P0-4 Core Ablation Table runner for Windows CMD.
REM Edit the variables below for your machine. No training is run.
REM ============================================================

set CODE_DIR=C:\caogang\RevealMAR\RevealMAR
set DATA_PATH=C:\caogang\MAR\tiny-imagenet-200-10percent
set VAE_PATH=C:\caogang\RevealMAR\pretrained_models\vae\kl16.ckpt
set BASE_RESUME=C:\caogang\RevealMAR\pretrained_models\mar\mar_base
set PLANMAR_RESUME=C:\caogang\RevealMAR\RevealMAR\output_mixed40_w03_e10
set OUTPUT_ROOT=C:\caogang\RevealMAR\RevealMAR\p0_runs
set NUM_IMAGES=1000
set EVAL_BSZ=32
set NUM_ITER=64
set NUM_WORKERS=0

set PLANNER_HEAD_ONLY_RESUME=
set GT_RESUME=
set PRED_RESUME=
set MIXED_RESUME=

set MODEL=revealmar_base
set DIFFLOSS_D=6
set DIFFLOSS_W=1024
set NUM_SAMPLING_STEPS=100
set CFG=1.0
set PLANNER_LOSS_WEIGHT=0.3
set CANDIDATE_POOL_SIZE=8
set TABLE_NAME=core
set TABLE_DIR=%OUTPUT_ROOT%\core_ablation

if "%PLANNER_HEAD_ONLY_RESUME%"=="" (
    echo [WARN] PLANNER_HEAD_ONLY_RESUME is not set; falling back to PLANMAR_RESUME.
    set PLANNER_HEAD_ONLY_RESUME=%PLANMAR_RESUME%
)
if "%GT_RESUME%"=="" (
    echo [WARN] GT_RESUME is not set; falling back to PLANMAR_RESUME.
    set GT_RESUME=%PLANMAR_RESUME%
)
if "%PRED_RESUME%"=="" (
    echo [WARN] PRED_RESUME is not set; falling back to PLANMAR_RESUME.
    set PRED_RESUME=%PLANMAR_RESUME%
)
if "%MIXED_RESUME%"=="" (
    echo [WARN] MIXED_RESUME is not set; falling back to PLANMAR_RESUME.
    set MIXED_RESUME=%PLANMAR_RESUME%
)

echo ============================================================
echo START P0-4 Core Ablation: %DATE% %TIME%
echo CODE_DIR=%CODE_DIR%
echo DATA_PATH=%DATA_PATH%
echo OUTPUT_ROOT=%OUTPUT_ROOT%
echo NUM_IMAGES=%NUM_IMAGES% EVAL_BSZ=%EVAL_BSZ% NUM_ITER=%NUM_ITER%
echo ============================================================

if not exist "%TABLE_DIR%" mkdir "%TABLE_DIR%"
cd /d "%CODE_DIR%"

call :RUN_VARIANT "MAR Baseline" "MAR Baseline" "baseline" "fixed" "%BASE_RESUME%" "baseline" "none" "hard"
call :RUN_VARIANT "Planner Head only" "PlanMAR-S" "planner-head-only" "fixed" "%PLANNER_HEAD_ONLY_RESUME%" "planner" "none" "hard"
call :RUN_VARIANT "GT future-aware target" "PlanMAR-S" "gt-reveal" "fixed" "%GT_RESUME%" "planner" "gt_reveal" "hard"
call :RUN_VARIANT "Pred-reveal" "PlanMAR-S" "pred-reveal" "fixed" "%PRED_RESUME%" "planner" "pred_reveal" "hard"
call :RUN_VARIANT "Mixed-reveal" "PlanMAR-S" "mixed-reveal" "fixed" "%MIXED_RESUME%" "planner" "mixed_reveal" "hard"
call :RUN_VARIANT "Mixed-reveal + score-derived budget" "PlanMAR-S" "mixed-reveal" "score-derived" "%MIXED_RESUME%" "planner" "mixed_reveal" "soft"

echo ============================================================
echo END P0-4 Core Ablation: %DATE% %TIME%
echo Results root: %TABLE_DIR%
echo ============================================================
exit /b 0

:RUN_VARIANT
set VARIANT=%~1
set METHOD=%~2
set REVEAL_STRATEGY=%~3
set BUDGET_RULE=%~4
set RESUME=%~5
set SAMPLING_POLICY=%~6
set PSEUDO_TARGET_TYPE=%~7
set BUDGET_MODE=%~8

if "%RESUME%"=="" (
    echo [WARN] Skipping %VARIANT% because RESUME is empty.
    exit /b 0
)

set RUN_NAME=%VARIANT%
set RUN_NAME=%RUN_NAME: =_%
set RUN_NAME=%RUN_NAME:-=_%
set RUN_NAME=%RUN_NAME:/=_%
set RUN_NAME=%RUN_NAME:(=_%
set RUN_NAME=%RUN_NAME:)=_%
set RUN_NAME=%RUN_NAME:+=plus%
set RUN_DIR=%TABLE_DIR%\%RUN_NAME%_n%NUM_IMAGES%_iter%NUM_ITER%
if not exist "%RUN_DIR%" mkdir "%RUN_DIR%"

set CMD=python main_revealmar.py --evaluate --model %MODEL% --data_path "%DATA_PATH%" --vae_path "%VAE_PATH%" --resume "%RESUME%" --diffloss_d %DIFFLOSS_D% --diffloss_w %DIFFLOSS_W% --num_images %NUM_IMAGES% --eval_bsz %EVAL_BSZ% --num_iter %NUM_ITER% --num_sampling_steps %NUM_SAMPLING_STEPS% --cfg %CFG% --num_workers %NUM_WORKERS% --output_dir "%RUN_DIR%" --sampling_policy %SAMPLING_POLICY% --pseudo_target_type %PSEUDO_TARGET_TYPE% --budget_mode %BUDGET_MODE% --planner_loss_weight %PLANNER_LOSS_WEIGHT% --candidate_pool_size %CANDIDATE_POOL_SIZE%

echo %CMD% > "%RUN_DIR%\run_args.txt"
call :WRITE_CONFIG "%RUN_DIR%" "%VARIANT%" "%METHOD%" "%REVEAL_STRATEGY%" "%BUDGET_RULE%" "%RESUME%" "%SAMPLING_POLICY%" "%PSEUDO_TARGET_TYPE%" "%BUDGET_MODE%"

echo ------------------------------------------------------------
echo RUN %VARIANT%
echo START %DATE% %TIME%
echo RUN_DIR=%RUN_DIR%
echo ------------------------------------------------------------

powershell -NoProfile -ExecutionPolicy Bypass -Command "& { %CMD% 2>&1 | Tee-Object -FilePath '%RUN_DIR%\eval.log' }"
if ERRORLEVEL 1 (
    echo [WARN] Run failed: %VARIANT%
)
echo END %VARIANT%: %DATE% %TIME%
exit /b 0

:WRITE_CONFIG
set CFG_RUN_DIR=%~1
set CFG_VARIANT=%~2
set CFG_METHOD=%~3
set CFG_REVEAL_STRATEGY=%~4
set CFG_BUDGET_RULE=%~5
set CFG_RESUME=%~6
set CFG_SAMPLING_POLICY=%~7
set CFG_PSEUDO_TARGET_TYPE=%~8
set CFG_BUDGET_MODE=%~9

powershell -NoProfile -ExecutionPolicy Bypass -Command "$cfg = [ordered]@{ table='%TABLE_NAME%'; variant='%CFG_VARIANT%'; method='%CFG_METHOD%'; reveal_strategy='%CFG_REVEAL_STRATEGY%'; budget_rule='%CFG_BUDGET_RULE%'; resume='%CFG_RESUME%'; data_path='%DATA_PATH%'; num_images=%NUM_IMAGES%; eval_bsz=%EVAL_BSZ%; num_iter=%NUM_ITER%; sampling_policy='%CFG_SAMPLING_POLICY%'; pseudo_target_type='%CFG_PSEUDO_TARGET_TYPE%'; budget_mode='%CFG_BUDGET_MODE%'; timestamp=(Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ'); extra_params=[ordered]@{ model='%MODEL%'; diffloss_d=%DIFFLOSS_D%; diffloss_w=%DIFFLOSS_W%; num_sampling_steps='%NUM_SAMPLING_STEPS%'; cfg=%CFG% } }; $cfg | ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 '%CFG_RUN_DIR%\config.json'"
exit /b 0
