@echo off
setlocal enabledelayedexpansion

REM ============================================================
REM P0 same-parameter control runner.
REM Tests whether planner gains come from useful supervision, not just params.
REM ============================================================

set "CODE_DIR=C:\caogang\RevealMAR\RevealMAR"
set "DATA_PATH=C:\caogang\MAR\tiny-imagenet-200-1percent"
set "VAE_PATH=C:\caogang\RevealMAR\pretrained_models\vae\kl16.ckpt"
set "SOURCE_CHECKPOINT=C:\caogang\RevealMAR\RevealMAR\checkpoint-last.pth"
set "OUTPUT_ROOT=C:\caogang\RevealMAR\RevealMAR\p0_runs\same_parameter_control"
set "MODEL_ONLY_INIT=%OUTPUT_ROOT%\model_only_init"
set "FINETUNE_INIT=%MODEL_ONLY_INIT%"

set "MODEL=revealmar_base"
set "BATCH_SIZE=4"
set "EPOCHS=1"
set "NUM_WORKERS=0"
set "DIFFLOSS_D=6"
set "DIFFLOSS_W=1024"
set "NUM_SAMPLING_STEPS=100"
set "DIFFUSION_BATCH_MUL=1"
set "PLANNER_LOSS_WEIGHT=0.3"
set "CANDIDATE_POOL_SIZE=8"
set "CONTROL_RANDOM_STD=1.0"
set "CONTROL_RANDOM_SEED=123"
set "REF_TARGET_HORIZON=1"
set "REF_TARGET_LOCAL_RADIUS=1"
set "REF_TARGET_MIX_ALPHA=0.5"
set "REF_TARGET_LOSS=feature_mse"
set "REF_TARGET_MAX_CANDIDATES=8"

set "NUM_IMAGES=1000"
set "CLASS_NUM=1000"
set "EVAL_BSZ=32"
set "NUM_ITER=64"
set "CFG=1.0"
set "CFG_SCHEDULE=linear"
set "TEMPERATURE=1.0"
set "LOG_PLANNER_SAMPLING_STEPS=8"

echo ============================================================
echo START SAME-PARAMETER CONTROL: %DATE% %TIME%
echo CODE_DIR=%CODE_DIR%
echo DATA_PATH=%DATA_PATH%
echo SOURCE_CHECKPOINT=%SOURCE_CHECKPOINT%
echo OUTPUT_ROOT=%OUTPUT_ROOT%
echo ============================================================

if not exist "%OUTPUT_ROOT%" mkdir "%OUTPUT_ROOT%"
if not exist "%MODEL_ONLY_INIT%" mkdir "%MODEL_ONLY_INIT%"
cd /d "%CODE_DIR%"

echo Creating model-only init checkpoint for same-parameter controls.
python tools\make_model_only_checkpoint.py --src "%SOURCE_CHECKPOINT%" --dst_dir "%MODEL_ONLY_INIT%"
if ERRORLEVEL 1 (
    echo [WARN] Failed to create model-only init checkpoint. Continuing, but training may fail.
)

call :RUN_TARGET control_zero
call :RUN_TARGET control_random
call :RUN_TARGET control_shuffle
call :RUN_TARGET ref_mixed_reveal

echo ============================================================
echo Collector command
echo ============================================================
echo python collect_same_parameter_control_results.py --root "%OUTPUT_ROOT%"
echo ============================================================
echo END SAME-PARAMETER CONTROL: %DATE% %TIME%
echo ============================================================
exit /b 0

:RUN_TARGET
set "TARGET_TYPE=%~1"
set "TRAIN_DIR=%OUTPUT_ROOT%\train_%TARGET_TYPE%"
if not exist "%TRAIN_DIR%" mkdir "%TRAIN_DIR%"

set "TRAIN_CMD=python main_revealmar.py --model %MODEL% --img_size 256 --vae_path "%VAE_PATH%" --vae_embed_dim 16 --vae_stride 16 --patch_size 1 --data_path "%DATA_PATH%" --resume "%FINETUNE_INIT%" --output_dir "%TRAIN_DIR%" --batch_size %BATCH_SIZE% --epochs %EPOCHS% --save_last_freq 1 --eval_freq 999999 --num_workers %NUM_WORKERS% --diffloss_d %DIFFLOSS_D% --diffloss_w %DIFFLOSS_W% --num_sampling_steps %NUM_SAMPLING_STEPS% --diffusion_batch_mul %DIFFUSION_BATCH_MUL% --planner_loss_weight %PLANNER_LOSS_WEIGHT% --candidate_pool_size %CANDIDATE_POOL_SIZE% --pseudo_target_type %TARGET_TYPE% --control_random_std %CONTROL_RANDOM_STD% --control_random_seed %CONTROL_RANDOM_SEED% --log_revealmar_losses --log_revealmar_loss_freq 20 --dist_url env://"
if /I "%TARGET_TYPE%"=="ref_mixed_reveal" (
    set "TRAIN_CMD=!TRAIN_CMD! --ref_target_horizon %REF_TARGET_HORIZON% --ref_target_local_radius %REF_TARGET_LOCAL_RADIUS% --ref_target_mix_alpha %REF_TARGET_MIX_ALPHA% --ref_target_loss %REF_TARGET_LOSS% --ref_target_max_candidates %REF_TARGET_MAX_CANDIDATES% --log_ref_target_debug --log_ref_target_freq 20"
)
echo !TRAIN_CMD! > "%TRAIN_DIR%\run_args.txt"
call :WRITE_CONFIG "%TRAIN_DIR%" "train" "%TARGET_TYPE%" "" "%TARGET_TYPE%" ""

echo ============================================================
echo TRAIN %TARGET_TYPE%
echo ============================================================
powershell -NoProfile -ExecutionPolicy Bypass -Command "& { !TRAIN_CMD! 2>&1 | Tee-Object -FilePath '%TRAIN_DIR%\train.log' }"
if ERRORLEVEL 1 (
    echo [WARN] Training failed for %TARGET_TYPE%. Continuing to eval attempts.
)

call :RUN_EVAL "%TARGET_TYPE%" baseline hard
call :RUN_EVAL "%TARGET_TYPE%" planner hard
call :RUN_EVAL "%TARGET_TYPE%" planner soft
exit /b 0

:RUN_EVAL
set "TARGET_TYPE=%~1"
set "SAMPLING_POLICY=%~2"
set "BUDGET_MODE=%~3"
if /I "%SAMPLING_POLICY%"=="baseline" (
    set "EVAL_VARIANT=baseline"
    set "EVAL_PSEUDO=none"
) else (
    set "EVAL_VARIANT=planner_%BUDGET_MODE%"
    set "EVAL_PSEUDO=%TARGET_TYPE%"
)
set "TRAIN_DIR=%OUTPUT_ROOT%\train_%TARGET_TYPE%"
set "EVAL_DIR=%OUTPUT_ROOT%\eval_%TARGET_TYPE%_%EVAL_VARIANT%"
if not exist "%EVAL_DIR%" mkdir "%EVAL_DIR%"

set "EVAL_CMD=python main_revealmar.py --evaluate --model %MODEL% --img_size 256 --vae_path "%VAE_PATH%" --vae_embed_dim 16 --vae_stride 16 --patch_size 1 --data_path "%DATA_PATH%" --resume "%TRAIN_DIR%" --class_num %CLASS_NUM% --diffloss_d %DIFFLOSS_D% --diffloss_w %DIFFLOSS_W% --diffusion_batch_mul %DIFFUSION_BATCH_MUL% --num_images %NUM_IMAGES% --eval_bsz %EVAL_BSZ% --num_iter %NUM_ITER% --num_sampling_steps %NUM_SAMPLING_STEPS% --cfg %CFG% --cfg_schedule %CFG_SCHEDULE% --temperature %TEMPERATURE% --num_workers %NUM_WORKERS% --output_dir "%EVAL_DIR%" --sampling_policy %SAMPLING_POLICY% --pseudo_target_type %EVAL_PSEUDO% --budget_mode %BUDGET_MODE% --planner_loss_weight %PLANNER_LOSS_WEIGHT% --candidate_pool_size %CANDIDATE_POOL_SIZE% --control_random_std %CONTROL_RANDOM_STD% --control_random_seed %CONTROL_RANDOM_SEED% --dist_url env://"
if /I "%SAMPLING_POLICY%"=="planner" (
    set "EVAL_CMD=!EVAL_CMD! --log_planner_sampling_debug --log_planner_sampling_steps %LOG_PLANNER_SAMPLING_STEPS%"
)
echo !EVAL_CMD! > "%EVAL_DIR%\run_args.txt"
call :WRITE_CONFIG "%EVAL_DIR%" "eval" "%TARGET_TYPE%" "%EVAL_VARIANT%" "%EVAL_PSEUDO%" "%BUDGET_MODE%"

echo ------------------------------------------------------------
echo EVAL %TARGET_TYPE% %EVAL_VARIANT%
echo ------------------------------------------------------------
powershell -NoProfile -ExecutionPolicy Bypass -Command "& { !EVAL_CMD! 2>&1 | Tee-Object -FilePath '%EVAL_DIR%\eval.log' }"
if ERRORLEVEL 1 (
    echo [WARN] Eval failed for %TARGET_TYPE% %EVAL_VARIANT%. Continuing.
)
exit /b 0

:WRITE_CONFIG
set "CFG_DIR=%~1"
set "CFG_PHASE=%~2"
set "CFG_TARGET=%~3"
set "CFG_EVAL_VARIANT=%~4"
set "CFG_PSEUDO=%~5"
set "CFG_BUDGET=%~6"
powershell -NoProfile -ExecutionPolicy Bypass -Command "$cfg = [ordered]@{ phase='%CFG_PHASE%'; target_type='%CFG_TARGET%'; eval_variant='%CFG_EVAL_VARIANT%'; pseudo_target_type='%CFG_PSEUDO%'; budget_mode='%CFG_BUDGET%'; data_path='%DATA_PATH%'; source_checkpoint='%SOURCE_CHECKPOINT%'; finetune_init='%FINETUNE_INIT%'; output_dir='%CFG_DIR%'; model='%MODEL%'; batch_size=%BATCH_SIZE%; epochs=%EPOCHS%; planner_loss_weight=%PLANNER_LOSS_WEIGHT%; candidate_pool_size=%CANDIDATE_POOL_SIZE%; control_random_std=%CONTROL_RANDOM_STD%; control_random_seed=%CONTROL_RANDOM_SEED%; num_images=%NUM_IMAGES%; eval_bsz=%EVAL_BSZ%; num_iter=%NUM_ITER%; cfg=%CFG%; ref_target_horizon=%REF_TARGET_HORIZON%; ref_target_local_radius=%REF_TARGET_LOCAL_RADIUS%; ref_target_mix_alpha=%REF_TARGET_MIX_ALPHA%; ref_target_loss='%REF_TARGET_LOSS%'; ref_target_max_candidates=%REF_TARGET_MAX_CANDIDATES%; timestamp=(Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ') }; $cfg | ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 '%CFG_DIR%\config.json'"
exit /b 0
