@echo off
setlocal enabledelayedexpansion

REM ============================================================
REM P0-3 Surrogate Validity runner for Windows CMD/PowerShell.
REM Smoke-test scale; no training or image-generation evaluation changes.
REM ============================================================

set "CODE_DIR=C:\caogang\RevealMAR\RevealMAR"
set "DATA_PATH=C:\caogang\MAR\tiny-imagenet-200-1percent"
set "VAE_PATH=C:\caogang\RevealMAR\pretrained_models\vae\kl16.ckpt"
set "RESUME=C:\caogang\RevealMAR\RevealMAR"
set "OUTPUT_DIR=C:\caogang\RevealMAR\RevealMAR\p0_runs\surrogate_validity_trained_ckpt"

set "MODEL=revealmar_base"
set "IMG_SIZE=256"
set "VAE_EMBED_DIM=16"
set "VAE_STRIDE=16"
set "PATCH_SIZE=1"
set "CLASS_NUM=1000"
set "DIFFLOSS_D=6"
set "DIFFLOSS_W=1024"
set "NUM_SAMPLING_STEPS=100"
set "DIFFUSION_BATCH_MUL=1"

REM Smoke-test settings. Increase these for paper-scale runs.
set "NUM_EVAL_IMAGES=16"
set "EVAL_BSZ=4"
set "NUM_WORKERS=0"
set "NUM_STATES_PER_IMAGE=1"
set "ROLLOUT_HORIZON=1"
set "TOPK_EVAL=3"
set "NUM_ITER=64"
set "STATE_STEP_MIN=0"
set "STATE_STEP_MAX=8"
set "LOCAL_RADIUS=1"
set "CANDIDATE_POOL_SIZE=8"
set "CANDIDATE_SELECTION_MODE=mixed"
set "UNCERTAINTY_MC_SAMPLES=2"
set "UNCERTAINTY_TEMPERATURE=1.0"

echo ============================================================
echo START P0-3 Surrogate Validity: %DATE% %TIME%
echo CODE_DIR=%CODE_DIR%
echo DATA_PATH=%DATA_PATH%
echo RESUME=%RESUME%
echo OUTPUT_DIR=%OUTPUT_DIR%
echo NUM_EVAL_IMAGES=%NUM_EVAL_IMAGES% EVAL_BSZ=%EVAL_BSZ% NUM_ITER=%NUM_ITER%
echo ============================================================

if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"
cd /d "%CODE_DIR%"

set "CMD=python eval_surrogate_validity.py --data_path "%DATA_PATH%" --vae_path "%VAE_PATH%" --resume "%RESUME%" --output_dir "%OUTPUT_DIR%" --model %MODEL% --img_size %IMG_SIZE% --vae_embed_dim %VAE_EMBED_DIM% --vae_stride %VAE_STRIDE% --patch_size %PATCH_SIZE% --class_num %CLASS_NUM% --diffloss_d %DIFFLOSS_D% --diffloss_w %DIFFLOSS_W% --num_sampling_steps %NUM_SAMPLING_STEPS% --diffusion_batch_mul %DIFFUSION_BATCH_MUL% --planner_loss_weight 0.3 --candidate_pool_size %CANDIDATE_POOL_SIZE% --pseudo_target_type mixed_reveal --sampling_policy planner --candidate_selection_mode %CANDIDATE_SELECTION_MODE% --budget_mode soft --num_eval_images %NUM_EVAL_IMAGES% --eval_bsz %EVAL_BSZ% --num_workers %NUM_WORKERS% --num_states_per_image %NUM_STATES_PER_IMAGE% --rollout_horizon %ROLLOUT_HORIZON% --topk_eval %TOPK_EVAL% --num_iter %NUM_ITER% --state_step_min %STATE_STEP_MIN% --state_step_max %STATE_STEP_MAX% --local_radius %LOCAL_RADIUS% --uncertainty_mc_samples %UNCERTAINTY_MC_SAMPLES% --uncertainty_temperature %UNCERTAINTY_TEMPERATURE% --no_pin_mem"

echo !CMD! > "%OUTPUT_DIR%\run_args.txt"

powershell -NoProfile -ExecutionPolicy Bypass -Command "$cfg = [ordered]@{ table='surrogate_validity'; resume='%RESUME%'; data_path='%DATA_PATH%'; output_dir='%OUTPUT_DIR%'; model='%MODEL%'; num_eval_images=%NUM_EVAL_IMAGES%; eval_bsz=%EVAL_BSZ%; num_workers=%NUM_WORKERS%; num_states_per_image=%NUM_STATES_PER_IMAGE%; rollout_horizon=%ROLLOUT_HORIZON%; topk_eval=%TOPK_EVAL%; num_iter=%NUM_ITER%; state_step_min=%STATE_STEP_MIN%; state_step_max=%STATE_STEP_MAX%; local_radius=%LOCAL_RADIUS%; candidate_pool_size=%CANDIDATE_POOL_SIZE%; candidate_selection_mode='%CANDIDATE_SELECTION_MODE%'; uncertainty_mc_samples=%UNCERTAINTY_MC_SAMPLES%; uncertainty_temperature=%UNCERTAINTY_TEMPERATURE%; metrics='spearman,kendall_tau,topk_overlap,ndcg_at_k,regret'; ranking_rows='random,confidence,entropy,current_loss,planner,oracle'; timestamp=(Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ') }; $cfg | ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 '%OUTPUT_DIR%\config.json'"

echo ------------------------------------------------------------
echo RUN surrogate validity smoke test
echo START %DATE% %TIME%
echo ------------------------------------------------------------

powershell -NoProfile -ExecutionPolicy Bypass -Command "& { !CMD! 2>&1 | Tee-Object -FilePath '%OUTPUT_DIR%\eval.log' }"
if ERRORLEVEL 1 (
    echo [WARN] Surrogate validity run failed. See %OUTPUT_DIR%\eval.log
)

echo ============================================================
echo Produced result files
echo ============================================================
powershell -NoProfile -ExecutionPolicy Bypass -Command "Get-ChildItem -LiteralPath '%OUTPUT_DIR%' -File | Where-Object { $_.Name -match 'surrogate_validity|config|run_args|eval\.log' } | ForEach-Object { $_.FullName }"

echo ============================================================
echo END P0-3 Surrogate Validity: %DATE% %TIME%
echo Results root: %OUTPUT_DIR%
echo ============================================================
exit /b 0
