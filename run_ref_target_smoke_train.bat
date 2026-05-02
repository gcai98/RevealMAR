@echo off
setlocal enabledelayedexpansion

REM ============================================================
REM Tiny smoke training run for paper-aligned ref_* pseudo targets.
REM This is for code-path validation only, not paper-scale training.
REM ============================================================

set "CODE_DIR=C:\caogang\RevealMAR\RevealMAR"
set "DATA_PATH=C:\caogang\MAR\tiny-imagenet-200-1percent"
set "VAE_PATH=C:\caogang\RevealMAR\pretrained_models\vae\kl16.ckpt"
set "OUTPUT_DIR=C:\caogang\RevealMAR\RevealMAR\p0_runs\ref_target_smoke_train"
set "SOURCE_CHECKPOINT=C:\caogang\RevealMAR\RevealMAR\checkpoint-last.pth"
set "MODEL_ONLY_INIT=%OUTPUT_DIR%\model_only_init"
set "RESUME=%MODEL_ONLY_INIT%"

echo ============================================================
echo START REF TARGET SMOKE TRAIN: %DATE% %TIME%
echo CODE_DIR=%CODE_DIR%
echo DATA_PATH=%DATA_PATH%
echo RESUME=%RESUME%
echo OUTPUT_DIR=%OUTPUT_DIR%
echo ============================================================

if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"
if not exist "%MODEL_ONLY_INIT%" mkdir "%MODEL_ONLY_INIT%"
cd /d "%CODE_DIR%"

echo Creating model-only init checkpoint for smoke training.
python tools\make_model_only_checkpoint.py --src "%SOURCE_CHECKPOINT%" --dst_dir "%MODEL_ONLY_INIT%"
if ERRORLEVEL 1 (
    echo [WARN] Failed to create model-only init checkpoint.
    exit /b 1
)

set "CMD=python main_revealmar.py --model revealmar_base --img_size 256 --vae_path "%VAE_PATH%" --vae_embed_dim 16 --vae_stride 16 --patch_size 1 --data_path "%DATA_PATH%" --resume "%RESUME%" --output_dir "%OUTPUT_DIR%" --batch_size 4 --epochs 1 --save_last_freq 1 --eval_freq 999999 --num_workers 0 --diffloss_d 6 --diffloss_w 1024 --num_sampling_steps 100 --diffusion_batch_mul 1 --pseudo_target_type ref_mixed_reveal --ref_target_horizon 1 --ref_target_local_radius 1 --ref_target_mix_alpha 0.5 --ref_target_loss feature_mse --ref_target_max_candidates 8 --log_ref_target_debug --log_ref_target_freq 1 --log_revealmar_losses --log_revealmar_loss_freq 1 --dist_url env://"

echo !CMD! > "%OUTPUT_DIR%\run_args.txt"

powershell -NoProfile -ExecutionPolicy Bypass -Command "$cfg = [ordered]@{ table='ref_target_smoke_train'; data_path='%DATA_PATH%'; resume='%RESUME%'; output_dir='%OUTPUT_DIR%'; model='revealmar_base'; batch_size=4; epochs=1; pseudo_target_type='ref_mixed_reveal'; ref_target_horizon=1; ref_target_local_radius=1; ref_target_mix_alpha=0.5; ref_target_loss='feature_mse'; ref_target_max_candidates=8; timestamp=(Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ') }; $cfg | ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 '%OUTPUT_DIR%\config.json'"

powershell -NoProfile -ExecutionPolicy Bypass -Command "& { !CMD! 2>&1 | Tee-Object -FilePath '%OUTPUT_DIR%\train.log' }"
if ERRORLEVEL 1 (
    echo [WARN] Ref target smoke training failed. See %OUTPUT_DIR%\train.log
)

echo ============================================================
echo END REF TARGET SMOKE TRAIN: %DATE% %TIME%
echo Results root: %OUTPUT_DIR%
echo ============================================================
exit /b 0
