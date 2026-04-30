@echo off
setlocal

cd /d C:\caogang\RevealMAR\RevealMAR

if not exist logs mkdir logs
set LOGFILE=logs\planner_sampling_smoke_1000.log

set DATA_PATH=C:\caogang\MAR\tiny-imagenet-200-40percent
set VAE_PATH=C:\caogang\RevealMAR\pretrained_models\vae\kl16.ckpt
set RESUME_DIR=.\output_new_gt40_w03_e10

set NUM_IMAGES=1000
set EVAL_BSZ=20
set NUM_ITER=8
set NUM_SAMPLING_STEPS=20

echo ================================================== > "%LOGFILE%"
echo [PLANNER SAMPLING SMOKE START] %date% %time% >> "%LOGFILE%"
echo ================================================== >> "%LOGFILE%"

REM ==================================================
REM 1) baseline sanity
REM ==================================================
echo.
echo ==================================================
echo [1/3] baseline sanity
echo ==================================================
powershell -NoProfile -Command "& { python main_revealmar.py --model revealmar_base --data_path '%DATA_PATH%' --vae_path '%VAE_PATH%' --resume '%RESUME_DIR%' --evaluate --diffloss_d 6 --diffloss_w 1024 --num_images %NUM_IMAGES% --eval_bsz %EVAL_BSZ% --num_iter %NUM_ITER% --num_sampling_steps %NUM_SAMPLING_STEPS% --cfg 1.0 --num_workers 0 --sampling_policy baseline --pseudo_target_type mixed_reveal --planner_loss_weight 0.3 --candidate_pool_size 8 --output_dir '.\eval_planner_smoke_baseline_1000' 2>&1 | Tee-Object -FilePath '%LOGFILE%' -Append }"
if errorlevel 1 (
    echo [FAILED] baseline sanity
    pause
    exit /b 1
)

REM ==================================================
REM 2) planner sanity: hard budget + topk
REM ==================================================
echo.
echo ==================================================
echo [2/3] planner sanity - hard budget + topk
echo ==================================================
powershell -NoProfile -Command "& { python main_revealmar.py --model revealmar_base --data_path '%DATA_PATH%' --vae_path '%VAE_PATH%' --resume '%RESUME_DIR%' --evaluate --diffloss_d 6 --diffloss_w 1024 --num_images %NUM_IMAGES% --eval_bsz %EVAL_BSZ% --num_iter %NUM_ITER% --num_sampling_steps %NUM_SAMPLING_STEPS% --cfg 1.0 --num_workers 0 --sampling_policy planner --budget_mode hard --candidate_selection_mode topk --pseudo_target_type mixed_reveal --planner_loss_weight 0.3 --candidate_pool_size 8 --output_dir '.\eval_planner_smoke_hard_topk_1000' 2>&1 | Tee-Object -FilePath '%LOGFILE%' -Append }"
if errorlevel 1 (
    echo [FAILED] planner sanity - hard budget + topk
    pause
    exit /b 1
)

REM ==================================================
REM 3) planner sanity: soft budget + mixed candidate
REM ==================================================
echo.
echo ==================================================
echo [3/3] planner sanity - soft budget + mixed candidate
echo ==================================================
powershell -NoProfile -Command "& { python main_revealmar.py --model revealmar_base --data_path '%DATA_PATH%' --vae_path '%VAE_PATH%' --resume '%RESUME_DIR%' --evaluate --diffloss_d 6 --diffloss_w 1024 --num_images %NUM_IMAGES% --eval_bsz %EVAL_BSZ% --num_iter %NUM_ITER% --num_sampling_steps %NUM_SAMPLING_STEPS% --cfg 1.0 --num_workers 0 --sampling_policy planner --budget_mode soft --candidate_selection_mode mixed --pseudo_target_type mixed_reveal --planner_loss_weight 0.3 --candidate_pool_size 8 --output_dir '.\eval_planner_smoke_soft_mixed_1000' 2>&1 | Tee-Object -FilePath '%LOGFILE%' -Append }"
if errorlevel 1 (
    echo [FAILED] planner sanity - soft budget + mixed candidate
    pause
    exit /b 1
)

echo.
echo ==================================================
echo [ALL DONE] planner sampling smoke finished successfully
echo ==================================================
echo [ALL DONE] planner sampling smoke finished successfully >> "%LOGFILE%"

pause
endlocal