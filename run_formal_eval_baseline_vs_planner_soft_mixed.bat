@echo off
setlocal enabledelayedexpansion

REM =========================
REM Formal evaluation:
REM same checkpoint, baseline vs planner-soft-mixed
REM =========================

cd /d C:\caogang\RevealMAR\RevealMAR

echo ========================================
echo Formal evaluation: baseline vs planner-soft-mixed
echo Checkpoint: .\output_new_gt40_w03_e10
echo Dataset: C:\caogang\MAR\tiny-imagenet-200-40percent
echo Num images: 4000
echo ========================================

REM -------- Common paths --------
set DATA_PATH=C:\caogang\MAR\tiny-imagenet-200-40percent
set VAE_PATH=C:\caogang\RevealMAR\pretrained_models\vae\kl16.ckpt
set RESUME_DIR=.\output_new_gt40_w03_e10

REM -------- Common eval args --------
set COMMON_ARGS=--model revealmar_base --data_path "%DATA_PATH%" --vae_path "%VAE_PATH%" --resume "%RESUME_DIR%" --evaluate --diffloss_d 6 --diffloss_w 1024 --num_images 4000 --eval_bsz 32 --num_iter 8 --num_sampling_steps 20 --cfg 1.0 --num_workers 0 --pseudo_target_type mixed_reveal --planner_loss_weight 0.3 --candidate_pool_size 8

echo.
echo [1/2] Running baseline formal evaluation...
python main_revealmar.py %COMMON_ARGS% --sampling_policy baseline --output_dir .\eval_formal_baseline > .\logs\eval_formal_baseline.log 2>&1
if errorlevel 1 (
    echo [FAILED] baseline formal evaluation
    echo Check log: .\logs\eval_formal_baseline.log
    pause
    exit /b 1
)
echo [DONE] baseline formal evaluation
echo Log saved to .\logs\eval_formal_baseline.log

echo.
echo [2/2] Running planner formal evaluation (soft + mixed)...
python main_revealmar.py %COMMON_ARGS% --sampling_policy planner --budget_mode soft --candidate_selection_mode mixed --output_dir .\eval_formal_planner_soft_mixed > .\logs\eval_formal_planner_soft_mixed.log 2>&1
if errorlevel 1 (
    echo [FAILED] planner formal evaluation
    echo Check log: .\logs\eval_formal_planner_soft_mixed.log
    pause
    exit /b 1
)
echo [DONE] planner formal evaluation
echo Log saved to .\logs\eval_formal_planner_soft_mixed.log

echo.
echo ========================================
echo ALL DONE
echo Baseline output: .\eval_formal_baseline
echo Planner output : .\eval_formal_planner_soft_mixed
echo Baseline log   : .\logs\eval_formal_baseline.log
echo Planner log    : .\logs\eval_formal_planner_soft_mixed.log
echo ========================================

pause
exit /b 0