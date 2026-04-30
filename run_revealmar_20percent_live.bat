@echo off
setlocal

cd /d C:\caogang\RevealMAR\RevealMAR

if not exist logs mkdir logs

set TRAIN_LOG=logs\train_20percent_all.log
set EVAL_LOG=logs\eval_20percent_all.log

set DATA_PATH=C:\caogang\MAR\tiny-imagenet-200-20percent
set VAE_PATH=C:\caogang\RevealMAR\pretrained_models\vae\kl16.ckpt
set BASE_RESUME=C:\caogang\RevealMAR\pretrained_models\mar\mar_base

echo ==================================================
echo [SCRIPT START] %date% %time%
echo ==================================================
>> "%TRAIN_LOG%" echo ==================================================
>> "%TRAIN_LOG%" echo [SCRIPT START] %date% %time%
>> "%TRAIN_LOG%" echo ==================================================
>> "%EVAL_LOG%" echo ==================================================
>> "%EVAL_LOG%" echo [SCRIPT START] %date% %time%
>> "%EVAL_LOG%" echo ==================================================

REM ==================================================
REM TRAIN 1: mixed_reveal, w=0.3
REM ==================================================
echo.
echo ==================================================
echo [START TRAIN] mixed_reveal_w03 %date% %time%
echo ==================================================
>> "%TRAIN_LOG%" echo.
>> "%TRAIN_LOG%" echo ==================================================
>> "%TRAIN_LOG%" echo [START TRAIN] mixed_reveal_w03 %date% %time%
>> "%TRAIN_LOG%" echo ==================================================
powershell -NoProfile -Command "& { python main_revealmar.py --model revealmar_base --data_path '%DATA_PATH%' --vae_path '%VAE_PATH%' --resume '%BASE_RESUME%' --diffloss_d 6 --diffloss_w 1024 --epochs 10 --batch_size 128 --num_workers 0 --save_last_freq 1 --eval_freq 999999 --pseudo_target_type mixed_reveal --planner_loss_weight 0.3 --candidate_pool_size 8 --log_revealmar_losses --log_revealmar_loss_freq 20 --output_dir .\output_mixed40_w03_e10 2>&1 | Tee-Object -FilePath '%TRAIN_LOG%' -Append }"

echo [END TRAIN] mixed_reveal_w03 %date% %time%
>> "%TRAIN_LOG%" echo [END TRAIN] mixed_reveal_w03 %date% %time%

REM ==================================================
REM TRAIN 2: mixed_reveal, w=0.0
REM ==================================================
echo.
echo ==================================================
echo [START TRAIN] mixed_reveal_w00 %date% %time%
echo ==================================================
>> "%TRAIN_LOG%" echo.
>> "%TRAIN_LOG%" echo ==================================================
>> "%TRAIN_LOG%" echo [START TRAIN] mixed_reveal_w00 %date% %time%
>> "%TRAIN_LOG%" echo ==================================================
powershell -NoProfile -Command "& { python main_revealmar.py --model revealmar_base --data_path '%DATA_PATH%' --vae_path '%VAE_PATH%' --resume '%BASE_RESUME%' --diffloss_d 6 --diffloss_w 1024 --epochs 10 --batch_size 128 --num_workers 0 --save_last_freq 1 --eval_freq 999999 --pseudo_target_type mixed_reveal --planner_loss_weight 0.0 --candidate_pool_size 8 --log_revealmar_losses --log_revealmar_loss_freq 20 --output_dir .\output_mixed40_w00_e10 2>&1 | Tee-Object -FilePath '%TRAIN_LOG%' -Append }"

echo [END TRAIN] mixed_reveal_w00 %date% %time%
>> "%TRAIN_LOG%" echo [END TRAIN] mixed_reveal_w00 %date% %time%

REM ==================================================
REM TRAIN 3: pred_reveal, w=0.3
REM ==================================================
echo.
echo ==================================================
echo [START TRAIN] pred_reveal_w03 %date% %time%
echo ==================================================
>> "%TRAIN_LOG%" echo.
>> "%TRAIN_LOG%" echo ==================================================
>> "%TRAIN_LOG%" echo [START TRAIN] pred_reveal_w03 %date% %time%
>> "%TRAIN_LOG%" echo ==================================================
powershell -NoProfile -Command "& { python main_revealmar.py --model revealmar_base --data_path '%DATA_PATH%' --vae_path '%VAE_PATH%' --resume '%BASE_RESUME%' --diffloss_d 6 --diffloss_w 1024 --epochs 10 --batch_size 128 --num_workers 0 --save_last_freq 1 --eval_freq 999999 --pseudo_target_type pred_reveal --planner_loss_weight 0.3 --candidate_pool_size 8 --log_revealmar_losses --log_revealmar_loss_freq 20 --output_dir .\output_pred40_w03_e10 2>&1 | Tee-Object -FilePath '%TRAIN_LOG%' -Append }"

echo [END TRAIN] pred_reveal_w03 %date% %time%
>> "%TRAIN_LOG%" echo [END TRAIN] pred_reveal_w03 %date% %time%

echo.
echo ==================================================
echo [TRAIN ALL DONE] %date% %time%
echo ==================================================
>> "%TRAIN_LOG%" echo.
>> "%TRAIN_LOG%" echo ==================================================
>> "%TRAIN_LOG%" echo [TRAIN ALL DONE] %date% %time%
>> "%TRAIN_LOG%" echo ==================================================

REM ==================================================
REM EVAL 1: mixed_reveal, w=0.3
REM ==================================================
echo.
echo ==================================================
echo [START EVAL] mixed_reveal_w03 %date% %time%
echo ==================================================
>> "%EVAL_LOG%" echo.
>> "%EVAL_LOG%" echo ==================================================
>> "%EVAL_LOG%" echo [START EVAL] mixed_reveal_w03 %date% %time%
>> "%EVAL_LOG%" echo ==================================================
powershell -NoProfile -Command "& { python main_revealmar.py --model revealmar_base --data_path '%DATA_PATH%' --vae_path '%VAE_PATH%' --resume '.\output_mixed40_w03_e10' --evaluate --diffloss_d 6 --diffloss_w 1024 --num_images 2000 --eval_bsz 32 --num_iter 64 --num_sampling_steps 100 --cfg 1.0 --num_workers 0 --output_dir .\eval_mixed40_w03_e10 2>&1 | Tee-Object -FilePath 'logs\eval_mixed40_w03_e10.log' -Append | Tee-Object -FilePath '%EVAL_LOG%' -Append }"

echo [END EVAL] mixed_reveal_w03 %date% %time%
>> "%EVAL_LOG%" echo [END EVAL] mixed_reveal_w03 %date% %time%

REM ==================================================
REM EVAL 2: mixed_reveal, w=0.0
REM ==================================================
echo.
echo ==================================================
echo [START EVAL] mixed_reveal_w00 %date% %time%
echo ==================================================
>> "%EVAL_LOG%" echo.
>> "%EVAL_LOG%" echo ==================================================
>> "%EVAL_LOG%" echo [START EVAL] mixed_reveal_w00 %date% %time%
>> "%EVAL_LOG%" echo ==================================================
powershell -NoProfile -Command "& { python main_revealmar.py --model revealmar_base --data_path '%DATA_PATH%' --vae_path '%VAE_PATH%' --resume '.\output_mixed40_w00_e10' --evaluate --diffloss_d 6 --diffloss_w 1024 --num_images 2000 --eval_bsz 32 --num_iter 64 --num_sampling_steps 100 --cfg 1.0 --num_workers 0 --output_dir .\eval_mixed40_w00_e10 2>&1 | Tee-Object -FilePath 'logs\eval_mixed40_w00_e10.log' -Append | Tee-Object -FilePath '%EVAL_LOG%' -Append }"

echo [END EVAL] mixed_reveal_w00 %date% %time%
>> "%EVAL_LOG%" echo [END EVAL] mixed_reveal_w00 %date% %time%

REM ==================================================
REM EVAL 3: pred_reveal, w=0.3
REM ==================================================
echo.
echo ==================================================
echo [START EVAL] pred_reveal_w03 %date% %time%
echo ==================================================
>> "%EVAL_LOG%" echo.
>> "%EVAL_LOG%" echo ==================================================
>> "%EVAL_LOG%" echo [START EVAL] pred_reveal_w03 %date% %time%
>> "%EVAL_LOG%" echo ==================================================
powershell -NoProfile -Command "& { python main_revealmar.py --model revealmar_base --data_path '%DATA_PATH%' --vae_path '%VAE_PATH%' --resume '.\output_pred40_w03_e10' --evaluate --diffloss_d 6 --diffloss_w 1024 --num_images 2000 --eval_bsz 32 --num_iter 64 --num_sampling_steps 100 --cfg 1.0 --num_workers 0 --output_dir .\eval_pred40_w03_e10 2>&1 | Tee-Object -FilePath 'logs\eval_pred40_w03_e10.log' -Append | Tee-Object -FilePath '%EVAL_LOG%' -Append }"

echo [END EVAL] pred_reveal_w03 %date% %time%
>> "%EVAL_LOG%" echo [END EVAL] pred_reveal_w03 %date% %time%

echo.
echo ==================================================
echo [SCRIPT END] %date% %time%
echo ==================================================
>> "%EVAL_LOG%" echo.
>> "%EVAL_LOG%" echo ==================================================
>> "%EVAL_LOG%" echo [SCRIPT END] %date% %time%
>> "%EVAL_LOG%" echo ==================================================

echo.
echo Done.
echo Training log: %TRAIN_LOG%
echo Eval summary log: %EVAL_LOG%
echo Individual eval logs:
echo   logs\eval_mixed40_w03_e10.log
echo   logs\eval_mixed40_w00_e10.log
echo   logs\eval_pred40_w03_e10.log

pause
endlocal