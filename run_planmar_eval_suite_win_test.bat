@echo off
setlocal enabledelayedexpansion

REM ============================================================
REM Windows one-click evaluation test for PlanMAR-S
REM Only evaluation, no training.
REM This script calls planmar_eval_suite.py, which calls:
REM python main_revealmar.py --evaluate
REM Generation and metrics are handled by engine_mar.py.
REM ============================================================

cd /d "%~dp0"

REM =========================
REM 0. Basic paths
REM =========================

set CODE_DIR=%CD%

REM Your lab server paths
set DATA_PATH=C:\caogang\MAR\tiny-imagenet-200-40percent
set VAE_PATH=C:\caogang\RevealMAR\pretrained_models\vae\kl16.ckpt
set BASE_RESUME=C:\caogang\RevealMAR\RevealMAR

REM Output folder
set EVAL_ROOT=%CODE_DIR%\outputs\planmar_paper_eval_win_test

REM Evaluation config
set NUM_IMAGES=1000
set EVAL_BSZ=32
set NUM_ITERS=256
set METHODS=baseline,planner

REM Because planmar_eval_suite.py expects PRETRAIN_ROOT/vae/kl16.ckpt,
REM we set PRETRAIN_ROOT to C:\caogang\RevealMAR\pretrained_models.
set PRETRAIN_ROOT=C:\caogang\RevealMAR\pretrained_models

REM =========================
REM 1. Activate conda env
REM =========================

call conda activate revealmar

IF ERRORLEVEL 1 (
    echo Failed to activate conda env by "conda activate revealmar".
    echo Trying default Miniconda path...
    call "%USERPROFILE%\miniconda3\Scripts\activate.bat" revealmar
)

IF ERRORLEVEL 1 (
    echo ERROR: Failed to activate conda environment revealmar.
    pause
    exit /b 1
)

set OMP_NUM_THREADS=8
set PYTHONUNBUFFERED=1

echo ============================================================
echo Windows PlanMAR-S evaluation test
echo CODE_DIR=%CODE_DIR%
echo DATA_PATH=%DATA_PATH%
echo VAE_PATH=%VAE_PATH%
echo BASE_RESUME=%BASE_RESUME%
echo PRETRAIN_ROOT=%PRETRAIN_ROOT%
echo EVAL_ROOT=%EVAL_ROOT%
echo NUM_IMAGES=%NUM_IMAGES%
echo EVAL_BSZ=%EVAL_BSZ%
echo NUM_ITERS=%NUM_ITERS%
echo METHODS=%METHODS%
echo ============================================================

python -V

REM =========================
REM 2. Check paths
REM =========================

if not exist "%CODE_DIR%\main_revealmar.py" (
    echo ERROR: main_revealmar.py not found in %CODE_DIR%
    pause
    exit /b 1
)

if not exist "%CODE_DIR%\engine_mar.py" (
    echo ERROR: engine_mar.py not found in %CODE_DIR%
    pause
    exit /b 1
)

if not exist "%CODE_DIR%\planmar_eval_suite.py" (
    echo ERROR: planmar_eval_suite.py not found in %CODE_DIR%
    pause
    exit /b 1
)

if not exist "%DATA_PATH%\train" (
    echo ERROR: %DATA_PATH%\train does not exist.
    echo DATA_PATH must be the parent folder containing train\.
    pause
    exit /b 1
)

if not exist "%VAE_PATH%" (
    echo ERROR: VAE checkpoint does not exist:
    echo %VAE_PATH%
    pause
    exit /b 1
)

if not exist "%BASE_RESUME%\checkpoint-last.pth" (
    echo ERROR: checkpoint-last.pth not found:
    echo %BASE_RESUME%\checkpoint-last.pth
    echo BASE_RESUME must be a directory containing checkpoint-last.pth.
    pause
    exit /b 1
)

mkdir "%EVAL_ROOT%" 2>nul

REM Syntax check
python -m py_compile main_revealmar.py
if ERRORLEVEL 1 (
    echo ERROR: main_revealmar.py syntax check failed.
    pause
    exit /b 1
)

python -m py_compile engine_mar.py
if ERRORLEVEL 1 (
    echo ERROR: engine_mar.py syntax check failed.
    pause
    exit /b 1
)

python -m py_compile planmar_eval_suite.py
if ERRORLEVEL 1 (
    echo ERROR: planmar_eval_suite.py syntax check failed.
    pause
    exit /b 1
)

REM Show GPU
nvidia-smi

REM =========================
REM 3. Run evaluation only
REM =========================

python planmar_eval_suite.py ^
  --code_dir "%CODE_DIR%" ^
  --data_root "%DATA_PATH%" ^
  --pretrain_root "%PRETRAIN_ROOT%" ^
  --planmar_ckpt "%BASE_RESUME%" ^
  --baseline_ckpt "%BASE_RESUME%" ^
  --eval_root "%EVAL_ROOT%" ^
  --num_images %NUM_IMAGES% ^
  --eval_bsz %EVAL_BSZ% ^
  --num_iters %NUM_ITERS% ^
  --methods %METHODS% ^
  --continue_on_error

echo ============================================================
echo Evaluation finished.
echo Results root: %EVAL_ROOT%
echo Logs: %EVAL_ROOT%\logs
echo ============================================================

pause