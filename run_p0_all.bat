@echo off
setlocal enabledelayedexpansion

REM ============================================================
REM Unified P0 smoke pipeline runner.
REM Runs main-result evaluation, main-result collection,
REM surrogate-validity evaluation, and surrogate-validity collection.
REM ============================================================

set "CODE_DIR=C:\caogang\RevealMAR\RevealMAR"
set "MAIN_ROOT=C:\caogang\RevealMAR\RevealMAR\p0_runs\main_results_trained_ckpt_full_policies"
set "SURROGATE_ROOT=C:\caogang\RevealMAR\RevealMAR\p0_runs\surrogate_validity_trained_ckpt"

echo ============================================================
echo P0 SMOKE PIPELINE START: %DATE% %TIME%
echo CODE_DIR=%CODE_DIR%
echo MAIN_ROOT=%MAIN_ROOT%
echo SURROGATE_ROOT=%SURROGATE_ROOT%
echo ============================================================

cd /d "%CODE_DIR%"
if ERRORLEVEL 1 (
    echo [ERROR] Could not cd to CODE_DIR: %CODE_DIR%
    exit /b 1
)

set "MISSING=0"
for %%F in (
    run_main_results.bat
    collect_p0_results.py
    run_surrogate_validity.bat
    collect_surrogate_validity.py
) do (
    if not exist "%%F" (
        echo [ERROR] Missing required file: %%F
        set "MISSING=1"
    )
)
if "%MISSING%"=="1" (
    echo [ERROR] Required pipeline files are missing. Aborting.
    exit /b 1
)

echo ============================================================
echo STEP 1: MAIN RESULTS
echo ============================================================
call run_main_results.bat
if ERRORLEVEL 1 (
    echo [WARN] run_main_results.bat reported failure. Continuing to collector if logs exist.
)

echo ============================================================
echo STEP 2: COLLECT MAIN RESULTS
echo ============================================================
python collect_p0_results.py --root "%MAIN_ROOT%"
if ERRORLEVEL 1 (
    echo [WARN] collect_p0_results.py reported failure. Continuing to surrogate validity.
)

echo ============================================================
echo STEP 3: SURROGATE VALIDITY
echo ============================================================
call run_surrogate_validity.bat
if ERRORLEVEL 1 (
    echo [WARN] run_surrogate_validity.bat reported failure. Continuing to surrogate collector.
)

echo ============================================================
echo STEP 4: COLLECT SURROGATE VALIDITY
echo ============================================================
python collect_surrogate_validity.py --root "%SURROGATE_ROOT%"
if ERRORLEVEL 1 (
    echo [WARN] collect_surrogate_validity.py reported failure.
)

echo ============================================================
echo P0 SMOKE PIPELINE END: %DATE% %TIME%
echo Expected outputs:
echo   %MAIN_ROOT%\p0_summary.csv
echo   %MAIN_ROOT%\p0_summary.json
echo   %SURROGATE_ROOT%\surrogate_validity_summary.csv
echo   %SURROGATE_ROOT%\surrogate_validity_summary.json
echo ============================================================
exit /b 0
