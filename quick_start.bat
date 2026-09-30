@echo off
REM Quick Start Script for Skin Cancer HSI Classification (Windows)
REM Run from project root: quick_start.bat

setlocal enabledelayedexpansion

set "PROJECT_ROOT=%cd%"
set "PREPROCESSING_DIR=%PROJECT_ROOT%\preprocessing"
set "MODEL_DIR=%PROJECT_ROOT%\model"

echo.
echo ==================================================
echo Skin Cancer HSI Classification - Quick Start
echo ==================================================
echo Project Root: %PROJECT_ROOT%
echo.

REM Phase 1: Preprocessing
echo PHASE 1: PREPROCESSING
echo ==================================================
cd /d "%PREPROCESSING_DIR%"

if not exist ".venv" (
    echo ERROR: .venv not found in preprocessing\
    echo Please create virtual environment first:
    echo   cd preprocessing
    echo   python -m venv .venv
    exit /b 1
)

call .venv\Scripts\activate.bat

echo [1/4] Generating train/val/test splits...
python scripts\run_splitting.py >nul 2>&1
if !errorlevel! equ 0 (
    echo 7 Splits generated
) else (
    echo X Failed to generate splits
    exit /b 1
)

echo [2/4] Running full preprocessing pipeline (Phase 10)...
python scripts\run_full_pipeline.py >nul 2>&1
if !errorlevel! equ 0 (
    echo 7 Pipeline complete (both per_pixel and per_band)
) else (
    echo X Pipeline failed
    exit /b 1
)

echo [3/4] Computing class weights...
python scripts\run_label_checks.py >nul 2>&1
if !errorlevel! equ 0 (
    echo 7 Class weights computed
) else (
    echo X Failed to compute weights
    exit /b 1
)

echo [4/4] Validating dataset schema...
python scripts\run_dataset_validation.py >nul 2>&1
if !errorlevel! equ 0 (
    echo 7 Dataset validation passed
) else (
    echo X Validation failed
    exit /b 1
)

call deactivate

REM Phase 2: Model Training
echo.
echo PHASE 2: MODEL TRAINING
echo ==================================================
cd /d "%MODEL_DIR%"

if not exist ".venv" (
    echo ERROR: .venv not found in model\
    echo Please create virtual environment first:
    echo   cd model
    echo   python -m venv .venv
    exit /b 1
)

call .venv\Scripts\activate.bat

echo [1/3] Verifying dataset loads...
python ..\dataset.py >nul 2>&1
if !errorlevel! equ 0 (
    echo 7 Dataset loads successfully
) else (
    echo X Dataset loading failed
    exit /b 1
)

echo [2/3] Checking splits configuration...
python check_splits.py >nul 2>&1
if !errorlevel! equ 0 (
    echo 7 Splits configuration OK
) else (
    echo X Splits check failed
    exit /b 1
)

echo [3/3] Training baseline model (quick test, 5 epochs)...
python -m src.cli train ^
    --model baseline ^
    --epochs 5 ^
    --batch_size 16 ^
    --lr 0.001 >nul 2>&1
if !errorlevel! equ 0 (
    echo 7 Training successful
) else (
    echo X Training failed
    exit /b 1
)

call deactivate

REM Summary
echo.
echo ==================================================
echo 7 QUICK START COMPLETE
echo ==================================================
echo.
echo Next steps:
echo.
echo 1. Review results in model\outputs\
echo.
echo 2. For full K-fold cross-validation:
echo    cd model
echo    .venv\Scripts\activate.bat
echo    python -m src.cli kfold --model self_attention --folds 5 --epochs 30
echo.
echo 3. For hyperparameter optimization:
echo    python -m src.cli optuna --trials 30
echo.
echo 4. For model evaluation:
echo    python evaluate_test.py
echo.
echo 5. For model export and deployment:
echo    python -m src.cli export --model outputs\best_model.pth
echo.
echo See IMPLEMENTATION_GUIDE.md for detailed commands.
echo.
