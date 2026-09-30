#!/bin/bash
# Quick Start Script for Skin Cancer HSI Classification
# Run from project root: bash quick_start.sh
# Or on Windows: .\quick_start.bat (see batch version below)

set -e  # Exit on error

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREPROCESSING_DIR="$PROJECT_ROOT/preprocessing"
MODEL_DIR="$PROJECT_ROOT/model"

echo "=================================================="
echo "Skin Cancer HSI Classification - Quick Start"
echo "=================================================="
echo "Project Root: $PROJECT_ROOT"
echo ""

# Phase 1: Preprocessing
echo "PHASE 1: PREPROCESSING"
echo "=================================================="
cd "$PREPROCESSING_DIR"

if [ ! -d ".venv" ]; then
    echo "ERROR: .venv not found in preprocessing/"
    echo "Please create virtual environment first:"
    echo "  cd preprocessing"
    echo "  python -m venv .venv"
    exit 1
fi

source .venv/Scripts/activate

echo "[1/4] Generating train/val/test splits..."
python scripts/run_splitting.py > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✓ Splits generated"
else
    echo "✗ Failed to generate splits"
    exit 1
fi

echo "[2/4] Running full preprocessing pipeline (Phase 10)..."
python scripts/run_full_pipeline.py > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✓ Pipeline complete (both per_pixel and per_band)"
else
    echo "✗ Pipeline failed"
    exit 1
fi

echo "[3/4] Computing class weights..."
python scripts/run_label_checks.py > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✓ Class weights computed"
else
    echo "✗ Failed to compute weights"
    exit 1
fi

echo "[4/4] Validating dataset schema..."
python scripts/run_dataset_validation.py > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✓ Dataset validation passed"
else
    echo "✗ Validation failed"
    exit 1
fi

deactivate

# Phase 2: Model Training
echo ""
echo "PHASE 2: MODEL TRAINING"
echo "=================================================="
cd "$MODEL_DIR"

if [ ! -d ".venv" ]; then
    echo "ERROR: .venv not found in model/"
    echo "Please create virtual environment first:"
    echo "  cd model"
    echo "  python -m venv .venv"
    exit 1
fi

source .venv/Scripts/activate

echo "[1/3] Verifying dataset loads..."
python ../dataset.py > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✓ Dataset loads successfully"
else
    echo "✗ Dataset loading failed"
    exit 1
fi

echo "[2/3] Checking splits configuration..."
python check_splits.py > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✓ Splits configuration OK"
else
    echo "✗ Splits check failed"
    exit 1
fi

echo "[3/3] Training baseline model (quick test, 5 epochs)..."
python -m src.cli train \
    --model baseline \
    --epochs 5 \
    --batch_size 16 \
    --lr 0.001 > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✓ Training successful"
else
    echo "✗ Training failed"
    exit 1
fi

deactivate

# Summary
echo ""
echo "=================================================="
echo "✓ QUICK START COMPLETE"
echo "=================================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Review results in model/outputs/"
echo ""
echo "2. For full K-fold cross-validation:"
echo "   cd model && source .venv/Scripts/activate"
echo "   python -m src.cli kfold --model self_attention --folds 5 --epochs 30"
echo ""
echo "3. For hyperparameter optimization:"
echo "   python -m src.cli optuna --trials 30"
echo ""
echo "4. For model evaluation:"
echo "   python evaluate_test.py"
echo ""
echo "5. For model export and deployment:"
echo "   python -m src.cli export --model outputs/best_model.pth"
echo ""
echo "See IMPLEMENTATION_GUIDE.md for detailed commands."
echo ""
