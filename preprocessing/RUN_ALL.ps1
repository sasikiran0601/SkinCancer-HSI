# HSI Preprocessing — Run All Commands
# Run every command from this directory:
#   c:\Users\sasik\OneDrive\Documents\SkinCancer_DATASET\preprocessing

# ============================================================
# STEP 1 — Activate the virtual environment
# ============================================================
.venv\Scripts\Activate.ps1

# ============================================================
# STEP 2 — Run the full test suite (106 tests)
# ============================================================
python -m pytest tests/ -v

# ============================================================
# STEP 3 — Phase 0.2 + 1.4: Schema validation + calibration sweep
#   Confirms all 76 samples have correct files, shapes, labels.
#   Saves: outputs/reports/dataset_validation_report.json
# ============================================================
python scripts/run_dataset_validation.py

# ============================================================
# STEP 4 — Phase 3.3: NaN/Inf sweep across all 76 samples
#   Logs which samples need NaN fixing and how many values.
#   Saves: outputs/reports/nan_inf_sweep_report.json
# ============================================================
python scripts/run_nan_sweep.py

# ============================================================
# STEP 5 — Phase 7: Label integrity checks + class weights
#   Validates binary/multi-label consistency, class distribution
#   vs. paper, and computes inverse-frequency class weights.
#   Saves: outputs/reports/label_checks_report.json
# ============================================================
python scripts/run_label_checks.py

# ============================================================
# STEP 6 — Phase 2: Patient-independent splits
#   Generates train/val/test splits by patient (not image ID).
#   Demonstrates and quantifies leakage vs image-level splits.
#   Saves: outputs/splits/patient_splits.json
#          outputs/splits/image_level_splits.json
#          outputs/reports/split_comparison_report.json
# ============================================================
python scripts/run_splitting.py

# ============================================================
# STEP 7 — Phase 4.1: Per-band noise/variance profile
#   Computes and plots band-level variance to verify the paper's
#   trim range (drop first 4, last 5 bands) is data-justified.
#   Saves: outputs/plots/band_variance_profile.png
#          outputs/reports/band_variance_report.json
# ============================================================
python scripts/run_band_noise_profile.py

# ============================================================
# STEP 8 — Phase 8: RGB cross-consistency check
#   Derives RGB from calibratedHsCube, compares to spectralRGB,
#   reports Pearson r and cosine similarity for 10 samples.
#   Saves: outputs/plots/rgb_consistency.png
#          outputs/reports/rgb_consistency_report.json
# ============================================================
python scripts/run_rgb_consistency.py

# ============================================================
# STEP 9 — Phase 10: Full end-to-end pipeline (MAIN STEP)
#   Processes all 76 samples through the complete pipeline:
#     fix NaN/Inf -> trim bands -> moving-avg filter -> normalize
#   Runs TWICE: per-pixel norm (paper baseline) + per-band (ablation)
#   Also plots spectral signatures per class (Phase 1.6).
#   Saves: outputs/processed/per_pixel/{sample_id}/*__processed.npy
#          outputs/processed/per_band/{sample_id}/*__processed.npy
#          outputs/processed/per_pixel/preprocessing_config.json
#          outputs/processed/per_pixel/checksums.json
#          outputs/plots/spectral_signatures_per_class.png
#          outputs/processed/per_pixel/pipeline_report.json
# ============================================================
python scripts/run_full_pipeline.py
