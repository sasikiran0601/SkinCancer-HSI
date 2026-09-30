#!/usr/bin/env python
"""
Phase 0.2 + 1.4 — Dataset-wide schema validation and calibratedHsCube sanity sweep.

Checks every sample in the dataset:
  1. Files present and correct shapes/dtypes (Phase 0.2)
  2. calibratedHsCube values in expected range, no NaN/Inf (Phase 1.4)

Saves report to: outputs/reports/dataset_validation_report.json
"""

import json
import sys
from pathlib import Path

# Allow running from anywhere
ROOT = Path(__file__).parent.parent.parent   # SkinCancer_DATASET/
PREPROC = Path(__file__).parent.parent       # preprocessing/
sys.path.insert(0, str(PREPROC))

import numpy as np
from hsi_preprocessing.validation import validate_mat_structure
from hsi_preprocessing.calibration import validate_calibrated_cube

NPY_ROOT = ROOT / "extracted_dataset" / "npy_arrays"
REPORT_DIR = PREPROC / "outputs" / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    sample_dirs = sorted(d for d in NPY_ROOT.iterdir() if d.is_dir())
    print(f"Scanning {len(sample_dirs)} samples in {NPY_ROOT}\n")

    schema_results = []
    calibration_results = []
    schema_failures = []
    calib_range_failures = []

    for sd in sample_dirs:
        sid = sd.name

        # --- Phase 0.2: Schema check
        schema_result = validate_mat_structure(sd, sid)
        schema_results.append(schema_result)
        if not schema_result["valid"]:
            schema_failures.append(sid)
            print(f"  [SCHEMA FAIL] {sid}: {schema_result['reason']}")
            continue

        # --- Phase 1.4: calibratedHsCube sanity check
        cube = np.load(str(sd / f"{sid}__calibratedHsCube.npy"))
        calib = validate_calibrated_cube(cube, sample_id=sid)
        calibration_results.append(calib)
        if not calib["valid"]:
            calib_range_failures.append(sid)
            print(f"  [CALIB WARN] {sid}: {calib['reason']}")

    # --- Summary
    n_total = len(sample_dirs)
    n_schema_ok = sum(r["valid"] for r in schema_results)
    n_calib_ok  = sum(r["valid"] for r in calibration_results)
    n_calib_checked = len(calibration_results)

    print(f"\n{'='*60}")
    print(f"Schema validation: {n_schema_ok}/{n_total} passed")
    print(f"Calibration check: {n_calib_ok}/{n_calib_checked} passed (of those with valid schema)")
    if schema_failures:
        print(f"Schema failures: {schema_failures}")
    if calib_range_failures:
        print(f"Calibration range warnings: {calib_range_failures}")
    print(f"{'='*60}\n")

    # --- Save report
    report = {
        "total_samples": n_total,
        "schema_passed": n_schema_ok,
        "schema_failed": len(schema_failures),
        "schema_failures": schema_failures,
        "calibration_checked": n_calib_checked,
        "calibration_passed": n_calib_ok,
        "calibration_warnings": calib_range_failures,
        "per_sample_schema": schema_results,
        "per_sample_calibration": calibration_results,
    }
    out_path = REPORT_DIR / "dataset_validation_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report saved -> {out_path}")

    # Exit with error if any schema failures (but not for calibration warnings)
    if schema_failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
