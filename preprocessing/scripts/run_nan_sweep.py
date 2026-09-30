#!/usr/bin/env python
"""
Phase 3.3 — Real-data NaN/Inf sweep.

Runs the full NaN/Inf fixing step on every sample, logs:
  - How many samples needed intervention
  - Which bands were affected
  - Total count of non-finite values fixed

Saves: outputs/reports/nan_inf_sweep_report.json
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
PREPROC = Path(__file__).parent.parent
sys.path.insert(0, str(PREPROC))

import numpy as np
from hsi_preprocessing.calibration import fix_non_finite

NPY_ROOT = ROOT / "extracted_dataset" / "npy_arrays"
REPORT_DIR = PREPROC / "outputs" / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    sample_dirs = sorted(d for d in NPY_ROOT.iterdir() if d.is_dir())
    print(f"Sweeping {len(sample_dirs)} samples for NaN/Inf values...\n")

    intervention_log = {}
    total_fixed = 0
    total_checked = 0

    for sd in sample_dirs:
        sid = sd.name
        cube_path = sd / f"{sid}__calibratedHsCube.npy"
        if not cube_path.exists():
            print(f"  [SKIP] {sid}: calibratedHsCube.npy not found")
            continue

        cube = np.load(str(cube_path))
        n_nonfinite_before = int((~np.isfinite(cube)).sum())

        if n_nonfinite_before > 0:
            _, n_fixed = fix_non_finite(cube)
            intervention_log[sid] = {
                "n_nonfinite_found": n_nonfinite_before,
                "n_fixed": n_fixed,
                "affected_bands": list(
                    map(int, np.where(~np.isfinite(cube).any(axis=(1, 2)))[0])
                ),
            }
            total_fixed += n_fixed
            print(f"  [FIX] {sid}: {n_nonfinite_before} non-finite values fixed")

        total_checked += 1

    n_with_intervention = len(intervention_log)
    frac = n_with_intervention / total_checked if total_checked > 0 else 0.0

    print(f"\n{'='*60}")
    print(f"Samples checked      : {total_checked}")
    print(f"Samples with NaN/Inf : {n_with_intervention} ({frac:.1%})")
    print(f"Total values fixed   : {total_fixed}")
    print(f"{'='*60}\n")

    report = {
        "total_checked": total_checked,
        "samples_with_intervention": n_with_intervention,
        "fraction_with_intervention": frac,
        "total_values_fixed": total_fixed,
        "intervention_log": intervention_log,
    }
    out_path = REPORT_DIR / "nan_inf_sweep_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report saved -> {out_path}")


if __name__ == "__main__":
    main()
