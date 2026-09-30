#!/usr/bin/env python
"""
Phase 7 — Label integrity and class distribution checks.

  7.1 Binary/multi-label consistency across the entire dataset
  7.2 Class distribution vs. paper's reported percentages
  7.3 Class weight computation

Saves: outputs/reports/label_checks_report.json
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
PREPROC = Path(__file__).parent.parent
sys.path.insert(0, str(PREPROC))

from hsi_preprocessing.label_utils import (
    validate_all_label_pairs,
    compute_class_distribution,
    compare_distribution_with_paper,
    compute_class_weights,
    PAPER_DISTRIBUTION,
)

NPY_ROOT = ROOT / "extracted_dataset" / "npy_arrays"
REPORT_DIR = PREPROC / "outputs" / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    sample_dirs = sorted(d for d in NPY_ROOT.iterdir() if d.is_dir())
    print(f"Checking labels for {len(sample_dirs)} samples...\n")

    records = []
    multi_labels = []

    for sd in sample_dirs:
        sid = sd.name
        label_file = sd / f"{sid}__labels.json"
        if not label_file.exists():
            print(f"  [MISSING] {sid}: labels.json not found")
            continue
        with open(label_file) as f:
            labels = json.load(f)
        records.append({
            "sample_id": sid,
            "binaryLabel": labels.get("binaryLabel", "?"),
            "multiLabel": labels.get("multiLabel", "?"),
        })
        multi_labels.append(labels.get("multiLabel", "?"))

    # --- Phase 7.1: Consistency check
    valid, invalid = validate_all_label_pairs(records)
    print(f"Phase 7.1 — Binary/Multi-Label Consistency:")
    print(f"  Valid pairs:   {len(valid)}")
    print(f"  Invalid pairs: {len(invalid)}")
    if invalid:
        print("  *** INCONSISTENCIES FOUND (data quality issue!) ***")
        for rec in invalid:
            print(f"    {rec['sample_id']}: {rec['binaryLabel']} / {rec['multiLabel']}")
    else:
        print("  OK Zero inconsistencies — all binary/multi pairs are consistent")

    # --- Phase 7.2: Class distribution
    actual_dist = compute_class_distribution(multi_labels)
    comparison = compare_distribution_with_paper(actual_dist)

    print(f"\nPhase 7.2 — Class Distribution vs Paper:")
    print(f"  {'Class':6s} {'Actual':8s} {'Paper':8s} {'Diff':8s} {'OK?':5s}")
    print(f"  {'-'*40}")
    all_within = True
    for cls, info in sorted(comparison.items()):
        ok = "OK" if info["within_tolerance"] else "FAIL"
        if not info["within_tolerance"]:
            all_within = False
        print(f"  {cls:6s} {info['actual']:8.3f} {info['paper']:8.3f} "
              f"{info['diff']:8.3f} {ok}")
    if all_within:
        print("  OK All classes within tolerance of paper's reported distribution")
    else:
        print("  WARN Some classes differ from paper — may be expected with patient-level split")

    # --- Phase 7.3: Class weights
    weights = compute_class_weights(multi_labels, method="inverse_frequency")
    print(f"\nPhase 7.3 — Class Weights (inverse frequency):")
    for cls, w in sorted(weights.items()):
        print(f"  {cls}: {w:.4f}")

    # --- Save report
    report = {
        "total_samples": len(records),
        "consistency": {
            "valid_count": len(valid),
            "invalid_count": len(invalid),
            "inconsistent_samples": [r["sample_id"] for r in invalid],
        },
        "class_distribution": {
            cls: {"actual": v["actual"], "paper": v["paper"],
                  "diff": v["diff"], "within_tolerance": v["within_tolerance"]}
            for cls, v in comparison.items()
        },
        "class_weights": weights,
    }
    out_path = REPORT_DIR / "label_checks_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved -> {out_path}")


if __name__ == "__main__":
    main()
