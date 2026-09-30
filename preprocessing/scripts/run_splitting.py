#!/usr/bin/env python
"""
Phase 2 — Patient-independent splitting.

Generates train/val/test splits by patient ID (not image ID),
compares against image-level baseline, and enforces the leakage assertion.

Saves:
  outputs/splits/patient_splits.json
  outputs/splits/image_level_splits.json
  outputs/reports/split_comparison_report.json
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
PREPROC = Path(__file__).parent.parent
sys.path.insert(0, str(PREPROC))

import numpy as np
from hsi_preprocessing.splitting import (
    generate_patient_splits,
    generate_image_level_splits,
    assert_no_patient_leakage,
    count_patient_overlap,
    save_splits,
    validate_patient_ids,
)

NPY_ROOT = ROOT / "extracted_dataset" / "npy_arrays"
SPLITS_DIR = PREPROC / "outputs" / "splits"
REPORT_DIR = PREPROC / "outputs" / "reports"
SPLITS_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def load_all_labels(npy_root: Path):
    """Load multiLabel for every sample in the dataset."""
    sample_ids = sorted(d.name for d in npy_root.iterdir() if d.is_dir())
    labels = []
    for sid in sample_ids:
        label_file = npy_root / sid / f"{sid}__labels.json"
        with open(label_file) as f:
            labels.append(json.load(f)["multiLabel"])
    return sample_ids, labels


def main():
    print(f"Loading samples from {NPY_ROOT}...")
    sample_ids, labels = load_all_labels(NPY_ROOT)
    n = len(sample_ids)
    print(f"Found {n} samples.\n")

    # --- Phase 2.1: Patient ID stats
    valid_map, unparseable = validate_patient_ids(sample_ids)
    unique_patients = set(valid_map.values())
    print(f"Phase 2.1 — Patient ID Extraction:")
    print(f"  Unique patients: {len(unique_patients)}")
    print(f"  Unparseable IDs: {len(unparseable)}")
    if unparseable:
        print(f"  Unparseable: {unparseable}")

    # --- Phase 2.2: Patient-independent splits
    patient_splits = generate_patient_splits(sample_ids, labels, seed=42)
    print(f"\nPhase 2.2 — Patient-independent splits:")
    for split, sids in patient_splits.items():
        print(f"  {split:5s}: {len(sids):3d} images")

    # Assert no leakage
    assert_no_patient_leakage(patient_splits)
    patient_overlap = count_patient_overlap(patient_splits)
    print(f"  Patient overlap: {patient_overlap} OK (zero leakage confirmed)")

    # Save patient splits
    save_splits(
        patient_splits,
        SPLITS_DIR / "patient_splits.json",
        config={"strategy": "patient_independent", "seed": 42},
    )

    # --- Phase 2.3: Image-level baseline (leakage demo)
    image_splits = generate_image_level_splits(sample_ids, labels, seed=42)
    image_overlap = count_patient_overlap(image_splits)
    print(f"\nPhase 2.3 — Image-level split (baseline / leakage demo):")
    for split, sids in image_splits.items():
        print(f"  {split:5s}: {len(sids):3d} images")
    print(f"  Patient overlap: {image_overlap}")

    save_splits(
        image_splits,
        SPLITS_DIR / "image_level_splits.json",
        config={"strategy": "image_level", "seed": 42},
    )

    # --- Save comparison report
    report = {
        "total_images": n,
        "unique_patients": len(unique_patients),
        "unparseable_ids": len(unparseable),
        "patient_splits": {
            "counts": {k: len(v) for k, v in patient_splits.items()},
            "patient_overlap": patient_overlap,
        },
        "image_splits": {
            "counts": {k: len(v) for k, v in image_splits.items()},
            "patient_overlap": image_overlap,
        },
        "leakage_reduced_by": {
            "train_val": image_overlap["train_val_overlap"] - patient_overlap["train_val_overlap"],
            "train_test": image_overlap["train_test_overlap"] - patient_overlap["train_test_overlap"],
            "val_test": image_overlap["val_test_overlap"] - patient_overlap["val_test_overlap"],
        },
    }
    rep_path = REPORT_DIR / "split_comparison_report.json"
    with open(rep_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved -> {rep_path}")
    print(f"Splits saved -> {SPLITS_DIR}/")

    # Summary
    any_img_overlap = image_overlap["any_overlap"]
    any_pat_overlap = patient_overlap["any_overlap"]
    print(f"\n{'='*60}")
    print(f"Image-level strategy: {any_img_overlap} patients leak across splits")
    print(f"Patient-level strategy: {any_pat_overlap} patients leak across splits")
    print(f"Improvement: {any_img_overlap - any_pat_overlap} fewer overlapping patients")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
