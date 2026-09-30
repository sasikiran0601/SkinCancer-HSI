#!/usr/bin/env python
"""
Phase 2.4 — Patient-Stratified 5-Fold Cross-Validation Splits.

Generates 5-fold splits using StratifiedGroupKFold:
- Grouped by patient ID (strictly 0 patient leakage across folds)
- Stratified by class label (all 4 classes represented in train & val in every fold)
- Evaluates 100% of the dataset (all 76 images) out-of-fold

Saves:
  preprocessing/outputs/splits/kfold_splits.json
  preprocessing/outputs/reports/kfold_split_report.json
"""

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
PREPROC = Path(__file__).parent.parent
sys.path.insert(0, str(PREPROC))

from hsi_preprocessing.splitting import (
    generate_stratified_group_kfold_splits,
    assert_no_kfold_patient_leakage,
    save_kfold_splits,
    validate_patient_ids,
    parse_patient_id,
)

NPY_ROOT = ROOT / "extracted_dataset" / "npy_arrays"
SPLITS_DIR = PREPROC / "outputs" / "splits"
REPORT_DIR = PREPROC / "outputs" / "reports"
SPLITS_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def load_all_labels(npy_root: Path):
    sample_ids = sorted(d.name for d in npy_root.iterdir() if d.is_dir())
    labels = []
    for sid in sample_ids:
        label_file = npy_root / sid / f"{sid}__labels.json"
        with open(label_file, "r", encoding="utf-8") as f:
            labels.append(json.load(f)["multiLabel"])
    return sample_ids, labels


def main():
    print("=" * 60)
    print("Phase 2.4 — Generating Patient-Stratified 5-Fold CV Splits")
    print("=" * 60)

    sample_ids, labels = load_all_labels(NPY_ROOT)
    print(f"Loaded {len(sample_ids)} samples.")

    valid_map, unparseable = validate_patient_ids(sample_ids)
    assert len(unparseable) == 0, f"Found unparseable IDs: {unparseable}"
    print(f"Found {len(set(valid_map.values()))} unique patients.")

    kfold_splits = generate_stratified_group_kfold_splits(
        sample_ids=sample_ids,
        labels=labels,
        n_splits=5,
        seed=42,
    )

    # Verify zero leakage across all folds
    assert_no_kfold_patient_leakage(kfold_splits)
    print("[OK] Verified zero patient leakage across all 5 folds!\n")

    id_to_label = dict(zip(sample_ids, labels))
    fold_report = {}

    for fold_key, splits in sorted(kfold_splits.items()):
        train_sids = splits["train"]
        val_sids = splits["val"]
        train_pats = {parse_patient_id(s) for s in train_sids}
        val_pats = {parse_patient_id(s) for s in val_sids}
        train_class_counts = dict(Counter(id_to_label[s] for s in train_sids))
        val_class_counts = dict(Counter(id_to_label[s] for s in val_sids))

        print(f"--- {fold_key.upper()} ---")
        print(f"  Train: {len(train_sids)} images from {len(train_pats)} patients -> {train_class_counts}")
        print(f"  Val  : {len(val_sids)} images from {len(val_pats)} patients -> {val_class_counts}")

        fold_report[fold_key] = {
            "train_images": len(train_sids),
            "train_patients": len(train_pats),
            "train_distribution": train_class_counts,
            "val_images": len(val_sids),
            "val_patients": len(val_pats),
            "val_distribution": val_class_counts,
            "patient_overlap": len(train_pats & val_pats),
        }

    splits_file = SPLITS_DIR / "kfold_splits.json"
    save_kfold_splits(
        kfold_splits=kfold_splits,
        output_path=splits_file,
        config={"strategy": "StratifiedGroupKFold", "n_splits": 5, "seed": 42},
    )
    print(f"\nSaved 5-Fold splits -> {splits_file}")

    report_file = REPORT_DIR / "kfold_split_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(fold_report, f, indent=2)
    print(f"Saved 5-Fold report -> {report_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
