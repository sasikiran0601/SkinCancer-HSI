"""
Phase 2 — Patient-independent splitting.

Responsibilities:
  - Parse patient ID from filenames like ``P{n}_C{m}``  (Phase 2.1)
  - Generate train/val/test splits keyed by PATIENT ID, not image ID  (Phase 2.2)
  - Stratify splits by class label where possible
  - Provide a leakage assertion (Phase 2.2)
  - Compare image-level vs. patient-level splitting  (Phase 2.3)

Filename format observed in the dataset:
    P<patient_id>_C<capture_id>
    e.g. P13_C1000, P86_C4000

Patients with multiple captures: P13 (3 images), P27 (4), P29 (3), P60 (3),
P86 (4), P15 (2), P17 (2), P25 (2).
"""

from __future__ import annotations

import re
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# 2.1  Patient ID extraction
# ---------------------------------------------------------------------------

_PATIENT_RE = re.compile(r"^P(\d+)_C\d+$", re.IGNORECASE)


def parse_patient_id(sample_id: str) -> Optional[int]:
    """Extract integer patient ID from a sample identifier like ``P13_C1000``.

    Parameters
    ----------
    sample_id : str
        Filename stem, e.g. ``P13_C1000``.

    Returns
    -------
    int or None
        Patient ID (e.g. 13) or ``None`` if the string doesn't match.
    """
    m = _PATIENT_RE.match(sample_id.strip())
    if m is None:
        return None
    return int(m.group(1))


def extract_all_patient_ids(
    sample_ids: List[str],
) -> Dict[str, Optional[int]]:
    """Map every sample ID to its patient ID.

    Parameters
    ----------
    sample_ids : list of str

    Returns
    -------
    dict mapping sample_id → patient_id (int or None for unparseable IDs).
    """
    return {sid: parse_patient_id(sid) for sid in sample_ids}


def validate_patient_ids(
    sample_ids: List[str],
) -> Tuple[Dict[str, int], List[str]]:
    """Validate and return the patient ID mapping.

    Parameters
    ----------
    sample_ids : list of str

    Returns
    -------
    valid_map : dict mapping sample_id → patient_id  (only parseable ones)
    unparseable : list of sample_ids that failed to parse
    """
    all_map = extract_all_patient_ids(sample_ids)
    valid_map = {k: v for k, v in all_map.items() if v is not None}
    unparseable = [k for k, v in all_map.items() if v is None]
    return valid_map, unparseable


# ---------------------------------------------------------------------------
# 2.2  Patient-independent split generation
# ---------------------------------------------------------------------------

def generate_patient_splits(
    sample_ids: List[str],
    labels: List[str],
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    # test_frac is implicitly 1 - train_frac - val_frac
    seed: int = 42,
) -> Dict[str, List[str]]:
    """Generate train/val/test splits keyed by **patient**, not image.

    Algorithm:
    1. Group samples by patient ID.
    2. Determine each patient's "primary" class (most common label among their
       images — used for stratified assignment).
    3. Shuffle patients within each class stratum, then assign patients to
       train/val/test so the fractions are as close as possible.
    4. Return the final splits as lists of *sample* IDs.

    Parameters
    ----------
    sample_ids : list of str
        All image/sample identifiers.
    labels : list of str
        Corresponding multiLabel strings (e.g. 'BE', 'BM', 'ME', 'MM').
        Must be parallel to ``sample_ids``.
    train_frac, val_frac : float
        Approximate fractional sizes of train and validation sets.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict with keys 'train', 'val', 'test', each a list of sample IDs.
    """
    assert len(sample_ids) == len(labels), "sample_ids and labels must have same length."
    rng = np.random.default_rng(seed)

    # --- Map sample → patient
    valid_map, unparseable = validate_patient_ids(sample_ids)
    if unparseable:
        raise ValueError(
            f"Cannot parse patient ID for {len(unparseable)} samples: {unparseable[:5]}"
        )

    # --- Group by patient
    patient_to_samples: Dict[int, List[str]] = {}
    patient_to_labels: Dict[int, List[str]] = {}
    label_map = dict(zip(sample_ids, labels))

    for sid in sample_ids:
        pid = valid_map[sid]
        patient_to_samples.setdefault(pid, []).append(sid)
        patient_to_labels.setdefault(pid, []).append(label_map[sid])

    # --- Assign a representative label to each patient (most frequent)
    def majority_label(lbls: List[str]) -> str:
        from collections import Counter
        return Counter(lbls).most_common(1)[0][0]

    patient_label = {pid: majority_label(lbls)
                     for pid, lbls in patient_to_labels.items()}

    # --- Group patients by label for stratified split
    label_to_patients: Dict[str, List[int]] = {}
    for pid, lbl in patient_label.items():
        label_to_patients.setdefault(lbl, []).append(pid)

    # --- Assign each patient to a split, stratum by stratum
    patient_split: Dict[int, str] = {}
    for lbl, patients in label_to_patients.items():
        patients = list(rng.permutation(patients))
        n = len(patients)
        n_train = max(1, round(n * train_frac))
        n_val   = max(0, round(n * val_frac))
        n_test  = n - n_train - n_val

        # Guarantee at least 1 sample in every split when we have >= 3 patients.
        # For tiny strata (n < 3), use round-robin to avoid collapsing all to train.
        if n == 1:
            # Only one patient — put in train; val and test get nothing for this class.
            patient_split[patients[0]] = "train"
            continue
        elif n == 2:
            # Two patients — put one in train, one in test (skip val for this class).
            patient_split[patients[0]] = "train"
            patient_split[patients[1]] = "test"
            continue
        else:
            # n >= 3: standard proportional assignment, enforce at least 1 each
            if n_test < 1:
                n_val = max(0, n_val - 1)
                n_test = 1
            if n_val < 1 and n >= 3:
                n_train -= 1
                n_val = 1

        for i, pid in enumerate(patients):
            if i < n_train:
                patient_split[pid] = "train"
            elif i < n_train + n_val:
                patient_split[pid] = "val"
            else:
                patient_split[pid] = "test"

    # --- Map back to sample IDs
    splits: Dict[str, List[str]] = {"train": [], "val": [], "test": []}
    for pid, samples in patient_to_samples.items():
        split = patient_split[pid]
        splits[split].extend(samples)

    return splits


def assert_no_patient_leakage(
    splits: Dict[str, List[str]],
) -> None:
    """Assert zero patient overlap across train/val/test splits.

    Parameters
    ----------
    splits : dict with keys 'train', 'val', 'test'

    Raises
    ------
    AssertionError with a descriptive message if any patient appears in
    more than one split.
    """
    split_patient_sets: Dict[str, Set[int]] = {}
    for split_name, sids in splits.items():
        pids: Set[int] = set()
        for sid in sids:
            pid = parse_patient_id(sid)
            if pid is not None:
                pids.add(pid)
        split_patient_sets[split_name] = pids

    split_names = list(split_patient_sets.keys())
    for i in range(len(split_names)):
        for j in range(i + 1, len(split_names)):
            a, b = split_names[i], split_names[j]
            overlap = split_patient_sets[a] & split_patient_sets[b]
            assert len(overlap) == 0, (
                f"PATIENT LEAKAGE detected between '{a}' and '{b}'! "
                f"Overlapping patient IDs: {sorted(overlap)}"
            )


def generate_image_level_splits(
    sample_ids: List[str],
    labels: List[str],
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = 42,
) -> Dict[str, List[str]]:
    """Generate train/val/test splits at the *image* level (paper's method).

    This is the naive baseline — splits images randomly without respecting
    patient boundaries, potentially leaking the same patient into multiple sets.

    Used in Phase 2.3 to demonstrate the leakage that patient-level splitting fixes.
    """
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(sample_ids))
    n = len(indices)
    n_train = round(n * train_frac)
    n_val = round(n * val_frac)

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    sids = np.array(sample_ids)
    return {
        "train": sids[train_idx].tolist(),
        "val": sids[val_idx].tolist(),
        "test": sids[test_idx].tolist(),
    }


def count_patient_overlap(splits: Dict[str, List[str]]) -> Dict[str, int]:
    """Count how many patients appear in >1 split (leakage metric).

    Returns
    -------
    dict with keys 'train_val_overlap', 'train_test_overlap',
    'val_test_overlap', 'any_overlap'.
    """
    def patient_set(sids: List[str]) -> Set[int]:
        return {pid for sid in sids if (pid := parse_patient_id(sid)) is not None}

    train_p = patient_set(splits.get("train", []))
    val_p   = patient_set(splits.get("val", []))
    test_p  = patient_set(splits.get("test", []))

    tv = len(train_p & val_p)
    tt = len(train_p & test_p)
    vt = len(val_p & test_p)
    return {
        "train_val_overlap": tv,
        "train_test_overlap": tt,
        "val_test_overlap": vt,
        "any_overlap": tv + tt + vt,
    }


def save_splits(splits: Dict[str, List[str]], output_path: Path, config: dict) -> None:
    """Save splits to a JSON file."""
    output_path = Path(output_path)
    payload = {
        "config": config,
        "splits": splits,
        "counts": {k: len(v) for k, v in splits.items()},
    }
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)


def load_splits(path: Path) -> Dict[str, List[str]]:
    """Load splits from a JSON file."""
    with open(path) as f:
        return json.load(f)["splits"]


# ---------------------------------------------------------------------------
# 2.4  Patient-Stratified K-Fold Cross-Validation (StratifiedGroupKFold)
# ---------------------------------------------------------------------------

def generate_stratified_group_kfold_splits(
    sample_ids: List[str],
    labels: List[str],
    n_splits: int = 5,
    seed: int = 42,
) -> Dict[str, Dict[str, List[str]]]:
    """Generate Patient-Stratified K-Fold Cross-Validation splits.

    Guarantees:
    1. Zero patient leakage: Samples from the same patient always appear in the
       same fold (either all in train or all in val).
    2. Class stratification: Patients are distributed such that all classes
       are represented in both train and val for every fold.
    3. Every sample is validated exactly once out-of-fold.

    Parameters
    ----------
    sample_ids : list of str
        Identifiers for each sample (e.g. 'P13_C1000').
    labels : list of str
        Multi-class labels corresponding to each sample.
    n_splits : int, default=5
        Number of cross-validation folds.
    seed : int, default=42
        Random state for reproducibility.

    Returns
    -------
    dict of fold_name -> {'train': [sample_ids], 'val': [sample_ids]}
    """
    from sklearn.model_selection import StratifiedGroupKFold

    assert len(sample_ids) == len(labels), "sample_ids and labels must match in length."
    valid_map, unparseable = validate_patient_ids(sample_ids)
    if unparseable:
        raise ValueError(
            f"Cannot parse patient ID for {len(unparseable)} samples: {unparseable[:5]}"
        )

    groups = [valid_map[sid] for sid in sample_ids]
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    folds: Dict[str, Dict[str, List[str]]] = {}
    for fold_idx, (train_idx, val_idx) in enumerate(sgkf.split(sample_ids, labels, groups)):
        fold_key = f"fold_{fold_idx}"
        folds[fold_key] = {
            "train": [sample_ids[i] for i in train_idx],
            "val": [sample_ids[i] for i in val_idx],
        }

    return folds


def assert_no_kfold_patient_leakage(
    kfold_splits: Dict[str, Dict[str, List[str]]],
) -> None:
    """Verify zero patient overlap between train and val across all folds.

    Parameters
    ----------
    kfold_splits : dict of fold_key -> {'train': [...], 'val': [...]}

    Raises
    ------
    AssertionError if any patient appears in both train and val in any fold.
    """
    for fold_key, splits in kfold_splits.items():
        train_patients = {
            pid for sid in splits["train"] if (pid := parse_patient_id(sid)) is not None
        }
        val_patients = {
            pid for sid in splits["val"] if (pid := parse_patient_id(sid)) is not None
        }
        overlap = train_patients & val_patients
        assert len(overlap) == 0, (
            f"PATIENT LEAKAGE detected in {fold_key}! "
            f"Overlapping patient IDs: {sorted(overlap)}"
        )


def save_kfold_splits(
    kfold_splits: Dict[str, Dict[str, List[str]]],
    output_path: Path,
    config: Optional[dict] = None,
) -> None:
    """Save K-Fold splits to a JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": config or {},
        "n_splits": len(kfold_splits),
        "folds": kfold_splits,
        "fold_counts": {
            f_key: {s_key: len(s_list) for s_key, s_list in f_val.items()}
            for f_key, f_val in kfold_splits.items()
        },
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_kfold_splits(path: Path) -> Dict[str, Dict[str, List[str]]]:
    """Load K-Fold splits from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("folds", data)

