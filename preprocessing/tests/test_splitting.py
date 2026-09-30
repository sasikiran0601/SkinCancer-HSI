"""
Tests for Phase 2 (patient-independent splitting).

2.1 — patient ID extraction and validation
2.2 — split generation with leakage assertion
2.3 — comparison of patient-level vs image-level splits
"""

import pytest
import numpy as np
from hsi_preprocessing.splitting import (
    parse_patient_id,
    extract_all_patient_ids,
    validate_patient_ids,
    generate_patient_splits,
    assert_no_patient_leakage,
    generate_image_level_splits,
    count_patient_overlap,
    generate_stratified_group_kfold_splits,
    assert_no_kfold_patient_leakage,
    save_kfold_splits,
    load_kfold_splits,
)


# ---------------------------------------------------------------------------
# Sample data fixtures for splitting tests
# ---------------------------------------------------------------------------

SAMPLE_IDS = [
    "P13_C1000", "P13_C2000", "P13_C3000",  # patient 13 (3 images)
    "P14_C1000",                              # patient 14 (1 image)
    "P15_C1000", "P15_C2000",                # patient 15 (2 images)
    "P16_C1000",                              # patient 16
    "P17_C1001", "P17_C2002",                # patient 17 (2 images)
    "P18_C1000",                              # patient 18
    "P21_C1000",                              # patient 21
    "P27_C1000", "P27_C2000",                # patient 27 (2 images)
    "P29_C1000", "P29_C2000", "P29_C3000",   # patient 29 (3 images)
    "P30_C1000",                              # patient 30
]

LABELS_MULTI = [
    "BM", "BM", "BM",   # P13
    "ME",                # P14
    "BE", "BE",          # P15
    "MM",                # P16
    "BM", "BM",          # P17
    "ME",                # P18
    "MM",                # P21
    "BM", "BM",          # P27
    "ME", "ME", "ME",    # P29
    "BE",                # P30
]

assert len(SAMPLE_IDS) == len(LABELS_MULTI)


class TestPatientIDExtraction:
    """Phase 2.1 — parse_patient_id."""

    def test_standard_format(self):
        """P13_C1000 → patient 13."""
        assert parse_patient_id("P13_C1000") == 13

    def test_large_patient_id(self):
        """P116_C1004 → patient 116."""
        assert parse_patient_id("P116_C1004") == 116

    def test_case_insensitive(self):
        """Parsing should work regardless of case."""
        assert parse_patient_id("p13_c1000") == 13

    def test_invalid_format_returns_none(self):
        """Non-matching strings should return None."""
        assert parse_patient_id("not_a_patient_id") is None
        assert parse_patient_id("13_C1000") is None
        assert parse_patient_id("PC1000") is None
        assert parse_patient_id("") is None

    def test_batch_extraction_all_valid(self):
        """All sample IDs in our fixture set should parse successfully."""
        result, unparseable = validate_patient_ids(SAMPLE_IDS)
        assert len(unparseable) == 0, f"Unparseable IDs: {unparseable}"
        assert len(result) == len(SAMPLE_IDS)

    def test_unique_patient_count(self):
        """The fixture set should have exactly the expected unique patient count."""
        result, _ = validate_patient_ids(SAMPLE_IDS)
        unique_patients = set(result.values())
        expected = {13, 14, 15, 16, 17, 18, 21, 27, 29, 30}
        assert unique_patients == expected, \
            f"Unique patients: {unique_patients}, expected: {expected}"

    def test_multi_image_patient_maps_same_id(self):
        """Multiple captures of same patient must all map to same patient ID."""
        result, _ = validate_patient_ids(["P13_C1000", "P13_C2000", "P13_C3000"])
        assert result["P13_C1000"] == result["P13_C2000"] == result["P13_C3000"] == 13


class TestPatientIndependentSplitting:
    """Phase 2.2 — generate_patient_splits + leakage assertion."""

    def test_all_samples_assigned(self):
        """Every sample ID must appear in exactly one split."""
        splits = generate_patient_splits(SAMPLE_IDS, LABELS_MULTI, seed=42)
        all_assigned = splits["train"] + splits["val"] + splits["test"]
        assert set(all_assigned) == set(SAMPLE_IDS), \
            "Some samples are missing from splits!"
        assert len(all_assigned) == len(SAMPLE_IDS), \
            "Duplicate samples in splits!"

    def test_no_patient_leakage(self):
        """Patient-independent splits must have zero patient overlap."""
        splits = generate_patient_splits(SAMPLE_IDS, LABELS_MULTI, seed=42)
        # This assertion must pass — if it raises, the test fails
        assert_no_patient_leakage(splits)

    def test_leakage_check_passes(self):
        """count_patient_overlap on patient splits must return 0 for all pairs."""
        splits = generate_patient_splits(SAMPLE_IDS, LABELS_MULTI, seed=42)
        overlap = count_patient_overlap(splits)
        assert overlap["any_overlap"] == 0, \
            f"Patient leakage detected: {overlap}"

    def test_train_is_largest_split(self):
        """Train set should always have more or equal patients than val and test."""
        splits = generate_patient_splits(SAMPLE_IDS, LABELS_MULTI, seed=42)
        train_p = {parse_patient_id(s) for s in splits["train"]}
        val_p = {parse_patient_id(s) for s in splits["val"]}
        test_p = {parse_patient_id(s) for s in splits["test"]}
        assert len(train_p) >= len(val_p), "Train patients should be >= val patients"
        assert len(train_p) >= len(test_p), "Train patients should be >= test patients"
        assert len(splits["train"]) >= len(splits["val"]), "Train samples should be >= val samples"

    def test_deterministic_with_same_seed(self):
        """Same seed → same splits every time."""
        s1 = generate_patient_splits(SAMPLE_IDS, LABELS_MULTI, seed=42)
        s2 = generate_patient_splits(SAMPLE_IDS, LABELS_MULTI, seed=42)
        assert s1["train"] == s2["train"]
        assert s1["val"] == s2["val"]
        assert s1["test"] == s2["test"]

    def test_different_seed_gives_different_splits(self):
        """Different seeds should generally give different splits."""
        s1 = generate_patient_splits(SAMPLE_IDS, LABELS_MULTI, seed=42)
        s2 = generate_patient_splits(SAMPLE_IDS, LABELS_MULTI, seed=99)
        # It's possible (but unlikely) they're identical by coincidence.
        # Use a dataset large enough that this is effectively impossible.
        assert s1["train"] != s2["train"] or s1["test"] != s2["test"], \
            "Different seeds gave identical splits (very unlikely)"


class TestImageLevelVsPatientLevelSplitting:
    """Phase 2.3 — demonstrate image-level split leaks patients."""

    def test_image_level_splits_leak_patients(self):
        """Image-level splits should show patient overlap (leakage > 0).

        With a dataset where multiple images share a patient (P13 has 3,
        P15 has 2, P27 has 2, P29 has 3), a random image-level split will
        almost certainly put the same patient in multiple sets.

        This test proves the PROBLEM exists (before our fix).
        """
        image_splits = generate_image_level_splits(SAMPLE_IDS, LABELS_MULTI, seed=42)
        overlap = count_patient_overlap(image_splits)
        # With this data, there WILL be leakage (multi-image patients are split)
        # Note: on rare seeds this MIGHT be 0 by chance, but seed=42 leaks
        # (we've verified this deterministically)
        assert overlap["any_overlap"] > 0, \
            "Expected patient leakage in image-level split, got 0 — verify test data"

    def test_patient_level_beats_image_level_on_leakage(self):
        """Patient-level split must have strictly less overlap than image-level."""
        patient_splits = generate_patient_splits(SAMPLE_IDS, LABELS_MULTI, seed=42)
        image_splits = generate_image_level_splits(SAMPLE_IDS, LABELS_MULTI, seed=42)

        patient_overlap = count_patient_overlap(patient_splits)["any_overlap"]
        image_overlap = count_patient_overlap(image_splits)["any_overlap"]

        assert patient_overlap == 0, \
            f"Patient-level split should have zero overlap, got {patient_overlap}"
        assert image_overlap > patient_overlap, \
            "Image-level split should have MORE overlap than patient-level"

    def test_both_splits_cover_same_samples(self):
        """Both strategies should assign the same total set of samples."""
        patient_splits = generate_patient_splits(SAMPLE_IDS, LABELS_MULTI, seed=42)
        image_splits = generate_image_level_splits(SAMPLE_IDS, LABELS_MULTI, seed=42)

        patient_all = set(patient_splits["train"] + patient_splits["val"] + patient_splits["test"])
        image_all = set(image_splits["train"] + image_splits["val"] + image_splits["test"])

        assert patient_all == image_all == set(SAMPLE_IDS)


class TestStratifiedGroupKFold:
    """Phase 2.4 — generate_stratified_group_kfold_splits."""

    def test_kfold_zero_leakage(self):
        """No patient should appear in both train and val of any fold."""
        folds = generate_stratified_group_kfold_splits(SAMPLE_IDS, LABELS_MULTI, n_splits=3, seed=42)
        assert len(folds) == 3
        # Should not raise AssertionError
        assert_no_kfold_patient_leakage(folds)

    def test_kfold_leakage_assertion_raises_on_leak(self):
        """Leakage assertion must raise when a patient appears in both train and val."""
        leaky_folds = {
            "fold_0": {
                "train": ["P13_C1000", "P14_C1000"],
                "val": ["P13_C2000", "P15_C1000"],  # P13 is in both train and val!
            }
        }
        with pytest.raises(AssertionError, match="PATIENT LEAKAGE detected in fold_0"):
            assert_no_kfold_patient_leakage(leaky_folds)

    def test_kfold_complete_out_of_fold_coverage(self):
        """Every sample must appear as validation in exactly one fold."""
        n_splits = 3
        folds = generate_stratified_group_kfold_splits(SAMPLE_IDS, LABELS_MULTI, n_splits=n_splits, seed=42)
        all_val_samples = []
        for fold in folds.values():
            all_val_samples.extend(fold["val"])
            assert len(set(fold["train"]) & set(fold["val"])) == 0
            assert set(fold["train"]) | set(fold["val"]) == set(SAMPLE_IDS)

        assert sorted(all_val_samples) == sorted(SAMPLE_IDS)
        assert len(all_val_samples) == len(SAMPLE_IDS)

    def test_save_and_load_kfold_splits(self, tmp_path):
        """Serialization roundtrip for kfold splits."""
        folds = generate_stratified_group_kfold_splits(SAMPLE_IDS, LABELS_MULTI, n_splits=3, seed=42)
        out_file = tmp_path / "kfold_splits.json"
        save_kfold_splits(folds, out_file, config={"test": True})
        loaded = load_kfold_splits(out_file)
        assert loaded == folds
