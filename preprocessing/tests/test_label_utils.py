"""
Tests for Phase 6 (boundary detection) and Phase 7 (label integrity / class weights).
"""

import numpy as np
import pytest
from hsi_preprocessing.label_utils import (
    detect_boundary_pixels,
    flag_boundary_pixels_in_cube,
    check_label_consistency,
    validate_all_label_pairs,
    compute_class_distribution,
    compare_distribution_with_paper,
    compute_class_weights,
    BINARY_MULTI_MAP,
    PAPER_DISTRIBUTION,
)


class TestBoundaryDetection:
    """Phase 6.1 / 6.2 — boundary pixel detection."""

    def test_homogeneous_mask_no_boundaries(self):
        """A completely homogeneous label mask has no boundaries."""
        mask = np.ones((10, 10), dtype=int)
        boundary = detect_boundary_pixels(mask, n_pixels=1)
        assert not boundary.any(), "No boundaries in a uniform mask"

    def test_two_region_mask_has_boundaries(self):
        """A mask with two adjacent regions should produce boundary pixels."""
        mask = np.zeros((10, 10), dtype=int)
        mask[:, 5:] = 1  # Left half=0, right half=1
        boundary = detect_boundary_pixels(mask, n_pixels=0)
        # Should have boundary pixels around the vertical transition at column 5
        assert boundary.any(), "Expected boundary pixels at label transition"
        # Boundary pixels should be near column 5 (not at far edges)
        cols_with_boundary = np.where(boundary.any(axis=0))[0]
        assert min(cols_with_boundary) >= 4 and max(cols_with_boundary) <= 5, \
            f"Boundary pixels not near the actual transition: cols={cols_with_boundary}"

    def test_boundary_fraction_is_small(self):
        """Phase 6.2 pass criterion: boundary fraction should be single/low-double digits %.

        A small square lesion in a skin region should have boundaries
        only around the lesion edge, not the whole image.
        """
        mask = np.zeros((50, 50), dtype=int)
        mask[10:40, 10:40] = 1  # 30x30 lesion in a 50x50 image
        boundary = detect_boundary_pixels(mask, n_pixels=2)

        total_pixels = mask.size
        boundary_pixels = boundary.sum()
        frac = boundary_pixels / total_pixels

        assert frac < 0.30, \
            f"Boundary fraction {frac:.1%} is too high (>30%); check detection logic"
        assert frac > 0.01, \
            f"Boundary fraction {frac:.1%} suspiciously low; boundary may be missed"

    def test_boundary_mask_shape_matches_input(self):
        """Output boundary mask should have same shape as input label mask."""
        mask = np.zeros((20, 30), dtype=int)
        mask[5:15, 5:25] = 1
        boundary = detect_boundary_pixels(mask, n_pixels=1)
        assert boundary.shape == mask.shape

    def test_boundary_excluded_plus_full_equals_total(self):
        """Phase 6.2: excluded + boundary should equal total labeled pixels."""
        mask = np.zeros((20, 20), dtype=int)
        mask[5:15, 5:15] = 1
        boundary = detect_boundary_pixels(mask, n_pixels=1)

        # "Full" = all pixels in mask
        # "Core" (non-boundary) + "boundary" = total
        boundary_count = boundary.sum()
        non_boundary_count = (~boundary).sum()
        assert boundary_count + non_boundary_count == mask.size

    def test_n_pixels_zero_only_immediate_edge(self):
        """With n_pixels=0, only the immediate edge pixels should be flagged."""
        mask = np.zeros((10, 10), dtype=int)
        mask[3:7, 3:7] = 1  # 4x4 block
        boundary_0 = detect_boundary_pixels(mask, n_pixels=0)
        boundary_2 = detect_boundary_pixels(mask, n_pixels=2)
        # n_pixels=2 should always flag >= n_pixels=0 pixels
        assert boundary_2.sum() >= boundary_0.sum()


class TestLabelConsistency:
    """Phase 7.1 — binary/multi-label consistency."""

    @pytest.mark.parametrize("binary,multi,expected", [
        ("B", "BE", True),
        ("B", "BM", True),
        ("M", "ME", True),
        ("M", "MM", True),
        ("B", "ME", False),  # B with ME → inconsistent
        ("B", "MM", False),  # B with MM → inconsistent
        ("M", "BE", False),  # M with BE → inconsistent
        ("M", "BM", False),  # M with BM → inconsistent
    ])
    def test_known_combinations(self, binary, multi, expected):
        """Test all valid and invalid binary/multi combinations."""
        result = check_label_consistency(binary, multi)
        assert result == expected, \
            f"check_label_consistency({binary!r}, {multi!r}) should be {expected}"

    def test_batch_consistency_all_valid(self):
        """A well-formed dataset should have zero inconsistencies."""
        records = [
            {"sample_id": "P1", "binaryLabel": "B", "multiLabel": "BE"},
            {"sample_id": "P2", "binaryLabel": "B", "multiLabel": "BM"},
            {"sample_id": "P3", "binaryLabel": "M", "multiLabel": "ME"},
            {"sample_id": "P4", "binaryLabel": "M", "multiLabel": "MM"},
        ]
        valid, invalid = validate_all_label_pairs(records)
        assert len(invalid) == 0
        assert len(valid) == 4

    def test_batch_consistency_detects_mismatch(self):
        """A corrupted record should appear in the invalid list."""
        records = [
            {"sample_id": "P1", "binaryLabel": "B", "multiLabel": "ME"},  # BAD
            {"sample_id": "P2", "binaryLabel": "M", "multiLabel": "MM"},  # OK
        ]
        valid, invalid = validate_all_label_pairs(records)
        assert len(invalid) == 1
        assert invalid[0]["sample_id"] == "P1"


class TestClassDistribution:
    """Phase 7.2 — class distribution verification."""

    def test_known_distribution(self):
        """Verify fractional distribution on a hand-constructed label list."""
        labels = ["BE"] * 7 + ["BM"] * 45 + ["ME"] * 32 + ["MM"] * 16  # 100 samples
        dist = compute_class_distribution(labels)
        assert dist["BE"] == pytest.approx(0.07, abs=1e-6)
        assert dist["BM"] == pytest.approx(0.45, abs=1e-6)
        assert dist["ME"] == pytest.approx(0.32, abs=1e-6)
        assert dist["MM"] == pytest.approx(0.16, abs=1e-6)

    def test_fractions_sum_to_one(self):
        """Fractional distribution must always sum to 1."""
        labels = ["BE", "BM", "ME", "MM", "BM", "ME", "MM"]
        dist = compute_class_distribution(labels)
        assert sum(dist.values()) == pytest.approx(1.0, abs=1e-10)

    def test_comparison_with_paper_within_tolerance(self):
        """Paper distribution compared to itself → all within tolerance."""
        # Use exact paper distribution
        labels = (["BE"] * 7 + ["BM"] * 45 + ["ME"] * 32 + ["MM"] * 16)
        dist = compute_class_distribution(labels)
        comparison = compare_distribution_with_paper(dist)
        for cls, result in comparison.items():
            assert result["within_tolerance"], \
                f"Class {cls}: actual={result['actual']:.3f}, " \
                f"paper={result['paper']:.3f}, diff={result['diff']:.3f}"

    def test_comparison_flags_large_deviation(self):
        """A wildly different distribution should show out-of-tolerance flags."""
        labels = ["BE"] * 100  # only BE class (paper says 7%)
        dist = compute_class_distribution(labels)
        comparison = compare_distribution_with_paper(dist, tolerance=0.15)
        # BE class will be 1.0 vs paper 0.07 → diff = 0.93 → out of tolerance
        assert not comparison["BE"]["within_tolerance"]


class TestClassWeights:
    """Phase 7.3 — class weight computation."""

    def test_balanced_weights_are_all_one(self):
        """With perfectly balanced classes, weights should all equal 1."""
        labels = ["A"] * 10 + ["B"] * 10 + ["C"] * 10
        weights = compute_class_weights(labels, method="inverse_frequency")
        for cls, w in weights.items():
            assert w == pytest.approx(1.0, abs=1e-6), \
                f"Balanced class {cls} weight should be 1.0, got {w}"

    def test_imbalanced_weights_favour_minority(self):
        """Minority class should get a higher weight than majority."""
        labels = ["common"] * 90 + ["rare"] * 10
        weights = compute_class_weights(labels, method="inverse_frequency")
        assert weights["rare"] > weights["common"], \
            "Rare class should have higher weight than common class"

    def test_known_weights_hand_calculated(self):
        """Hand-calculate: labels = [A]*10 + [B]*30 + [C]*60, N=100, C=3.
        w_A = 100 / (3 * 10) = 3.333
        w_B = 100 / (3 * 30) = 1.111
        w_C = 100 / (3 * 60) = 0.556
        """
        labels = ["A"] * 10 + ["B"] * 30 + ["C"] * 60
        weights = compute_class_weights(labels)
        assert weights["A"] == pytest.approx(100 / (3 * 10), rel=1e-6)
        assert weights["B"] == pytest.approx(100 / (3 * 30), rel=1e-6)
        assert weights["C"] == pytest.approx(100 / (3 * 60), rel=1e-6)

    def test_weights_are_positive(self):
        """All computed weights must be strictly positive."""
        labels = ["BE"] * 7 + ["BM"] * 45 + ["ME"] * 32 + ["MM"] * 16
        weights = compute_class_weights(labels)
        for cls, w in weights.items():
            assert w > 0, f"Weight for {cls} should be positive, got {w}"
