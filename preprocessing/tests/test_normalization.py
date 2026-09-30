"""
Tests for Phase 1.5 (per-pixel normalization) and Phase 5.1 (per-band normalization).
"""

import numpy as np
import pytest
from hsi_preprocessing.normalization import (
    normalize_per_pixel,
    normalize_pixel_vector,
    compute_global_band_stats,
    normalize_per_band,
)
from tests.fixtures.synthetic_data import BANDS_SMALL, H, W, PIXEL_MONO, PIXEL_CONST


class TestPerPixelNormalization:
    """Phase 1.5 — per-pixel min-max normalization."""

    def test_output_min_is_zero(self):
        """Normalized pixel vectors must have min = 0.0 exactly."""
        cube = np.random.default_rng(1).uniform(0.5, 2.0, (BANDS_SMALL, H, W))
        normalized = normalize_per_pixel(cube)
        # Min over band axis for each pixel
        pixel_mins = normalized.min(axis=0)
        np.testing.assert_allclose(pixel_mins, 0.0, atol=1e-14,
                                   err_msg="Per-pixel min should be exactly 0")

    def test_output_max_is_one(self):
        """Normalized pixel vectors must have max = 1.0 exactly."""
        cube = np.random.default_rng(2).uniform(0.5, 2.0, (BANDS_SMALL, H, W))
        normalized = normalize_per_pixel(cube)
        pixel_maxs = normalized.max(axis=0)
        np.testing.assert_allclose(pixel_maxs, 1.0, atol=1e-14,
                                   err_msg="Per-pixel max should be exactly 1")

    def test_monotonic_transform_on_pixel_vector(self):
        """A monotonically increasing pixel vector should remain monotonically
        increasing after normalization (it's just a linear rescaling)."""
        normalized = normalize_pixel_vector(PIXEL_MONO)
        diffs = np.diff(normalized)
        assert np.all(diffs > 0), "Ordering must be preserved after normalization"

    def test_constant_pixel_becomes_zeros(self):
        """A constant pixel (x_max == x_min) must be mapped to all-zeros."""
        normalized = normalize_pixel_vector(PIXEL_CONST)
        np.testing.assert_array_equal(normalized, 0.0,
                                      err_msg="Constant pixel should map to all-zeros")

    def test_constant_pixel_no_division_by_zero(self):
        """Normalization of a constant pixel must not produce NaN."""
        cube = np.full((BANDS_SMALL, H, W), 7.777)
        normalized = normalize_per_pixel(cube)
        assert np.all(np.isfinite(normalized)), "No NaN/Inf for constant cube"

    def test_known_pixel_vector(self):
        """Hand-calculate normalization: [0, 1, 2, 3, 4] → [0, .25, .5, .75, 1]."""
        pixel = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        expected = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        normalized = normalize_pixel_vector(pixel)
        np.testing.assert_allclose(normalized, expected, atol=1e-14)

    def test_shape_preserved(self):
        """Normalization should not change cube shape."""
        cube = np.random.default_rng(3).uniform(0.0, 1.0, (BANDS_SMALL, H, W))
        normalized = normalize_per_pixel(cube)
        assert normalized.shape == cube.shape

    def test_dtype_is_float64(self):
        """Output should be float64 regardless of input dtype."""
        cube = np.random.default_rng(4).uniform(0.0, 1.0, (BANDS_SMALL, H, W)).astype(np.float32)
        normalized = normalize_per_pixel(cube)
        assert normalized.dtype == np.float64

    def test_relative_ordering_preserved_across_cube(self):
        """If pixel A is always > pixel B band-wise before normalization,
        this ordering should hold after (since each pixel is normalized
        independently — actually this check is per-pixel only).
        Verify that a specific pixel's relative band ordering is preserved."""
        cube = np.random.default_rng(5).uniform(1.0, 5.0, (BANDS_SMALL, H, W))
        orig_order = np.argsort(cube[:, 0, 0])
        normalized = normalize_per_pixel(cube)
        new_order = np.argsort(normalized[:, 0, 0])
        np.testing.assert_array_equal(orig_order, new_order,
                                      err_msg="Band ranking within a pixel must be preserved")


class TestPerBandNormalization:
    """Phase 5.1 — per-band global normalization (ablation)."""

    def test_global_stats_computed_correctly(self):
        """global_min and global_max should span the minimum and maximum
        across ALL cubes for each band."""
        cubes = [
            np.array([[[1.0, 3.0]], [[5.0, 7.0]]]),   # shape (2, 1, 2)
            np.array([[[2.0, 0.0]], [[6.0, 8.0]]]),   # shape (2, 1, 2)
        ]
        gmin, gmax = compute_global_band_stats(cubes)
        # Band 0: values are [1,3,2,0] → min=0, max=3
        assert gmin[0] == pytest.approx(0.0)
        assert gmax[0] == pytest.approx(3.0)
        # Band 1: values are [5,7,6,8] → min=5, max=8
        assert gmin[1] == pytest.approx(5.0)
        assert gmax[1] == pytest.approx(8.0)

    def test_per_band_normalization_output_range(self):
        """After per-band normalization, all values should be in [0, 1]."""
        cubes = [np.random.default_rng(i).uniform(0.0, 1.0, (BANDS_SMALL, H, W))
                 for i in range(5)]
        gmin, gmax = compute_global_band_stats(cubes)

        for cube in cubes:
            normalized = normalize_per_band(cube, gmin, gmax)
            assert normalized.min() >= -1e-12, f"Min below 0: {normalized.min()}"
            assert normalized.max() <= 1 + 1e-12, f"Max above 1: {normalized.max()}"

    def test_per_band_differs_from_per_pixel(self):
        """Per-band normalization must produce DIFFERENT output than per-pixel
        on non-uniform cubes (proves they are genuinely distinct methods)."""
        # Create a cube where per-pixel and per-band normalization diverge
        cube = np.zeros((BANDS_SMALL, H, W), dtype=np.float64)
        for b in range(BANDS_SMALL):
            cube[b] = float(b)  # each band is constant spatially → per-pixel is all-zeros or error

        gmin = np.zeros(BANDS_SMALL)
        gmax = np.arange(BANDS_SMALL, dtype=np.float64)
        gmax[0] = 1.0  # avoid degenerate band 0

        per_pixel_result = normalize_per_pixel(cube)
        per_band_result = normalize_per_band(cube, gmin, gmax)

        # They should differ (not be identical)
        assert not np.allclose(per_pixel_result, per_band_result), \
            "per-pixel and per-band normalizations should diverge on this input"

    def test_global_min_band_is_zero_after_norm(self):
        """A pixel at the global minimum of a band should normalize to 0."""
        cubes = [
            np.array([[[0.0, 0.5]], [[0.5, 1.0]]]),  # shape (2, 1, 2)
        ]
        gmin, gmax = compute_global_band_stats(cubes)
        normalized = normalize_per_band(cubes[0], gmin, gmax)
        # First band, first pixel (value=0) → should be 0
        assert normalized[0, 0, 0] == pytest.approx(0.0)

    def test_global_max_band_is_one_after_norm(self):
        """A pixel at the global maximum of a band should normalize to 1."""
        cubes = [
            np.array([[[0.0, 0.5]], [[0.5, 1.0]]]),  # shape (2, 1, 2)
        ]
        gmin, gmax = compute_global_band_stats(cubes)
        normalized = normalize_per_band(cubes[0], gmin, gmax)
        # Second band, second pixel (value=1.0) → should be 1
        assert normalized[1, 0, 1] == pytest.approx(1.0)

    def test_degenerate_band_no_division_by_zero(self):
        """A band with zero global range (all-same values) → output should be 0."""
        cube = np.ones((BANDS_SMALL, H, W), dtype=np.float64)
        gmin = np.zeros(BANDS_SMALL)
        gmax = np.zeros(BANDS_SMALL)  # zero range for all bands
        normalized = normalize_per_band(cube, gmin, gmax)
        assert np.all(np.isfinite(normalized)), "No NaN/Inf for degenerate bands"
        np.testing.assert_array_equal(normalized, 0.0)

    def test_per_band_shape_preserved(self):
        """Per-band normalization must not change array shape."""
        cubes = [np.random.default_rng(i).uniform(0, 1, (BANDS_SMALL, H, W)) for i in range(3)]
        gmin, gmax = compute_global_band_stats(cubes)
        normalized = normalize_per_band(cubes[0], gmin, gmax)
        assert normalized.shape == cubes[0].shape
