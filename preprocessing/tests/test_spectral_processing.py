"""
Tests for Phase 1.2 (band trimming) and Phase 1.3 (moving average filter).
"""

import numpy as np
import pytest
from hsi_preprocessing.spectral_processing import (
    trim_bands,
    moving_average_filter,
    moving_average_1d,
    TRIM_DROP_FIRST,
    TRIM_DROP_LAST,
    EXPECTED_BANDS_BEFORE_TRIM,
    EXPECTED_BANDS_AFTER_TRIM,
    MOVING_AVG_HALF_WINDOW,
)
from tests.fixtures.synthetic_data import (
    CUBE_125_BANDS,
    SIGNAL_1D,
    BANDS_SMALL,
    BANDS_FULL,
    BANDS_TRIMMED,
    H, W,
)


class TestBandTrimming:
    """Phase 1.2 — trim_bands."""

    def test_band_count_exact(self):
        """After trimming a 125-band cube → must have exactly 116 bands."""
        cube = CUBE_125_BANDS.copy()
        trimmed = trim_bands(cube)
        assert trimmed.shape[0] == EXPECTED_BANDS_AFTER_TRIM, \
            f"Expected {EXPECTED_BANDS_AFTER_TRIM} bands, got {trimmed.shape[0]}"

    def test_correct_bands_retained(self):
        """Bands [4:120] (indices 4 to 119 inclusive) should be kept.

        Original bands 0-3 (first 4) and 120-124 (last 5) should be removed.
        """
        # Create a cube where each band b has all-values = b (easy to track)
        cube = np.zeros((BANDS_FULL, H, W), dtype=np.float64)
        for b in range(BANDS_FULL):
            cube[b] = float(b)

        trimmed = trim_bands(cube)

        # First retained band should be original band 4 → value 4.0
        np.testing.assert_allclose(trimmed[0], 4.0, rtol=0,
                                   err_msg="First retained band should be original band 4")
        # Last retained band should be original band 119 → value 119.0
        np.testing.assert_allclose(trimmed[-1], 119.0, rtol=0,
                                   err_msg="Last retained band should be original band 119")

    def test_no_reversal(self):
        """The band ordering must be preserved (not reversed)."""
        cube = np.zeros((BANDS_FULL, H, W), dtype=np.float64)
        for b in range(BANDS_FULL):
            cube[b] = float(b)
        trimmed = trim_bands(cube)
        # Values should still be monotonically increasing along the band axis
        means = trimmed[:, 0, 0]
        assert np.all(np.diff(means) > 0), "Band ordering was changed!"

    def test_no_off_by_one(self):
        """Check exact start and end indices match spec."""
        cube = np.zeros((BANDS_FULL, H, W), dtype=np.float64)
        for b in range(BANDS_FULL):
            cube[b] = float(b)
        trimmed = trim_bands(cube)

        # Band 3 (dropped) should NOT be present (value 3.0)
        assert not np.any(trimmed[:, 0, 0] == 3.0), "Band 3 should have been dropped"
        # Band 120 (dropped) should NOT be present (value 120.0)
        assert not np.any(trimmed[:, 0, 0] == 120.0), "Band 120 should have been dropped"
        # Band 4 (kept) SHOULD be present
        assert np.any(trimmed[:, 0, 0] == 4.0), "Band 4 should be kept"
        # Band 119 (kept) SHOULD be present
        assert np.any(trimmed[:, 0, 0] == 119.0), "Band 119 should be kept"

    def test_spatial_dimensions_unchanged(self):
        """Trimming must not alter the spatial dimensions."""
        cube = CUBE_125_BANDS.copy()
        trimmed = trim_bands(cube)
        assert trimmed.shape[1] == H
        assert trimmed.shape[2] == W

    def test_custom_trim_parameters(self):
        """Custom drop_first / drop_last should work correctly."""
        cube = np.zeros((20, H, W), dtype=np.float64)
        for b in range(20):
            cube[b] = float(b)
        trimmed = trim_bands(cube, drop_first=3, drop_last=2)
        assert trimmed.shape[0] == 15, f"Expected 15 bands, got {trimmed.shape[0]}"
        np.testing.assert_allclose(trimmed[0], 3.0)  # first kept band = 3
        np.testing.assert_allclose(trimmed[-1], 17.0)  # last kept band = 17

    def test_no_drop(self):
        """With drop_first=0, drop_last=0 → all bands retained."""
        cube = CUBE_125_BANDS.copy()
        trimmed = trim_bands(cube, drop_first=0, drop_last=0)
        assert trimmed.shape[0] == BANDS_FULL

    def test_dtype_preserved(self):
        """Output dtype should remain float64."""
        cube = CUBE_125_BANDS.copy()
        trimmed = trim_bands(cube)
        assert trimmed.dtype == np.float64


class TestMovingAverageFilter:
    """Phase 1.3 — moving_average_filter and moving_average_1d."""

    def test_constant_signal_unchanged(self):
        """A constant signal should remain constant after smoothing."""
        signal = np.full(20, 5.0)
        filtered = moving_average_1d(signal)
        np.testing.assert_allclose(filtered, 5.0, atol=1e-12,
                                   err_msg="Constant signal should be unchanged")

    def test_midpoint_matches_manual_calculation(self):
        """Manually verify the filter at an interior point.

        Signal: [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        At i=5 (value=60), N=5 → window is [max(0,0):min(10,11)] = [0:10]
        But since the signal only has 10 elements, window = entire signal.
        mean = (10+20+30+40+50+60+70+80+90+100)/10 = 55.0
        """
        signal = SIGNAL_1D.copy()  # [10..100]
        # For a 10-element signal with N=5, all windows will include most of the signal
        # Let's test with a smaller signal where we can hand-calculate
        small_signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        # At i=3 (value=4), N=5 → window [max(0,-2):min(7,9)] = [0:7] = all 7 values
        # mean = (1+2+3+4+5+6+7)/7 = 4.0
        filtered = moving_average_1d(small_signal, half_window=5)
        np.testing.assert_allclose(filtered[3], 4.0, atol=1e-12,
                                   err_msg="Interior point should equal mean of full window")

    def test_narrow_window_specific_values(self):
        """Test with N=1 (3-band window) where we can hand-compute all values.

        Signal: [1, 2, 3, 4, 5]
        i=0: window [0:2] = [1,2] → mean=1.5
        i=1: window [0:3] = [1,2,3] → mean=2.0
        i=2: window [1:4] = [2,3,4] → mean=3.0
        i=3: window [2:5] = [3,4,5] → mean=4.0
        i=4: window [3:5] = [4,5] → mean=4.5
        """
        signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        expected = np.array([1.5, 2.0, 3.0, 4.0, 4.5])
        filtered = moving_average_1d(signal, half_window=1)
        np.testing.assert_allclose(filtered, expected, atol=1e-12,
                                   err_msg="N=1 filter values don't match hand calculation")

    def test_edge_behavior_no_nan_zeros(self):
        """Edge bands must not produce NaN or zero-padding artefacts."""
        cube = np.random.default_rng(42).uniform(0.0, 1.0, (BANDS_TRIMMED, H, W))
        filtered = moving_average_filter(cube)
        assert np.all(np.isfinite(filtered)), "No NaN/Inf at edges!"
        # First and last bands should not be zero (unless the input was zero)
        assert filtered[0].mean() > 0, "First band went to zero (wrong edge handling)"
        assert filtered[-1].mean() > 0, "Last band went to zero (wrong edge handling)"

    def test_output_shape_unchanged(self):
        """Filter should not change array shape."""
        cube = CUBE_125_BANDS.copy()
        filtered = moving_average_filter(cube)
        assert filtered.shape == cube.shape

    def test_independent_reference_comparison(self):
        """Cross-check our filter against scipy's uniform filter (independent impl).

        scipy.ndimage.uniform_filter1d with mode='reflect' differs at edges,
        but at interior points they should agree when the window fits.
        We verify on a long signal with N=1 at interior points only.
        """
        from scipy.ndimage import uniform_filter1d

        signal = np.random.default_rng(7).uniform(0.0, 1.0, 50)
        our_result = moving_average_1d(signal, half_window=1)

        # At interior points (away from first/last 1 element), compare with
        # a symmetric scipy filter (size=3)
        scipy_result = uniform_filter1d(signal.astype(float), size=3, mode='mirror')

        # Compare interior only (indices 1 to -2)
        np.testing.assert_allclose(
            our_result[1:-1], scipy_result[1:-1], atol=1e-10,
            err_msg="Interior filter values should match scipy reference"
        )

    def test_smoothing_reduces_variance(self):
        """A noisy signal should have lower variance after smoothing."""
        rng = np.random.default_rng(77)
        noisy = rng.uniform(0.0, 1.0, 50)
        smoothed = moving_average_1d(noisy, half_window=5)
        assert smoothed.var() < noisy.var(), \
            "Smoothed signal should have lower variance than noisy input"

    def test_cube_filter_matches_1d_filter(self):
        """Applying the cube filter per-band should match the 1D filter per-pixel."""
        cube = np.random.default_rng(13).uniform(0.0, 1.0, (20, 3, 3))
        filtered_cube = moving_average_filter(cube)

        # Verify one specific pixel (0, 0) matches 1D filter on its spectrum
        pixel = cube[:, 0, 0]
        filtered_pixel = moving_average_1d(pixel)
        np.testing.assert_allclose(
            filtered_cube[:, 0, 0], filtered_pixel, atol=1e-12,
            err_msg="Cube filter at pixel (0,0) should match 1D filter"
        )
