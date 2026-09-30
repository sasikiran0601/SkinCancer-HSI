"""
Tests for Phase 1.1 (calibration formula) and Phase 3 (NaN/Inf handling).

Phase 1.1 — calibration formula PI = (RI - DI) / (WI - DI)
  - Exact match on constant inputs (hand-verifiable)
  - Output in sane reflectance range on synthetic data
  - Division by zero produces non-finite values (documenting the failure mode)

Phase 3.1 — inject Inf and confirm failure
Phase 3.2 — fix_non_finite resolves Inf/NaN via interpolation
"""

import numpy as np
import pytest
from hsi_preprocessing.calibration import (
    calibrate,
    fix_non_finite,
    validate_calibrated_cube,
    VALID_REFLECTANCE_MIN,
    VALID_REFLECTANCE_MAX,
)
from tests.fixtures.synthetic_data import (
    make_calibration_inputs,
    make_inf_calibration_case,
    make_cube_with_nans,
    BANDS_SMALL, H, W,
)


class TestCalibrationFormula:
    """Phase 1.1 — PI = (RI - DI) / (WI - DI)."""

    def test_exact_constant_inputs(self):
        """Hand-calculated: PI = (0.8 - 0.2) / (1.0 - 0.2) = 0.75 exactly."""
        ri, wi, di = make_calibration_inputs(ri_val=0.8, wi_val=1.0, di_val=0.2)
        pi = calibrate(ri, wi, di)
        expected = (0.8 - 0.2) / (1.0 - 0.2)  # = 0.75
        np.testing.assert_allclose(pi, expected, rtol=0,
                                   err_msg="Calibration formula mismatch on constant inputs")

    def test_zero_dark_image(self):
        """When DI=0: PI = RI / WI."""
        ri, wi, _ = make_calibration_inputs(ri_val=0.6, wi_val=1.0, di_val=0.0)
        di = np.zeros_like(ri)
        pi = calibrate(ri, wi, di)
        np.testing.assert_allclose(pi, 0.6, rtol=1e-12)

    def test_perfect_white_reference(self):
        """When RI == WI (perfectly white pixel): PI should be 1.0."""
        ri, wi, di = make_calibration_inputs(ri_val=1.0, wi_val=1.0, di_val=0.0)
        pi = calibrate(ri, wi, di)
        np.testing.assert_allclose(pi, 1.0, rtol=1e-12)

    def test_perfect_dark_image(self):
        """When RI == DI (pure dark pixel): PI should be 0.0."""
        ri, wi, di = make_calibration_inputs(ri_val=0.2, wi_val=1.0, di_val=0.2)
        pi = calibrate(ri, wi, di)
        np.testing.assert_allclose(pi, 0.0, rtol=1e-12)

    def test_output_shape_preserved(self):
        """Calibration should not change array shape."""
        ri, wi, di = make_calibration_inputs()
        pi = calibrate(ri, wi, di)
        assert pi.shape == ri.shape

    def test_1d_vector_input(self):
        """Calibration works on 1-D pixel vectors too."""
        ri = np.array([0.5])
        wi = np.array([1.0])
        di = np.array([0.0])
        pi = calibrate(ri, wi, di)
        np.testing.assert_allclose(pi, 0.5)

    def test_reflectance_range_on_synthetic(self):
        """On reasonable inputs, output should be in roughly [0, 1.5]."""
        rng = np.random.default_rng(99)
        ri = rng.uniform(0.1, 0.9, (BANDS_SMALL, H, W))
        di = rng.uniform(0.0, 0.1, (BANDS_SMALL, H, W))
        wi = rng.uniform(0.9, 1.0, (BANDS_SMALL, H, W))
        pi = calibrate(ri, wi, di)
        finite_pi = pi[np.isfinite(pi)]
        assert np.all(finite_pi >= VALID_REFLECTANCE_MIN), \
            f"Some values below {VALID_REFLECTANCE_MIN}: min={finite_pi.min()}"
        assert np.all(finite_pi <= VALID_REFLECTANCE_MAX), \
            f"Some values above {VALID_REFLECTANCE_MAX}: max={finite_pi.max()}"


class TestPhase31InfInjection:
    """Phase 3.1 — confirm that WI==DI produces Inf/NaN (red test)."""

    def test_division_by_zero_produces_nonfinite(self):
        """When WI == DI, calibration MUST produce non-finite values.

        This is a 'red' test that documents the failure mode exists
        before the fix is applied.
        """
        ri, wi, di = make_inf_calibration_case()
        # All wi == di, denominator = 0
        pi = calibrate(ri, wi, di)
        # We EXPECT non-finite values here — that's the point
        assert not np.all(np.isfinite(pi)), \
            "Expected non-finite values when WI==DI, but got all-finite output!"

    def test_partial_zero_denominator(self):
        """Inject zero denominator at a single band only."""
        ri = np.ones((BANDS_SMALL, H, W)) * 0.5
        wi = np.ones((BANDS_SMALL, H, W)) * 1.0
        di = np.zeros((BANDS_SMALL, H, W))
        # Make band 3 degenerate
        wi[3] = 0.0
        di[3] = 0.0
        pi = calibrate(ri, wi, di)
        # Bands 0-2, 4-9 should be finite
        for b in range(BANDS_SMALL):
            if b == 3:
                assert not np.all(np.isfinite(pi[b])), \
                    f"Band {b} should be non-finite"
            else:
                assert np.all(np.isfinite(pi[b])), \
                    f"Band {b} should be finite"


class TestPhase32NanInfFix:
    """Phase 3.2 — fix_non_finite resolves NaN/Inf correctly."""

    def test_no_nonfinite_values_unchanged(self):
        """If no NaN/Inf, cube should be returned unchanged."""
        # 10 bands × 5 × 5 = 250 elements (not 60 — use explicit shape)
        cube = np.arange(250, dtype=np.float64).reshape(BANDS_SMALL, H, W)
        fixed, n = fix_non_finite(cube)
        assert n == 0
        np.testing.assert_array_equal(fixed, cube)

    def test_nan_at_middle_band_interpolated(self):
        """NaN at a middle band should be interpolated from neighbours."""
        cube = np.zeros((BANDS_SMALL, H, W), dtype=np.float64)
        # Set all pixels to their band index for easy verification
        for b in range(BANDS_SMALL):
            cube[b] = float(b)
        # Inject NaN at band 5
        cube[5] = np.nan

        fixed, n = fix_non_finite(cube)
        assert n == H * W, f"Expected {H*W} fixed values, got {n}"
        # Band 5 should be linearly interpolated between band 4 (=4) and band 6 (=6) → 5.0
        np.testing.assert_allclose(fixed[5], 5.0, atol=1e-10,
                                   err_msg="Band 5 should interpolate to 5.0")

    def test_nan_at_first_band_uses_right_neighbour(self):
        """NaN at band 0 → should use band 1's value (no left neighbour)."""
        cube = np.zeros((BANDS_SMALL, H, W), dtype=np.float64)
        for b in range(BANDS_SMALL):
            cube[b] = float(b * 2)   # values: 0, 2, 4, ..., 18
        cube[0] = np.nan

        fixed, n = fix_non_finite(cube)
        # Only right neighbour (band 1 = 2.0) is available
        np.testing.assert_allclose(fixed[0], 2.0, atol=1e-10)

    def test_nan_at_last_band_uses_left_neighbour(self):
        """NaN at last band → should use second-to-last value."""
        cube = np.zeros((BANDS_SMALL, H, W), dtype=np.float64)
        for b in range(BANDS_SMALL):
            cube[b] = float(b)
        cube[-1] = np.nan

        fixed, n = fix_non_finite(cube)
        np.testing.assert_allclose(fixed[-1], float(BANDS_SMALL - 2), atol=1e-10)

    def test_inf_values_fixed(self):
        """Inf values (from divide-by-zero) are also repaired."""
        ri, wi, di = make_inf_calibration_case()
        pi = calibrate(ri, wi, di)
        assert not np.all(np.isfinite(pi))

        fixed, n = fix_non_finite(pi)
        assert n > 0, "Should have fixed some non-finite values"
        assert np.all(np.isfinite(fixed)), "All values should be finite after fix"

    def test_output_values_are_reasonable(self):
        """Fixed values should be close to their neighbouring bands' values."""
        cube = np.zeros((BANDS_SMALL, H, W), dtype=np.float64)
        for b in range(BANDS_SMALL):
            cube[b] = b * 0.1  # smooth gradient: 0, 0.1, ..., 0.9
        cube[4] = np.nan  # between 0.3 and 0.5 → expect ~0.4

        fixed, _ = fix_non_finite(cube)
        np.testing.assert_allclose(fixed[4], 0.4, atol=1e-10)

    def test_count_returned_correctly(self):
        """The returned count must equal the actual number of non-finite values."""
        cube = make_cube_with_nans(nan_band_indices=[1, 3, 7])
        n_expected = int(np.sum(~np.isfinite(cube)))
        _, n_returned = fix_non_finite(cube)
        assert n_returned == n_expected, \
            f"Count mismatch: expected {n_expected}, got {n_returned}"

    def test_all_nan_pixel_set_to_zero(self):
        """A pixel where all bands are NaN is set to 0 (documented fallback)."""
        cube = np.zeros((BANDS_SMALL, H, W), dtype=np.float64)
        for b in range(BANDS_SMALL):
            cube[b] = float(b)
        # Make pixel (0, 0) all NaN
        cube[:, 0, 0] = np.nan

        fixed, _ = fix_non_finite(cube)
        np.testing.assert_array_equal(fixed[:, 0, 0], 0.0,
                                      err_msg="All-NaN pixel should become all-zeros")


class TestValidateCalibratedCube:
    """Phase 1.4 — validate_calibrated_cube sanity checks."""

    def test_clean_cube_is_valid(self):
        """A clean cube in [0, 1] should pass validation."""
        cube = np.random.default_rng(7).uniform(0.0, 1.0, (BANDS_SMALL, H, W))
        result = validate_calibrated_cube(cube)
        assert result["valid"], f"Should be valid, got: {result['reason']}"
        assert result["n_nan"] == 0
        assert result["n_inf"] == 0

    def test_nan_cube_fails(self):
        """Cube with NaN → invalid."""
        cube = np.random.default_rng(7).uniform(0.0, 1.0, (BANDS_SMALL, H, W))
        cube[0, 0, 0] = np.nan
        result = validate_calibrated_cube(cube)
        assert not result["valid"]
        assert result["n_nan"] == 1

    def test_inf_cube_fails(self):
        """Cube with Inf → invalid."""
        cube = np.random.default_rng(7).uniform(0.0, 1.0, (BANDS_SMALL, H, W))
        cube[2, 2, 2] = np.inf
        result = validate_calibrated_cube(cube)
        assert not result["valid"]
        assert result["n_inf"] == 1

    def test_out_of_range_cube_fails(self):
        """Values outside [VALID_REFLECTANCE_MIN, VALID_REFLECTANCE_MAX] → flagged."""
        cube = np.full((BANDS_SMALL, H, W), 3.0, dtype=np.float64)  # way too high
        result = validate_calibrated_cube(cube)
        assert not result["valid"]
        assert result["n_out_of_range"] > 0
