"""
Phase 1.1, 1.4, 3 — Calibration and NaN/Inf handling.

Formula (Leon et al. 2020, Eq. 1):
    PI = (RI - DI) / (WI - DI)

Where:
    RI = Raw Image reflectance
    DI = Dark Image (sensor dark current)
    WI = White reference Image (Lambertian white panel)
    PI = Pseudo-reflectance (calibrated cube)

The dataset already ships ``calibratedHsCube`` pre-computed by the original
researchers.  This module:
  1. Implements the formula from scratch (for documentation + cross-check).
  2. Provides a ``validate_calibrated_cube`` helper that checks a pre-computed
     ``calibratedHsCube`` for sanity (no NaN/Inf, values in expected range).
  3. Provides a ``fix_non_finite`` helper (Phase 3) that interpolates NaN/Inf
     values from neighbouring spectral bands.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple


# ---------------------------------------------------------------------------
# 1.1  Calibration formula
# ---------------------------------------------------------------------------

def calibrate(
    ri: np.ndarray,
    wi: np.ndarray,
    di: np.ndarray,
) -> np.ndarray:
    """Apply the pseudo-reflectance calibration formula.

    PI = (RI - DI) / (WI - DI)

    Parameters
    ----------
    ri : np.ndarray, shape (B, H, W)  or  (B,)
        Raw image cube (or pixel vector).
    wi : np.ndarray, same shape as ri
        White reference (Lambertian panel).
    di : np.ndarray, same shape as ri
        Dark image (sensor dark current).

    Returns
    -------
    np.ndarray
        Calibrated pseudo-reflectance, same shape as inputs.

    Notes
    -----
    Divisions by zero (when ``WI == DI``) are left as ``NaN`` / ``Inf``
    and should be handled by :func:`fix_non_finite` immediately after.
    """
    ri = np.asarray(ri, dtype=np.float64)
    wi = np.asarray(wi, dtype=np.float64)
    di = np.asarray(di, dtype=np.float64)

    denom = wi - di
    # Suppress the divide-by-zero numpy warning; we handle non-finites below.
    with np.errstate(divide="ignore", invalid="ignore"):
        pi = (ri - di) / denom
    return pi


# ---------------------------------------------------------------------------
# 3.1 / 3.2  Non-finite value handling
# ---------------------------------------------------------------------------

def fix_non_finite(cube: np.ndarray, axis: int = 0) -> Tuple[np.ndarray, int]:
    """Replace NaN / Inf values with linear interpolation from adjacent bands.

    Strategy (spectral interpolation):
        For each pixel (H, W) location, scan along the band axis.
        Any band that is non-finite is replaced by the mean of its two nearest
        finite neighbours.  For edge bands with only one finite neighbour the
        single neighbour value is used.  If *all* bands for a pixel are
        non-finite the pixel is set to 0 (last-resort fallback, logged as a
        warning).

    Parameters
    ----------
    cube : np.ndarray, shape (B, H, W)
        Input hyperspectral cube, may contain NaN / Inf.
    axis : int
        Band axis (default 0 for shape (B, H, W)).

    Returns
    -------
    fixed_cube : np.ndarray
        Copy of ``cube`` with all non-finite values replaced.
    n_fixed : int
        Total number of non-finite values that were replaced.
    """
    cube = np.array(cube, dtype=np.float64)  # make a copy
    non_finite_mask = ~np.isfinite(cube)
    n_fixed = int(non_finite_mask.sum())

    if n_fixed == 0:
        return cube, 0

    # Move band axis to front for uniform indexing
    cube = np.moveaxis(cube, axis, 0)  # (B, ...)
    non_finite_mask = np.moveaxis(non_finite_mask, axis, 0)

    B = cube.shape[0]
    band_indices = np.arange(B)

    # Iterate over every spatial location that has at least one bad band
    bad_spatial = np.any(non_finite_mask, axis=0)  # shape (H, W) or (...,)
    bad_positions = np.argwhere(bad_spatial)  # list of (row, col) tuples

    for pos in bad_positions:
        idx = tuple(pos)
        pixel = cube[(slice(None),) + idx]          # shape (B,)
        bad_bands = np.where(non_finite_mask[(slice(None),) + idx])[0]
        good_bands = np.where(np.isfinite(pixel))[0]

        if len(good_bands) == 0:
            # Degenerate: no finite values at all for this pixel → zero
            cube[(slice(None),) + idx] = 0.0
            continue

        for b in bad_bands:
            # Find nearest finite neighbours on each side
            left = good_bands[good_bands < b]
            right = good_bands[good_bands > b]

            if len(left) > 0 and len(right) > 0:
                l, r = int(left[-1]), int(right[0])
                # Linear interpolation
                alpha = (b - l) / (r - l)
                val = pixel[l] * (1 - alpha) + pixel[r] * alpha
            elif len(left) > 0:
                val = pixel[int(left[-1])]
            else:
                val = pixel[int(right[0])]

            cube[(b,) + idx] = val

    # Move band axis back
    cube = np.moveaxis(cube, 0, axis)
    return cube, n_fixed


# ---------------------------------------------------------------------------
# 1.4  Validation of pre-computed calibratedHsCube
# ---------------------------------------------------------------------------

VALID_REFLECTANCE_MIN: float = -0.2   # allow slight undershoot
VALID_REFLECTANCE_MAX: float = 2.0    # allow some overshoot past 1.0


def validate_calibrated_cube(
    cube: np.ndarray,
    sample_id: str = "unknown",
    low: float = VALID_REFLECTANCE_MIN,
    high: float = VALID_REFLECTANCE_MAX,
) -> dict:
    """Sanity-check a ``calibratedHsCube`` array.

    Checks:
    - No NaN values
    - No Inf values
    - Values mostly within [low, high]

    Parameters
    ----------
    cube : np.ndarray, shape (B, H, W)
    sample_id : str
        Image identifier for reporting.
    low, high : float
        Expected reflectance range.

    Returns
    -------
    dict with keys:
        valid : bool
        n_nan : int
        n_inf : int
        n_out_of_range : int
        frac_out_of_range : float
        min_val, max_val : float
        reason : str   (empty if valid)
    """
    cube = np.asarray(cube, dtype=np.float64)
    n_nan = int(np.isnan(cube).sum())
    n_inf = int(np.isinf(cube).sum())
    finite_vals = cube[np.isfinite(cube)]
    n_out = int(((finite_vals < low) | (finite_vals > high)).sum())
    total = cube.size
    frac_out = n_out / total if total > 0 else 0.0

    reasons = []
    if n_nan > 0:
        reasons.append(f"{n_nan} NaN values")
    if n_inf > 0:
        reasons.append(f"{n_inf} Inf values")
    if n_out > 0:
        reasons.append(
            f"{n_out}/{total} values ({frac_out:.2%}) outside [{low}, {high}]"
        )

    valid = len(reasons) == 0
    return {
        "sample_id": sample_id,
        "valid": valid,
        "n_nan": n_nan,
        "n_inf": n_inf,
        "n_out_of_range": n_out,
        "frac_out_of_range": frac_out,
        "min_val": float(finite_vals.min()) if len(finite_vals) > 0 else float("nan"),
        "max_val": float(finite_vals.max()) if len(finite_vals) > 0 else float("nan"),
        "reason": "; ".join(reasons),
    }
