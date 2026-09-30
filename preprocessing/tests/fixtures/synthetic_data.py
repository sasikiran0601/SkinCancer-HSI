"""
Phase 0.1 — Test harness setup.
Generates synthetic test fixtures used by all other test modules.

Fixture conventions:
  CUBE_5x5x10  : (10, 5, 5) float64 cube with known values
  CUBE_125_BANDS : (125, 5, 5) cube simulating a real image
  PIXEL_MONO   : monotonically increasing 10-band pixel vector
  SIGNAL_1D    : 1-D spectral signal for filter tests
"""

import numpy as np
import json
import pytest
from pathlib import Path
from typing import List, Optional


# ---------------------------------------------------------------------------
# Synthetic fixture data (deterministic, hand-verifiable)
# ---------------------------------------------------------------------------

BANDS_SMALL = 10
BANDS_FULL = 125
BANDS_TRIMMED = 116
H = W = 5  # spatial dimensions

RNG_SEED = 0
_rng = np.random.default_rng(RNG_SEED)

# Simple ascending pixel vector (values 0..9 → easy to reason about)
PIXEL_MONO = np.arange(BANDS_SMALL, dtype=np.float64)

# Cube with known pixel structure: each pixel = its (row, col) index × band
CUBE_5x5x10 = np.zeros((BANDS_SMALL, H, W), dtype=np.float64)
for _r in range(H):
    for _c in range(W):
        CUBE_5x5x10[:, _r, _c] = np.arange(BANDS_SMALL) * (_r + 1) + (_c + 1)

# A realistic 125-band cube (random but seeded)
CUBE_125_BANDS = _rng.uniform(0.0, 1.0, size=(BANDS_FULL, H, W))

# 1-D signal for moving average tests
SIGNAL_1D = np.array([10.0, 20.0, 30.0, 40.0, 50.0,
                       60.0, 70.0, 80.0, 90.0, 100.0])

# Constant pixel (degenerate normalization case)
PIXEL_CONST = np.full(BANDS_SMALL, 3.14)

# Pixel with one extreme outlier
PIXEL_OUTLIER = np.array([0.1, 0.2, 0.3, 100.0, 0.5,
                           0.6, 0.7, 0.8, 0.9, 1.0])


# ---------------------------------------------------------------------------
# Fixture factory helpers
# ---------------------------------------------------------------------------

def make_synthetic_cube(bands: int = BANDS_SMALL,
                        h: int = H, w: int = W,
                        seed: int = 1) -> np.ndarray:
    """Create a reproducible synthetic float64 cube."""
    return np.random.default_rng(seed).uniform(0.0, 1.0, (bands, h, w))


def make_calibration_inputs(
    h: int = H, w: int = W,
    ri_val: float = 0.8,
    wi_val: float = 1.0,
    di_val: float = 0.2,
    bands: int = BANDS_SMALL,
) -> tuple:
    """Return (RI, WI, DI) constant-value cubes for calibration unit tests."""
    shape = (bands, h, w)
    ri = np.full(shape, ri_val, dtype=np.float64)
    wi = np.full(shape, wi_val, dtype=np.float64)
    di = np.full(shape, di_val, dtype=np.float64)
    return ri, wi, di


def make_cube_with_nans(
    bands: int = BANDS_SMALL,
    h: int = H, w: int = W,
    nan_band_indices: Optional[List[int]] = None,
) -> np.ndarray:
    """Create a cube with NaN injected at specific bands."""
    cube = make_synthetic_cube(bands, h, w, seed=42)
    if nan_band_indices is None:
        nan_band_indices = [0, 5]
    for b in nan_band_indices:
        cube[b] = np.nan
    return cube


def make_inf_calibration_case() -> tuple:
    """Create (RI, WI=DI) to provoke division by zero → Inf/NaN.

    Returns (ri, wi, di) where wi == di for all pixels/bands.
    """
    ri = np.ones((BANDS_SMALL, H, W), dtype=np.float64) * 0.5
    wi = np.ones((BANDS_SMALL, H, W), dtype=np.float64) * 0.8
    di = wi.copy()   # <-- WI == DI → denominator = 0
    return ri, wi, di


# ---------------------------------------------------------------------------
# Pytest fixtures (available via conftest import)
# ---------------------------------------------------------------------------

@pytest.fixture
def cube_small():
    return CUBE_5x5x10.copy()


@pytest.fixture
def cube_full_bands():
    return CUBE_125_BANDS.copy()


@pytest.fixture
def pixel_mono():
    return PIXEL_MONO.copy()


@pytest.fixture
def pixel_const():
    return PIXEL_CONST.copy()


@pytest.fixture
def signal_1d():
    return SIGNAL_1D.copy()


@pytest.fixture
def calibration_inputs():
    return make_calibration_inputs()


@pytest.fixture
def cube_with_nans():
    return make_cube_with_nans()


@pytest.fixture
def inf_calibration_case():
    return make_inf_calibration_case()
