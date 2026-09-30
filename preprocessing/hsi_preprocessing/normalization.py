"""
Phase 1.5 — Per-pixel min-max normalization (paper baseline).
Phase 5.1 — Per-band (global) normalization (ablation alternative).

Per-pixel normalization (paper):
    For each pixel's spectral vector x:
        x' = (x - x_min) / (x_max - x_min)
    Degenerate case (x_max == x_min, constant pixel): output all-zeros.

Per-band normalization (ablation):
    For each band b, compute global min/max across the entire dataset.
        x'[b] = (x[b] - global_min[b]) / (global_max[b] - global_min[b])
    Degenerate bands (zero range) → output 0 for that band.
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# 1.5  Per-pixel min-max normalization
# ---------------------------------------------------------------------------

def normalize_per_pixel(
    cube: np.ndarray,
    band_axis: int = 0,
) -> np.ndarray:
    """Per-pixel min-max normalization: normalise each pixel's spectral vector
    independently to [0, 1].

    Parameters
    ----------
    cube : np.ndarray, shape (B, H, W)
        Hyperspectral cube with bands on ``band_axis``.
    band_axis : int
        Band axis (default 0).

    Returns
    -------
    np.ndarray
        Normalized cube, same shape, dtype float64.
        Constant pixels (all-same value) → all-zero output.
    """
    cube = np.asarray(cube, dtype=np.float64)

    # Move band axis to 0 for vectorized ops
    cube = np.moveaxis(cube, band_axis, 0)  # (B, H, W)

    px_min = cube.min(axis=0, keepdims=True)   # (1, H, W)
    px_max = cube.max(axis=0, keepdims=True)   # (1, H, W)
    denom = px_max - px_min                    # (1, H, W)

    # Avoid division by zero for constant pixels
    safe_denom = np.where(denom == 0, 1.0, denom)
    normalized = (cube - px_min) / safe_denom

    # Force constant pixels to 0 (not 1 or NaN)
    normalized = np.where(denom == 0, 0.0, normalized)

    # Move band axis back
    normalized = np.moveaxis(normalized, 0, band_axis)
    return normalized


def normalize_pixel_vector(
    pixel: np.ndarray,
) -> np.ndarray:
    """Normalize a single 1-D pixel spectral vector to [0, 1].

    Parameters
    ----------
    pixel : np.ndarray, shape (B,)

    Returns
    -------
    np.ndarray, shape (B,)
        Normalized vector.  Constant pixels → all-zeros.
    """
    pixel = np.asarray(pixel, dtype=np.float64)
    pmin, pmax = pixel.min(), pixel.max()
    if pmax == pmin:
        return np.zeros_like(pixel)
    return (pixel - pmin) / (pmax - pmin)


# ---------------------------------------------------------------------------
# 5.1  Per-band (global) normalization
# ---------------------------------------------------------------------------

def compute_global_band_stats(
    cubes: list[np.ndarray],
    band_axis: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-band global min/max across a collection of cubes.

    Parameters
    ----------
    cubes : list of np.ndarray, each shape (B, H, W)
        All cubes in the dataset.
    band_axis : int
        Band axis (default 0).

    Returns
    -------
    global_min : np.ndarray, shape (B,)
    global_max : np.ndarray, shape (B,)
    """
    # Stack all cubes → (N_cubes, B, H, W) then take min/max across
    # cubes × H × W for each band.
    all_cubes = [np.moveaxis(np.asarray(c, dtype=np.float64), band_axis, 0) for c in cubes]
    B = all_cubes[0].shape[0]

    global_min = np.full(B, np.inf)
    global_max = np.full(B, -np.inf)

    for c in all_cubes:
        # c shape: (B, H, W)
        c_min = c.reshape(B, -1).min(axis=1)
        c_max = c.reshape(B, -1).max(axis=1)
        global_min = np.minimum(global_min, c_min)
        global_max = np.maximum(global_max, c_max)

    return global_min, global_max


def normalize_per_band(
    cube: np.ndarray,
    global_min: np.ndarray,
    global_max: np.ndarray,
    band_axis: int = 0,
) -> np.ndarray:
    """Per-band global min-max normalization.

    Each band is normalized using the global min/max computed across the
    entire dataset (not per-pixel).

    Parameters
    ----------
    cube : np.ndarray, shape (B, H, W)
    global_min : np.ndarray, shape (B,)
    global_max : np.ndarray, shape (B,)
    band_axis : int

    Returns
    -------
    np.ndarray
        Normalized cube, same shape as input.
        Degenerate bands (zero range) → 0.
    """
    cube = np.asarray(cube, dtype=np.float64)
    cube = np.moveaxis(cube, band_axis, 0)  # (B, H, W)

    B = cube.shape[0]
    denom = global_max - global_min         # (B,)
    safe_denom = np.where(denom == 0, 1.0, denom)

    # Broadcast: (B,) → (B, 1, 1)
    gmin = global_min.reshape(B, *([1] * (cube.ndim - 1)))
    safe_d = safe_denom.reshape(B, *([1] * (cube.ndim - 1)))
    zero_band = (denom == 0).reshape(B, *([1] * (cube.ndim - 1)))

    normalized = (cube - gmin) / safe_d
    normalized = np.where(zero_band, 0.0, normalized)

    normalized = np.moveaxis(normalized, 0, band_axis)
    return normalized
