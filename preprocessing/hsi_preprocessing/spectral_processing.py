"""
Phase 1.2 — Band trimming.
Phase 1.3 — Moving average spectral filter.

Band trimming (Leon et al. 2020):
    Drop the first 4 and last 5 bands from the 125-band raw cube.
    125 - 4 - 5 = 116 bands retained.

Moving average filter (Eq. 2, Leon et al. 2020):
    y(i) = mean(y[i-N : i+N+1])
    with N = 5  (window half-width), so full window = 11 bands.
    Edge behaviour: shrink window at array edges (no zero-padding),
    equivalent to a variable-width average that always uses only valid bands.
"""

from __future__ import annotations

import numpy as np

# Paper constants
TRIM_DROP_FIRST: int = 4
TRIM_DROP_LAST: int = 5
EXPECTED_BANDS_BEFORE_TRIM: int = 125
EXPECTED_BANDS_AFTER_TRIM: int = 116   # 125 - 4 - 5
MOVING_AVG_HALF_WINDOW: int = 5        # N in the paper's Eq. 2


# ---------------------------------------------------------------------------
# 1.2  Band trimming
# ---------------------------------------------------------------------------

def trim_bands(
    cube: np.ndarray,
    drop_first: int = TRIM_DROP_FIRST,
    drop_last: int = TRIM_DROP_LAST,
    band_axis: int = 0,
) -> np.ndarray:
    """Trim noisy edge bands from a hyperspectral cube.

    Parameters
    ----------
    cube : np.ndarray
        Input cube, default shape (B, H, W) with bands along ``band_axis``.
    drop_first : int
        Number of leading bands to drop (default 4, per paper).
    drop_last : int
        Number of trailing bands to drop (default 5, per paper).
    band_axis : int
        Which axis is the band axis (default 0).

    Returns
    -------
    np.ndarray
        Trimmed cube with ``drop_first + drop_last`` fewer bands.
    """
    cube = np.asarray(cube)
    n_bands = cube.shape[band_axis]
    end_idx = n_bands - drop_last if drop_last > 0 else n_bands
    slices = [slice(None)] * cube.ndim
    slices[band_axis] = slice(drop_first, end_idx)
    return cube[tuple(slices)]


# ---------------------------------------------------------------------------
# 1.3  Moving average filter
# ---------------------------------------------------------------------------

def moving_average_filter(
    cube: np.ndarray,
    half_window: int = MOVING_AVG_HALF_WINDOW,
    band_axis: int = 0,
) -> np.ndarray:
    """Apply per-pixel moving average smoothing along the spectral dimension.

    For each band i:
        y(i) = mean(y[max(0, i-N) : min(B, i+N+1)])

    Edge behaviour:
        Window shrinks at edges — no zero-padding, no boundary artefacts.
        This matches a uniform 1-D convolution with 'valid'-equivalent edges.

    Parameters
    ----------
    cube : np.ndarray, shape (B, H, W)
        Input cube (bands on ``band_axis``).
    half_window : int
        N in the paper's Eq. 2  (default 5 → 11-band full window).
    band_axis : int
        Band axis index (default 0).

    Returns
    -------
    np.ndarray
        Smoothed cube, same shape as input, dtype float64.
    """
    cube = np.asarray(cube, dtype=np.float64)

    # Move band axis to position 0 for uniform indexing
    cube = np.moveaxis(cube, band_axis, 0)  # → (B, ...)
    B = cube.shape[0]
    result = np.empty_like(cube)

    for i in range(B):
        lo = max(0, i - half_window)
        hi = min(B, i + half_window + 1)
        result[i] = cube[lo:hi].mean(axis=0)

    # Move band axis back to original position
    result = np.moveaxis(result, 0, band_axis)
    return result


def moving_average_1d(
    signal: np.ndarray,
    half_window: int = MOVING_AVG_HALF_WINDOW,
) -> np.ndarray:
    """Moving average on a 1-D spectral signal (convenience wrapper).

    Parameters
    ----------
    signal : np.ndarray, shape (B,)
    half_window : int

    Returns
    -------
    np.ndarray, shape (B,)
    """
    signal = np.asarray(signal, dtype=np.float64)
    B = len(signal)
    result = np.empty(B, dtype=np.float64)
    for i in range(B):
        lo = max(0, i - half_window)
        hi = min(B, i + half_window + 1)
        result[i] = signal[lo:hi].mean()
    return result
