"""Prepare a calibrated 125-band HSI cube for the Raspberry Pi model."""

import argparse
from pathlib import Path

import numpy as np


def fix_non_finite(cube: np.ndarray) -> np.ndarray:
    """Interpolate invalid values along the spectral band axis."""
    bands, height, width = cube.shape
    flattened = cube.reshape(bands, -1)
    band_indices = np.arange(bands)

    for pixel_index in range(flattened.shape[1]):
        values = flattened[:, pixel_index]
        finite = np.isfinite(values)
        if finite.all():
            continue
        if not finite.any():
            flattened[:, pixel_index] = 0.0
            continue
        flattened[:, pixel_index] = np.interp(
            band_indices,
            band_indices[finite],
            values[finite],
        )

    return flattened.reshape(bands, height, width)


def preprocess(cube: np.ndarray) -> np.ndarray:
    """Match the training pipeline: repair, trim, smooth, and normalize."""
    cube = np.asarray(cube, dtype=np.float64)
    if cube.ndim != 3:
        raise ValueError(f"Expected a 3D cube, received shape {cube.shape}")
    if cube.shape[0] != 125:
        raise ValueError(
            f"Expected calibrated input shape (125, H, W), received {cube.shape}. "
            "An uncalibrated hsCube requires white/dark reference data first."
        )

    cube = fix_non_finite(cube)
    cube = cube[4:-5]

    smoothed = np.empty_like(cube)
    for band in range(cube.shape[0]):
        low = max(0, band - 5)
        high = min(cube.shape[0], band + 6)
        smoothed[band] = cube[low:high].mean(axis=0)

    minimum = smoothed.min(axis=0, keepdims=True)
    maximum = smoothed.max(axis=0, keepdims=True)
    denominator = maximum - minimum
    return np.divide(
        smoothed - minimum,
        denominator,
        out=np.zeros_like(smoothed),
        where=denominator != 0,
    ).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a calibrated 125-band HSI cube to model input."
    )
    parser.add_argument("--input", required=True, help="Raw calibrated .npy cube")
    parser.add_argument("--output", default="sample_processed.npy")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    processed = preprocess(np.load(input_path))
    np.save(args.output, processed)
    print(f"Saved {args.output}: shape={processed.shape}, dtype={processed.dtype}")


if __name__ == "__main__":
    main()
