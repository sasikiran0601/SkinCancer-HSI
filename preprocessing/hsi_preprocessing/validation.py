"""
Phase 0.2 — Raw data structure schema validation.
Phase 8   — Cross-consistency check (calibratedHsCube vs spectralRGB).

Dataset schema (Leon et al. 2020 / extracted NPY format):
    calibratedHsCube : float64, shape (125, H, W)
    hsCube           : float64, shape (125, H, W)
    spectralRGB      : float64, shape (3, H, W)
    binaryLabel      : str in {'B', 'M'}
    multiLabel       : str in {'BE', 'BM', 'ME', 'MM'}

After our extraction step, data is stored as .npy and .json files per sample.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# 0.2  Schema definition
# ---------------------------------------------------------------------------

EXPECTED_BANDS: int = 125
EXPECTED_SPATIAL: int = 50
EXPECTED_RGB_CHANNELS: int = 3
VALID_BINARY_LABELS: set = {"B", "M"}
VALID_MULTI_LABELS: set = {"BE", "BM", "ME", "MM"}

SCHEMA: Dict[str, dict] = {
    "calibratedHsCube": {
        "shape": (EXPECTED_BANDS, EXPECTED_SPATIAL, EXPECTED_SPATIAL),
        "dtype_family": "float",
        "description": "Calibrated pseudo-reflectance cube (PI = (RI-DI)/(WI-DI))",
    },
    "hsCube": {
        "shape": (EXPECTED_BANDS, EXPECTED_SPATIAL, EXPECTED_SPATIAL),
        "dtype_family": "float",
        "description": "Raw (uncalibrated) hyperspectral cube",
    },
    "spectralRGB": {
        "shape": (EXPECTED_RGB_CHANNELS, EXPECTED_SPATIAL, EXPECTED_SPATIAL),
        "dtype_family": "float",
        "description": "RGB reconstruction from spectral data",
    },
    "binaryLabel": {
        "valid_values": VALID_BINARY_LABELS,
        "description": "Binary malignancy label: B (Benign) or M (Malignant)",
    },
    "multiLabel": {
        "valid_values": VALID_MULTI_LABELS,
        "description": "4-class label: BE, BM, ME, MM",
    },
}


def validate_mat_structure(
    sample_dir: Path,
    sample_id: str,
) -> Dict:
    """Validate the extracted NPY/JSON files for a single sample against the schema.

    Parameters
    ----------
    sample_dir : Path
        Directory containing ``{sample_id}__*.npy`` and ``{sample_id}__labels.json``.
    sample_id : str
        E.g. ``P13_C1000``.

    Returns
    -------
    dict with:
        valid : bool
        checks : list of individual check results
        reason : str (summary if invalid)
    """
    sample_dir = Path(sample_dir)
    checks = []
    reasons = []

    def add_check(name: str, passed: bool, detail: str = ""):
        checks.append({"check": name, "passed": passed, "detail": detail})
        if not passed:
            reasons.append(f"[{name}] {detail}")

    # --- Check calibratedHsCube
    for arr_key in ("calibratedHsCube", "hsCube", "spectralRGB"):
        npy_file = sample_dir / f"{sample_id}__{arr_key}.npy"
        if not npy_file.exists():
            add_check(f"{arr_key}_exists", False, f"File not found: {npy_file.name}")
            continue
        try:
            arr = np.load(str(npy_file))
            spec = SCHEMA[arr_key]
            expected_shape = spec["shape"]
            shape_ok = arr.shape == expected_shape
            add_check(
                f"{arr_key}_shape",
                shape_ok,
                f"got {arr.shape}, expected {expected_shape}" if not shape_ok else f"shape OK {arr.shape}",
            )
            dtype_ok = np.issubdtype(arr.dtype, np.floating)
            add_check(
                f"{arr_key}_dtype",
                dtype_ok,
                f"got {arr.dtype}" if not dtype_ok else f"dtype OK {arr.dtype}",
            )
        except Exception as e:
            add_check(f"{arr_key}_loadable", False, str(e))

    # --- Check labels JSON
    labels_file = sample_dir / f"{sample_id}__labels.json"
    if not labels_file.exists():
        add_check("labels_json_exists", False, f"File not found: {labels_file.name}")
    else:
        try:
            with open(labels_file) as f:
                labels = json.load(f)

            bin_ok = labels.get("binaryLabel") in VALID_BINARY_LABELS
            add_check(
                "binaryLabel_valid",
                bin_ok,
                f"got {labels.get('binaryLabel')!r}" if not bin_ok else f"OK ({labels['binaryLabel']})",
            )
            multi_ok = labels.get("multiLabel") in VALID_MULTI_LABELS
            add_check(
                "multiLabel_valid",
                multi_ok,
                f"got {labels.get('multiLabel')!r}" if not multi_ok else f"OK ({labels['multiLabel']})",
            )
        except Exception as e:
            add_check("labels_json_parseable", False, str(e))

    valid = all(c["passed"] for c in checks)
    return {
        "sample_id": sample_id,
        "valid": valid,
        "checks": checks,
        "reason": "; ".join(reasons),
    }


def validate_dataset(
    npy_root: Path,
) -> Tuple[List[Dict], List[Dict]]:
    """Run ``validate_mat_structure`` on every sample in the dataset.

    Parameters
    ----------
    npy_root : Path
        Root directory containing one sub-directory per sample.

    Returns
    -------
    valid_list : list of passing validation results
    invalid_list : list of failing validation results
    """
    npy_root = Path(npy_root)
    valid_list, invalid_list = [], []
    for sample_dir in sorted(npy_root.iterdir()):
        if not sample_dir.is_dir():
            continue
        sample_id = sample_dir.name
        result = validate_mat_structure(sample_dir, sample_id)
        (valid_list if result["valid"] else invalid_list).append(result)
    return valid_list, invalid_list


# ---------------------------------------------------------------------------
# 8  Cross-consistency: calibratedHsCube ↔ spectralRGB
# ---------------------------------------------------------------------------

# Approximate wavelength band indices for RGB reconstruction
# Wavelength range of the sensor: ~450 nm – 950 nm (125 bands, ~4 nm/band)
# After trim (bands 4..119), range is roughly 466–946 nm
# R ≈ 620-750 nm → bands ~42-75 in original 125-band space
# G ≈ 495-570 nm → bands ~11-30
# B ≈ 450-495 nm → bands ~0-11
# These are rough estimates — exact wavelengths stored in Wavelength.mat
RGB_APPROX_BAND_RANGES = {
    "R": (42, 75),
    "G": (11, 30),
    "B": (0, 11),
}


def derive_rgb_from_cube(
    calibrated_cube: np.ndarray,
    band_axis: int = 0,
    band_ranges: Optional[Dict] = None,
) -> np.ndarray:
    """Derive approximate RGB composite from a calibrated hyperspectral cube.

    Averages bands within each RGB wavelength range to produce a 3-channel
    (R, G, B) composite.

    Parameters
    ----------
    calibrated_cube : np.ndarray, shape (B, H, W) with bands on ``band_axis``.
    band_axis : int
    band_ranges : dict mapping 'R'/'G'/'B' → (start_band, end_band) indices.

    Returns
    -------
    np.ndarray, shape (3, H, W), approximate RGB.
    """
    if band_ranges is None:
        band_ranges = RGB_APPROX_BAND_RANGES

    cube = np.asarray(calibrated_cube, dtype=np.float64)
    cube = np.moveaxis(cube, band_axis, 0)  # (B, H, W)

    channels = []
    for ch in ("R", "G", "B"):
        lo, hi = band_ranges[ch]
        channels.append(cube[lo:hi].mean(axis=0))  # (H, W)

    rgb = np.stack(channels, axis=0)  # (3, H, W)
    return rgb


def compute_rgb_similarity(
    derived_rgb: np.ndarray,
    spectral_rgb: np.ndarray,
) -> Dict[str, float]:
    """Compute similarity metrics between derived and ground-truth RGB.

    Both arrays must have shape (3, H, W).

    Returns
    -------
    dict with:
        pearson_r : float  — Pearson correlation (flattened)
        cosine_sim : float
        mean_abs_diff : float
    """
    a = derived_rgb.ravel().astype(np.float64)
    b = spectral_rgb.ravel().astype(np.float64)

    # Pearson correlation
    if a.std() == 0 or b.std() == 0:
        pearson_r = 0.0
    else:
        pearson_r = float(np.corrcoef(a, b)[0, 1])

    # Cosine similarity
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    cosine_sim = float(np.dot(a, b) / (norm_a * norm_b)) if norm_a > 0 and norm_b > 0 else 0.0

    mean_abs_diff = float(np.abs(a - b).mean())

    return {
        "pearson_r": pearson_r,
        "cosine_sim": cosine_sim,
        "mean_abs_diff": mean_abs_diff,
    }
