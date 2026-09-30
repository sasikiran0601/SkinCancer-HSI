"""
Phase 9 — Reproducibility: preprocessing config management.

Saves every preprocessing decision into a versioned JSON config file.
The pipeline can be reconstructed entirely from this config — no hardcoded
values anywhere else in the preprocessing code.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# Default configuration values (all overridable)
# ---------------------------------------------------------------------------

DEFAULT_CONFIG: Dict[str, Any] = {
    "version": "1.0.0",
    # 1.2  Band trimming
    "band_trim_drop_first": 4,
    "band_trim_drop_last": 5,
    "expected_bands_before_trim": 125,
    "expected_bands_after_trim": 116,
    # 1.3  Moving average
    "moving_avg_half_window": 5,       # N in Eq.2
    "moving_avg_full_window": 11,      # 2N+1
    # 1.5 / 5.1  Normalization
    "normalization_method": "per_pixel",  # "per_pixel" | "per_band"
    # 2  Splitting
    "split_train_frac": 0.70,
    "split_val_frac": 0.15,
    "split_seed": 42,
    "split_strategy": "patient_independent",  # "patient_independent" | "image_level"
    # 3  NaN/Inf handling
    "nan_inf_strategy": "spectral_interpolation",  # "spectral_interpolation" | "zero"
    # 4  Band noise profiling (data-driven trim validation)
    "band_noise_sample_size": 20,      # number of images to profile
    # 6  Boundary detection
    "boundary_n_pixels": 2,
    # 7  Class weights
    "class_weight_method": "inverse_frequency",
    # 8  RGB cross-consistency
    "rgb_similarity_threshold": 0.7,   # Pearson r threshold
    "rgb_r_band_range": [42, 75],
    "rgb_g_band_range": [11, 30],
    "rgb_b_band_range": [0, 11],
    # Paths (relative to project root)
    "npy_root": "extracted_dataset/npy_arrays",
    "output_processed": "preprocessing/outputs/processed",
    "output_splits": "preprocessing/outputs/splits",
    "output_reports": "preprocessing/outputs/reports",
    "output_plots": "preprocessing/outputs/plots",
}


def get_default_config() -> Dict[str, Any]:
    """Return a copy of the default configuration."""
    return dict(DEFAULT_CONFIG)


def save_config(config: Dict[str, Any], output_path: Path) -> None:
    """Save a preprocessing config to JSON.

    Adds a timestamp and Python/package version metadata.
    """
    output_path = Path(output_path)
    import numpy as np
    enriched = {
        "_meta": {
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "python_version": sys.version,
            "numpy_version": np.__version__,
        },
        **config,
    }
    with open(output_path, "w") as f:
        json.dump(enriched, f, indent=2)


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load a preprocessing config from JSON.

    Strips the ``_meta`` key so the returned dict contains only
    pipeline parameters.
    """
    with open(config_path) as f:
        raw = json.load(f)
    return {k: v for k, v in raw.items() if k != "_meta"}


def compute_array_checksum(arr_path: Path) -> str:
    """Compute a SHA-256 checksum of a .npy file for reproducibility checks."""
    h = hashlib.sha256()
    with open(arr_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_output_checksums(output_dir: Path) -> Dict[str, str]:
    """Compute SHA-256 checksums for every .npy file in ``output_dir``."""
    output_dir = Path(output_dir)
    checksums = {}
    for p in sorted(output_dir.rglob("*.npy")):
        rel = str(p.relative_to(output_dir))
        checksums[rel] = compute_array_checksum(p)
    return checksums


def save_checksums(checksums: Dict[str, str], output_path: Path) -> None:
    """Save checksum dict to JSON."""
    with open(output_path, "w") as f:
        json.dump(checksums, f, indent=2)


def compare_checksums(
    old: Dict[str, str],
    new: Dict[str, str],
) -> Dict[str, list]:
    """Compare two checksum dicts.

    Returns
    -------
    dict with:
        matching : list of keys with identical checksums
        changed  : list of keys with different checksums
        added    : list of keys only in new
        removed  : list of keys only in old
    """
    old_keys = set(old.keys())
    new_keys = set(new.keys())
    matching = [k for k in old_keys & new_keys if old[k] == new[k]]
    changed  = [k for k in old_keys & new_keys if old[k] != new[k]]
    added    = sorted(new_keys - old_keys)
    removed  = sorted(old_keys - new_keys)
    return {
        "matching": sorted(matching),
        "changed": sorted(changed),
        "added": added,
        "removed": removed,
    }
