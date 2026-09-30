"""
Phase 6 — Boundary / spectral-mixing awareness.

For each labeled image, detect pixels within N pixels of a label transition
(lesion/skin boundary) using morphological dilation of the edge map.

Phase 7 — Label integrity checks.

  7.1  Binary/multi-label consistency
  7.2  Class distribution verification
  7.3  Class weight computation
"""

from __future__ import annotations

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import skimage.morphology as morph
from skimage.segmentation import find_boundaries


# ---------------------------------------------------------------------------
# 6.1 / 6.2  Boundary pixel detection
# ---------------------------------------------------------------------------

def detect_boundary_pixels(
    label_mask: np.ndarray,
    n_pixels: int = 2,
) -> np.ndarray:
    """Return a boolean mask of pixels within ``n_pixels`` of a label boundary.

    Uses scikit-image ``find_boundaries`` (inner+outer mode) followed by
    binary dilation with a disk of radius ``n_pixels``.

    Parameters
    ----------
    label_mask : np.ndarray, shape (H, W), integer label map.
        Each unique integer value represents a different tissue class.
        0 is treated as background/unlabeled if needed, but the function
        simply detects transitions between any two adjacent values.
    n_pixels : int
        Dilation radius around boundaries (default 2 → 2-pixel margin).

    Returns
    -------
    boundary_mask : np.ndarray, shape (H, W), bool
        True for boundary-adjacent pixels.
    """
    label_mask = np.asarray(label_mask)
    # Detect pixels that are on the boundary between any two different labels
    edge = find_boundaries(label_mask, mode="outer").astype(bool)
    if n_pixels > 0:
        selem = morph.disk(n_pixels)
        edge = morph.dilation(edge, selem).astype(bool)
    return edge


def flag_boundary_pixels_in_cube(
    cube: np.ndarray,
    label_mask: np.ndarray,
    n_pixels: int = 2,
    band_axis: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return cube and boundary mask for a single image.

    Parameters
    ----------
    cube : np.ndarray, shape (B, H, W)
    label_mask : np.ndarray, shape (H, W)
    n_pixels : int
    band_axis : int

    Returns
    -------
    cube : np.ndarray — unchanged (returned for pipeline chaining)
    boundary_mask : np.ndarray, shape (H, W), bool
    """
    boundary_mask = detect_boundary_pixels(label_mask, n_pixels=n_pixels)
    return cube, boundary_mask


# ---------------------------------------------------------------------------
# 7.1  Binary / multiLabel consistency check
# ---------------------------------------------------------------------------

# Expected mappings per the dataset's README:
#   B → {BE, BM}
#   M → {ME, MM}
BINARY_MULTI_MAP: Dict[str, set] = {
    "B": {"BE", "BM"},
    "M": {"ME", "MM"},
}


def check_label_consistency(
    binary_label: str,
    multi_label: str,
) -> bool:
    """Return True if binary and multi labels are mutually consistent."""
    expected = BINARY_MULTI_MAP.get(binary_label, set())
    return multi_label in expected


def validate_all_label_pairs(
    records: List[Dict],
) -> Tuple[List[Dict], List[Dict]]:
    """Check label consistency for a list of records.

    Parameters
    ----------
    records : list of dict, each with keys 'sample_id', 'binaryLabel', 'multiLabel'.

    Returns
    -------
    valid : list of records that passed consistency check
    invalid : list of records that failed (potential data-quality finding)
    """
    valid, invalid = [], []
    for r in records:
        if check_label_consistency(r["binaryLabel"], r["multiLabel"]):
            valid.append(r)
        else:
            invalid.append(r)
    return valid, invalid


# ---------------------------------------------------------------------------
# 7.2  Class distribution
# ---------------------------------------------------------------------------

ALL_MULTI_CLASSES = ("BE", "BM", "ME", "MM")
ALL_BINARY_CLASSES = ("B", "M")

# Paper-reported approximate class distribution (Table 1, Leon et al. 2020)
PAPER_DISTRIBUTION = {
    "BE": 0.07,
    "BM": 0.45,
    "ME": 0.32,
    "MM": 0.16,
}


def compute_class_distribution(
    labels: List[str],
) -> Dict[str, float]:
    """Compute fractional class distribution from a list of label strings.

    Parameters
    ----------
    labels : list of str (e.g. ['BE', 'BM', 'ME', ...])

    Returns
    -------
    dict mapping class → fraction
    """
    from collections import Counter
    counts = Counter(labels)
    total = sum(counts.values())
    return {cls: counts.get(cls, 0) / total for cls in sorted(counts.keys())}


def compare_distribution_with_paper(
    actual_dist: Dict[str, float],
    paper_dist: Dict[str, float] = PAPER_DISTRIBUTION,
    tolerance: float = 0.15,
) -> Dict[str, dict]:
    """Compare actual class distribution to paper's reported values.

    Parameters
    ----------
    actual_dist : dict class → fraction
    paper_dist : dict class → fraction
    tolerance : float
        Maximum allowed absolute deviation from paper's values (default 0.15 = 15pp).

    Returns
    -------
    dict mapping class → {actual, paper, diff, within_tolerance}
    """
    result = {}
    for cls in sorted(set(list(actual_dist.keys()) + list(paper_dist.keys()))):
        actual = actual_dist.get(cls, 0.0)
        paper = paper_dist.get(cls, 0.0)
        diff = abs(actual - paper)
        result[cls] = {
            "actual": actual,
            "paper": paper,
            "diff": diff,
            "within_tolerance": diff <= tolerance,
        }
    return result


# ---------------------------------------------------------------------------
# 7.3  Class weight computation
# ---------------------------------------------------------------------------

def compute_class_weights(
    labels: List[str],
    method: str = "inverse_frequency",
) -> Dict[str, float]:
    """Compute class weights for an imbalanced classification task.

    Parameters
    ----------
    labels : list of str
    method : str
        'inverse_frequency' — weight_c = N / (C * count_c)
        where N = total samples, C = number of classes, count_c = class count.

    Returns
    -------
    dict mapping class → weight (float)
    """
    from collections import Counter
    counts = Counter(labels)
    n_total = len(labels)
    n_classes = len(counts)

    if method == "inverse_frequency":
        weights = {
            cls: n_total / (n_classes * cnt)
            for cls, cnt in counts.items()
        }
    else:
        raise ValueError(f"Unknown method: {method!r}")

    return weights
