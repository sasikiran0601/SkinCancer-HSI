"""
Phase 1.6 + Phase 10 — Full end-to-end pipeline.
Phase 4   — Band noise/variance profiling.

This module ties together all preprocessing stages:
  1. Load calibratedHsCube + labels from extracted .npy files
  2. Fix NaN/Inf (Phase 3)
  3. Trim bands (Phase 1.2)
  4. Apply moving-average filter (Phase 1.3)
  5. Normalize (per-pixel or per-band) (Phase 1.5 / 5.1)
  6. Save outputs with metadata

The pipeline is fully driven by a config dict (Phase 9) — no hardcoded values.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .calibration import validate_calibrated_cube, fix_non_finite
from .spectral_processing import trim_bands, moving_average_filter
from .normalization import (
    normalize_per_pixel,
    normalize_per_band,
    compute_global_band_stats,
)
from .config import get_default_config, save_config, compute_output_checksums

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_sample(
    sample_dir: Path,
    sample_id: str,
) -> Dict:
    """Load all NPY and label data for one sample.

    Returns
    -------
    dict with keys:
        calibratedHsCube, hsCube, spectralRGB : np.ndarray
        binaryLabel, multiLabel : str
    """
    sample_dir = Path(sample_dir)

    def load_npy(key: str) -> np.ndarray:
        path = sample_dir / f"{sample_id}__{key}.npy"
        return np.load(str(path))

    with open(sample_dir / f"{sample_id}__labels.json") as f:
        labels = json.load(f)

    return {
        "calibratedHsCube": load_npy("calibratedHsCube"),
        "hsCube": load_npy("hsCube"),
        "spectralRGB": load_npy("spectralRGB"),
        "binaryLabel": labels["binaryLabel"],
        "multiLabel": labels["multiLabel"],
        "sample_id": sample_id,
    }


def discover_samples(npy_root: Path) -> List[str]:
    """Return sorted list of all sample IDs in ``npy_root``."""
    npy_root = Path(npy_root)
    return sorted(
        d.name
        for d in npy_root.iterdir()
        if d.is_dir()
    )


# ---------------------------------------------------------------------------
# Band noise profiling (Phase 4.1)
# ---------------------------------------------------------------------------

def compute_band_variance_profile(
    npy_root: Path,
    sample_ids: Optional[List[str]] = None,
    n_samples: int = 20,
    band_axis: int = 0,
    seed: int = 42,
) -> np.ndarray:
    """Compute per-band signal variance averaged across a sample of images.

    Parameters
    ----------
    npy_root : Path
    sample_ids : list or None. If None, randomly select ``n_samples`` images.
    n_samples : int
    band_axis : int
    seed : int

    Returns
    -------
    np.ndarray, shape (125,) — mean per-band variance across selected images.
    """
    npy_root = Path(npy_root)
    all_ids = discover_samples(npy_root)

    if sample_ids is None:
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(all_ids), size=min(n_samples, len(all_ids)), replace=False)
        sample_ids = [all_ids[i] for i in sorted(indices)]

    band_variances = []
    for sid in sample_ids:
        sample_dir = npy_root / sid
        cube = np.load(str(sample_dir / f"{sid}__calibratedHsCube.npy"))
        cube, _ = fix_non_finite(cube, axis=band_axis)
        cube = np.moveaxis(cube, band_axis, 0)  # (B, H, W)
        B = cube.shape[0]
        # Per-band variance across spatial dimensions
        var = cube.reshape(B, -1).var(axis=1)
        band_variances.append(var)

    return np.stack(band_variances, axis=0).mean(axis=0)  # (B,)


# ---------------------------------------------------------------------------
# Single-sample preprocessing
# ---------------------------------------------------------------------------

def preprocess_sample(
    cube: np.ndarray,
    config: Dict,
    global_band_min: Optional[np.ndarray] = None,
    global_band_max: Optional[np.ndarray] = None,
    band_axis: int = 0,
) -> Tuple[np.ndarray, Dict]:
    """Apply the full preprocessing chain to a single calibrated cube.

    Steps:
    1. Fix NaN/Inf (spectral interpolation)
    2. Trim bands
    3. Moving-average filter
    4. Normalize (per-pixel or per-band)

    Parameters
    ----------
    cube : np.ndarray, shape (125, H, W) — calibratedHsCube.
    config : dict — preprocessing configuration.
    global_band_min / max : np.ndarray, shape (116,) — required if
        normalization_method == 'per_band'.
    band_axis : int

    Returns
    -------
    processed_cube : np.ndarray, shape (116, H, W)
    meta : dict with diagnostics (n_nan_fixed, band_count, etc.)
    """
    meta = {}

    # Step 1: NaN/Inf handling
    cube, n_fixed = fix_non_finite(cube, axis=band_axis)
    meta["n_nan_inf_fixed"] = n_fixed

    # Step 2: Band trimming
    cube = trim_bands(
        cube,
        drop_first=config["band_trim_drop_first"],
        drop_last=config["band_trim_drop_last"],
        band_axis=band_axis,
    )
    meta["bands_after_trim"] = cube.shape[band_axis]

    # Step 3: Moving average smoothing
    cube = moving_average_filter(
        cube,
        half_window=config["moving_avg_half_window"],
        band_axis=band_axis,
    )

    # Step 4: Normalization
    norm_method = config.get("normalization_method", "per_pixel")
    if norm_method == "per_pixel":
        cube = normalize_per_pixel(cube, band_axis=band_axis)
    elif norm_method == "per_band":
        assert global_band_min is not None and global_band_max is not None, \
            "per_band normalization requires global_band_min/max"
        cube = normalize_per_band(cube, global_band_min, global_band_max, band_axis=band_axis)
    else:
        raise ValueError(f"Unknown normalization_method: {norm_method!r}")

    meta["normalization_method"] = norm_method
    return cube, meta


# ---------------------------------------------------------------------------
# Full pipeline (Phase 10)
# ---------------------------------------------------------------------------

def run_full_pipeline(
    npy_root: Path,
    output_dir: Path,
    config: Optional[Dict] = None,
    normalization_method: str = "per_pixel",
    save_checksums: bool = True,
) -> Dict:
    """Run the complete preprocessing pipeline on the entire dataset.

    Phase 10 checklist enforced:
    - Zero NaN/Inf in final output arrays
    - Every output has exactly 116 bands
    - Output count matches input count
    - Config saved alongside outputs

    Parameters
    ----------
    npy_root : Path
    output_dir : Path — where processed .npy files are written.
    config : dict or None (uses DEFAULT_CONFIG).
    normalization_method : str — overrides config['normalization_method'].
    save_checksums : bool

    Returns
    -------
    report : dict with pipeline statistics
    """
    from .config import save_config, compute_output_checksums, save_checksums as _save_checksums
    from .calibration import validate_calibrated_cube

    npy_root = Path(npy_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if config is None:
        config = get_default_config()
    config["normalization_method"] = normalization_method

    # Save config
    config_path = output_dir / "preprocessing_config.json"
    save_config(config, config_path)

    sample_ids = discover_samples(npy_root)
    logger.info(f"Found {len(sample_ids)} samples in {npy_root}")

    # --- Phase 5.1: compute global band stats if per_band normalization
    global_band_min = global_band_max = None
    if normalization_method == "per_band":
        logger.info("Computing global per-band statistics for per-band normalization...")
        cubes = []
        for sid in sample_ids:
            sd = npy_root / sid
            raw = np.load(str(sd / f"{sid}__calibratedHsCube.npy"))
            fixed_cube, _ = fix_non_finite(raw)
            # Trim first for stats
            trimmed = trim_bands(
                fixed_cube,
                drop_first=config["band_trim_drop_first"],
                drop_last=config["band_trim_drop_last"],
            )
            cubes.append(trimmed)
        global_band_min, global_band_max = compute_global_band_stats(cubes)
        np.save(str(output_dir / "global_band_min.npy"), global_band_min)
        np.save(str(output_dir / "global_band_max.npy"), global_band_max)

    # --- Process each sample
    report: Dict[str, Any] = {
        "config": config,
        "total_input": len(sample_ids),
        "total_processed": 0,
        "total_skipped": 0,
        "nan_inf_intervention_log": {},
        "validation_failures": [],
        "label_records": [],
    }

    processed_dir = output_dir
    t0 = time.time()

    for sid in sample_ids:
        sd = npy_root / sid
        try:
            data = load_sample(sd, sid)
        except Exception as e:
            logger.error(f"[{sid}] Failed to load: {e}")
            report["validation_failures"].append({"sample_id": sid, "error": str(e)})
            report["total_skipped"] += 1
            continue

        cube = data["calibratedHsCube"]

        # Pre-processing validation (Phase 1.4)
        val_result = validate_calibrated_cube(cube, sample_id=sid)
        if not val_result["valid"]:
            logger.warning(f"[{sid}] Pre-process validation warning: {val_result['reason']}")

        # Full preprocessing chain
        processed, meta = preprocess_sample(
            cube, config,
            global_band_min=global_band_min,
            global_band_max=global_band_max,
        )

        # Phase 10 checklist: verify no NaN/Inf in output
        assert np.all(np.isfinite(processed)), \
            f"[{sid}] Non-finite values remain after processing!"
        # Verify band count
        assert processed.shape[0] == config["expected_bands_after_trim"], \
            f"[{sid}] Wrong band count: {processed.shape[0]}"

        # Log NaN/Inf interventions
        if meta["n_nan_inf_fixed"] > 0:
            report["nan_inf_intervention_log"][sid] = meta["n_nan_inf_fixed"]

        # Save processed cube + metadata
        sample_out_dir = processed_dir / sid
        sample_out_dir.mkdir(parents=True, exist_ok=True)
        np.save(str(sample_out_dir / f"{sid}__processed.npy"), processed)

        # Save label record
        label_rec = {
            "sample_id": sid,
            "binaryLabel": data["binaryLabel"],
            "multiLabel": data["multiLabel"],
            "n_nan_inf_fixed": meta["n_nan_inf_fixed"],
            "bands": processed.shape[0],
        }
        with open(sample_out_dir / f"{sid}__meta.json", "w") as f:
            json.dump(label_rec, f, indent=2)

        report["label_records"].append(label_rec)
        report["total_processed"] += 1

    elapsed = time.time() - t0
    report["elapsed_seconds"] = round(elapsed, 2)
    report["total_nan_inf_interventions"] = sum(report["nan_inf_intervention_log"].values())
    report["fraction_samples_with_interventions"] = (
        len(report["nan_inf_intervention_log"]) / report["total_processed"]
        if report["total_processed"] > 0 else 0.0
    )

    # Phase 10 checklist: output count must match input count
    assert report["total_processed"] + report["total_skipped"] == report["total_input"], \
        "Sample count mismatch!"

    # Save report
    report_path = output_dir / "pipeline_report.json"
    serializable_report = {k: v for k, v in report.items() if k != "config"}
    serializable_report["config"] = config
    with open(report_path, "w") as f:
        json.dump(serializable_report, f, indent=2)

    if save_checksums:
        from .config import compute_output_checksums, save_checksums as _sc
        checksums = compute_output_checksums(output_dir)
        _sc(checksums, output_dir / "checksums.json")

    logger.info(
        f"Pipeline complete: {report['total_processed']}/{report['total_input']} samples "
        f"in {elapsed:.1f}s. NaN/Inf interventions: {report['total_nan_inf_interventions']}"
    )
    return report
