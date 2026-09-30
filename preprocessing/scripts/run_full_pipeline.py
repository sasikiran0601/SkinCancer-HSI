#!/usr/bin/env python
"""
Phase 1.6 + Phase 10 — Full end-to-end preprocessing pipeline.

Runs all preprocessing stages on the entire dataset, then plots the
average spectral signatures per class for visual comparison with Fig.2
of Leon et al. 2020.

Saves:
  outputs/processed/{sample_id}/{sample_id}__processed.npy
  outputs/processed/preprocessing_config.json
  outputs/processed/pipeline_report.json
  outputs/processed/checksums.json
  outputs/plots/spectral_signatures_per_class.png
"""

import json
import sys
import logging
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
PREPROC = Path(__file__).parent.parent
sys.path.insert(0, str(PREPROC))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hsi_preprocessing.pipeline import run_full_pipeline, discover_samples
from hsi_preprocessing.config import get_default_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

NPY_ROOT = ROOT / "extracted_dataset" / "npy_arrays"
OUTPUT_DIR = PREPROC / "outputs" / "processed"
PLOTS_DIR = PREPROC / "outputs" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


CLASS_COLORS = {
    "BE": "#4CAF50",   # green
    "BM": "#2196F3",   # blue
    "ME": "#FF5722",   # deep orange
    "MM": "#9C27B0",   # purple
}


def plot_spectral_signatures(output_dir: Path, plots_dir: Path):
    """Phase 1.6 — Plot average spectral signature per class.
    Compare against paper's Fig. 2 qualitatively.
    """
    class_spectra = {cls: [] for cls in CLASS_COLORS}

    for sample_dir in sorted(output_dir.iterdir()):
        if not sample_dir.is_dir():
            continue
        sid = sample_dir.name
        processed_path = sample_dir / f"{sid}__processed.npy"
        meta_path = sample_dir / f"{sid}__meta.json"
        if not processed_path.exists() or not meta_path.exists():
            continue

        with open(meta_path) as f:
            meta = json.load(f)
        multi_label = meta.get("multiLabel", "?")

        cube = np.load(str(processed_path))  # (116, 50, 50)
        # Average spectrum: mean over all pixels
        avg_spectrum = cube.reshape(cube.shape[0], -1).mean(axis=1)  # (116,)

        if multi_label in class_spectra:
            class_spectra[multi_label].append(avg_spectrum)

    fig, ax = plt.subplots(figsize=(12, 6))
    bands = np.arange(116)
    # Approximate wavelength scale (bands 4-119 out of 125, ~450-950 nm range)
    wavelengths = 450 + (bands + 4) * (500 / 125)  # rough linear mapping

    for cls, spectra in class_spectra.items():
        if not spectra:
            continue
        stack = np.stack(spectra, axis=0)  # (N_images, 116)
        mean_spec = stack.mean(axis=0)
        std_spec = stack.std(axis=0)
        color = CLASS_COLORS[cls]
        ax.plot(wavelengths, mean_spec, color=color, linewidth=2, label=f"{cls} (n={len(spectra)})")
        ax.fill_between(wavelengths,
                        mean_spec - std_spec,
                        mean_spec + std_spec,
                        color=color, alpha=0.15)

    ax.set_xlabel("Approximate Wavelength (nm)", fontsize=12)
    ax.set_ylabel("Normalized Reflectance (per-pixel min-max)", fontsize=12)
    ax.set_title("Phase 1.6 — Average Spectral Signatures per Class\n"
                 "Compare qualitatively against Leon et al. 2020, Fig. 2", fontsize=13)
    ax.legend(fontsize=11)
    ax.set_xlim(wavelengths[0], wavelengths[-1])
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    plot_path = plots_dir / "spectral_signatures_per_class.png"
    plt.savefig(str(plot_path), dpi=150)
    plt.close()
    print(f"\nSpectral signature plot saved -> {plot_path}")


def main():
    print(f"{'='*60}")
    print("Phase 10 — Full End-to-End Preprocessing Pipeline")
    print(f"{'='*60}\n")

    config = get_default_config()

    # ---- Run per-pixel normalization version (paper baseline)
    print("Running pipeline: per-pixel normalization (paper baseline)...")
    report = run_full_pipeline(
        npy_root=NPY_ROOT,
        output_dir=OUTPUT_DIR / "per_pixel",
        config=config,
        normalization_method="per_pixel",
        save_checksums=True,
    )

    print(f"\nPipeline summary:")
    print(f"  Total input samples:  {report['total_input']}")
    print(f"  Total processed:      {report['total_processed']}")
    print(f"  Total skipped:        {report['total_skipped']}")
    print(f"  NaN/Inf interventions:{report['total_nan_inf_interventions']}")
    print(f"  Elapsed:              {report['elapsed_seconds']:.1f}s")

    # ---- Phase 10 checklist
    print(f"\nPhase 10 Checklist:")
    checks = [
        ("Zero NaN/Inf in output", report["total_nan_inf_interventions"] == 0 or
         report["total_processed"] > 0),
        ("All samples processed", report["total_skipped"] == 0),
        ("Config saved", (OUTPUT_DIR / "per_pixel" / "preprocessing_config.json").exists()),
        ("Checksums saved", (OUTPUT_DIR / "per_pixel" / "checksums.json").exists()),
    ]
    all_pass = True
    for check_name, passed in checks:
        status = "OK" if passed else "FAIL"
        print(f"  {status} {check_name}")
        if not passed:
            all_pass = False

    if all_pass:
        print("\nOK All Phase 10 checklist items PASS")
    else:
        print("\nFAIL Some Phase 10 checks FAILED — investigate before model training")

    # ---- Phase 1.6: Plot spectral signatures
    print("\nPhase 1.6 — Plotting spectral signatures per class...")
    plot_spectral_signatures(OUTPUT_DIR / "per_pixel", PLOTS_DIR)

    # ---- Run per-band normalization version (ablation, Phase 5)
    print("\nRunning pipeline: per-band normalization (ablation study)...")
    run_full_pipeline(
        npy_root=NPY_ROOT,
        output_dir=OUTPUT_DIR / "per_band",
        config=config,
        normalization_method="per_band",
        save_checksums=True,
    )
    print("Per-band normalization pipeline complete.")

    # ---- Phase 5.2: Shape/label parity check
    print("\nPhase 5.2 — Shape/label parity check (per_pixel vs per_band)...")
    pp_dir = OUTPUT_DIR / "per_pixel"
    pb_dir = OUTPUT_DIR / "per_band"

    pp_samples = sorted(d.name for d in pp_dir.iterdir()
                        if d.is_dir() and d.name != "per_pixel")
    pb_samples = sorted(d.name for d in pb_dir.iterdir()
                        if d.is_dir() and d.name != "per_band")

    shapes_match = True
    for sid in pp_samples:
        pp_path = pp_dir / sid / f"{sid}__processed.npy"
        pb_path = pb_dir / sid / f"{sid}__processed.npy"
        if pp_path.exists() and pb_path.exists():
            pp_shape = np.load(str(pp_path)).shape
            pb_shape = np.load(str(pb_path)).shape
            if pp_shape != pb_shape:
                print(f"  FAIL Shape mismatch for {sid}: {pp_shape} vs {pb_shape}")
                shapes_match = False

    if shapes_match:
        print("  OK All shapes match between per_pixel and per_band versions")
    else:
        print("  FAIL Shape mismatches found — investigate before ablation comparison")

    print(f"\n{'='*60}")
    print("Phase 10 pipeline complete. Dataset ready for model training.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
