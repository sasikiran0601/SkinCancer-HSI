#!/usr/bin/env python
"""
Phase 4.1 — Per-band noise/variance profiling.

Computes per-band signal variance across a sample of images, plots the
profile, and verifies whether the paper's trim range (drop first 4,
last 5 bands) is justified by elevated noise at the edges.

Saves:
  outputs/plots/band_variance_profile.png
  outputs/reports/band_variance_report.json
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
PREPROC = Path(__file__).parent.parent
sys.path.insert(0, str(PREPROC))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from hsi_preprocessing.pipeline import compute_band_variance_profile

NPY_ROOT = ROOT / "extracted_dataset" / "npy_arrays"
PLOTS_DIR = PREPROC / "outputs" / "plots"
REPORT_DIR = PREPROC / "outputs" / "reports"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("Computing per-band variance profile...")
    variance_profile = compute_band_variance_profile(
        NPY_ROOT, n_samples=20, seed=42
    )

    # --- Analysis
    n_bands = len(variance_profile)
    edge_first = variance_profile[:4]
    edge_last  = variance_profile[-5:]
    middle     = variance_profile[4:-5]

    mean_edge_first = edge_first.mean()
    mean_edge_last  = edge_last.mean()
    mean_middle     = middle.mean()

    paper_trim_justified = (
        mean_edge_first > mean_middle * 1.1 or
        mean_edge_last  > mean_middle * 1.1
    )

    print(f"\nVariance profile summary:")
    print(f"  Bands 0-3 (to be dropped):   mean_var = {mean_edge_first:.6f}")
    print(f"  Bands 4-119 (retained):       mean_var = {mean_middle:.6f}")
    print(f"  Bands 120-124 (to be dropped): mean_var = {mean_edge_last:.6f}")
    print(f"  Paper's trim justified by data: {paper_trim_justified}")

    # --- Plot
    fig, ax = plt.subplots(figsize=(12, 5))
    bands = np.arange(n_bands)
    ax.plot(bands, variance_profile, color="#2196F3", linewidth=1.5,
            label="Per-band variance (mean over 20 images)")

    # Shade dropped regions
    ax.axvspan(-0.5, 3.5, alpha=0.2, color="red", label="Dropped (paper): bands 0-3")
    ax.axvspan(119.5, n_bands - 0.5, alpha=0.2, color="orange",
               label="Dropped (paper): bands 120-124")
    ax.axvspan(3.5, 119.5, alpha=0.05, color="green", label="Retained: bands 4-119")

    # Annotate means
    ax.axhline(mean_middle, color="green", linestyle="--", alpha=0.7,
               label=f"Retained mean={mean_middle:.5f}")
    ax.axhline(mean_edge_first, color="red", linestyle=":", alpha=0.7,
               label=f"First-4 mean={mean_edge_first:.5f}")
    ax.axhline(mean_edge_last, color="orange", linestyle=":", alpha=0.7,
               label=f"Last-5 mean={mean_edge_last:.5f}")

    ax.set_xlabel("Band index (0–124)", fontsize=12)
    ax.set_ylabel("Signal variance", fontsize=12)
    ax.set_title("Phase 4.1 — Per-band Signal Variance Profile\n"
                 "(Leon et al. trim range: drop first 4, last 5)", fontsize=13)
    ax.legend(fontsize=9, loc="upper right")
    ax.set_xlim(-1, n_bands)
    plt.tight_layout()

    plot_path = PLOTS_DIR / "band_variance_profile.png"
    plt.savefig(str(plot_path), dpi=150)
    plt.close()
    print(f"\nPlot saved -> {plot_path}")

    # --- Save report
    report = {
        "n_bands": n_bands,
        "n_samples_profiled": 20,
        "mean_var_edge_first_4": float(mean_edge_first),
        "mean_var_edge_last_5": float(mean_edge_last),
        "mean_var_middle_116": float(mean_middle),
        "paper_trim_justified": bool(paper_trim_justified),
        "variance_profile": variance_profile.tolist(),
    }
    rep_path = REPORT_DIR / "band_variance_report.json"
    with open(rep_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report saved -> {rep_path}")


if __name__ == "__main__":
    main()
