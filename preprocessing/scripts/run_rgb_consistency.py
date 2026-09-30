#!/usr/bin/env python
"""
Phase 8 — RGB cross-consistency check.

For each of 10 sample images, derives an approximate RGB composite from
calibratedHsCube and compares it against the dataset's spectralRGB using
Pearson correlation and cosine similarity.

Saves:
  outputs/reports/rgb_consistency_report.json
  outputs/plots/rgb_consistency_scatter.png
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

from hsi_preprocessing.validation import derive_rgb_from_cube, compute_rgb_similarity

NPY_ROOT = ROOT / "extracted_dataset" / "npy_arrays"
PLOTS_DIR = PREPROC / "outputs" / "plots"
REPORT_DIR = PREPROC / "outputs" / "reports"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

SIMILARITY_THRESHOLD = 0.7


def main():
    sample_dirs = sorted(d for d in NPY_ROOT.iterdir() if d.is_dir())
    # Evaluate on first 10 samples
    sample_dirs = sample_dirs[:10]
    print(f"Phase 8 — RGB Cross-Consistency (first 10 samples)\n")

    results = []
    pearson_scores = []

    for sd in sample_dirs:
        sid = sd.name
        try:
            cube = np.load(str(sd / f"{sid}__calibratedHsCube.npy"))
            ref_rgb = np.load(str(sd / f"{sid}__spectralRGB.npy"))
        except Exception as e:
            print(f"  [SKIP] {sid}: {e}")
            continue

        from hsi_preprocessing.calibration import fix_non_finite
        cube, _ = fix_non_finite(cube)
        ref_rgb, _ = fix_non_finite(ref_rgb)

        derived_rgb = derive_rgb_from_cube(cube)
        metrics = compute_rgb_similarity(derived_rgb, ref_rgb)
        results.append({"sample_id": sid, **metrics})
        pearson_scores.append(metrics["pearson_r"])

        ok = "OK" if metrics["pearson_r"] > SIMILARITY_THRESHOLD else "FAIL"
        print(f"  {ok} {sid:16s} | pearson_r={metrics['pearson_r']:.4f} | "
              f"cosine={metrics['cosine_sim']:.4f} | "
              f"MAD={metrics['mean_abs_diff']:.4f}")

    avg_r = float(np.mean(pearson_scores)) if pearson_scores else 0.0
    n_above = sum(r > SIMILARITY_THRESHOLD for r in pearson_scores)

    print(f"\n{'='*60}")
    print(f"Average Pearson r: {avg_r:.4f}")
    print(f"Samples above threshold ({SIMILARITY_THRESHOLD}): {n_above}/{len(pearson_scores)}")
    if avg_r > SIMILARITY_THRESHOLD:
        print("OK Phase 8 PASS: RGB consistency above threshold")
    else:
        print("WARN Phase 8 WARNING: Low RGB consistency — check cube orientation/bands")
    print(f"{'='*60}\n")

    # --- Plot
    if results:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        sids = [r["sample_id"] for r in results]
        pearson_vals = [r["pearson_r"] for r in results]
        cosine_vals = [r["cosine_sim"] for r in results]

        x = range(len(sids))
        axes[0].bar(x, pearson_vals, color=["#4CAF50" if v > SIMILARITY_THRESHOLD else "#F44336"
                                             for v in pearson_vals])
        axes[0].axhline(SIMILARITY_THRESHOLD, color="black", linestyle="--",
                        label=f"Threshold ({SIMILARITY_THRESHOLD})")
        axes[0].set_xticks(list(x))
        axes[0].set_xticklabels(sids, rotation=45, ha="right", fontsize=8)
        axes[0].set_ylabel("Pearson r")
        axes[0].set_title("RGB Consistency — Pearson r")
        axes[0].legend()
        axes[0].set_ylim(0, 1.1)

        axes[1].bar(x, cosine_vals, color="#2196F3")
        axes[1].set_xticks(list(x))
        axes[1].set_xticklabels(sids, rotation=45, ha="right", fontsize=8)
        axes[1].set_ylabel("Cosine Similarity")
        axes[1].set_title("RGB Consistency — Cosine Similarity")
        axes[1].set_ylim(0, 1.1)

        plt.suptitle("Phase 8 — calibratedHsCube ↔ spectralRGB Consistency", fontsize=13)
        plt.tight_layout()
        plot_path = PLOTS_DIR / "rgb_consistency.png"
        plt.savefig(str(plot_path), dpi=150)
        plt.close()
        print(f"Plot saved -> {plot_path}")

    # Save report
    report = {
        "n_samples": len(results),
        "similarity_threshold": SIMILARITY_THRESHOLD,
        "average_pearson_r": avg_r,
        "n_above_threshold": n_above,
        "phase8_pass": avg_r > SIMILARITY_THRESHOLD,
        "per_sample": results,
    }
    out_path = REPORT_DIR / "rgb_consistency_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report saved -> {out_path}")


if __name__ == "__main__":
    main()
