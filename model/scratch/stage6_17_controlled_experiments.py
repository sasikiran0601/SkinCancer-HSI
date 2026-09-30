"""Run the controlled experiments and reports from implementation stages 6--17.

This is an execution script only; importing it does not train or inspect the
locked test split.  Run from ``model`` with a CUDA-enabled PyTorch install.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from src.experiment_runner import (
    CLASS_NAMES, DEFAULT_EXPERIMENTS, ExperimentConfig, run_experiment,
)


ROOT = Path("scratch/experiments")


def _read_summaries() -> list[dict[str, Any]]:
    result = []
    for config in DEFAULT_EXPERIMENTS:
        path = ROOT / config.name / "metrics.json"
        result.append(json.loads(path.read_text(encoding="utf-8")))
    return result


def _select_best(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    # Predeclared priority; no arbitrary combined score or test-set selection.
    return max(summaries, key=lambda s: tuple(
        s["summary"][key]["mean"]
        for key in ("BE_f1", "BE_recall", "macro_f1", "MM_f1", "binary_macro_f1")
    ))


def _probability_report(summaries: list[dict[str, Any]]) -> None:
    report: dict[str, Any] = {}
    for summary in summaries:
        name = summary["experiment"]["name"]
        rows = []
        for path in sorted((ROOT / name).glob("fold_*/validation_predictions.csv")):
            with path.open(newline="", encoding="utf-8") as handle:
                rows.extend(csv.DictReader(handle))
        be = [r for r in rows if r["true_label"] == "BE"]
        non_be = [r for r in rows if r["true_label"] != "BE"]
        def values(key: str, subset: list[dict[str, str]]) -> list[float]:
            return [float(r[key]) for r in subset]
        be_prob = values("P(BE)", be)
        non_be_prob = values("P(BE)", non_be)
        report[name] = {
            "true_be_count": len(be), "non_be_count": len(non_be),
            "mean_p_be_true_be": sum(be_prob) / len(be_prob) if be_prob else 0.0,
            "mean_p_be_non_be": sum(non_be_prob) / len(non_be_prob) if non_be_prob else 0.0,
            "median_p_be_true_be": sorted(be_prob)[len(be_prob) // 2] if be_prob else 0.0,
            "percentiles_p_be_true_be": {
                str(p): _percentile(be_prob, p) for p in (5, 25, 50, 75, 95, 100)
            },
            "true_be_probability_assigned_to": {
                cls: sum(1 for r in be if r["predicted_label"] == cls) / len(be) if be else 0.0
                for cls in CLASS_NAMES
            },
        }
        with (ROOT / name / "true_be_validation_probabilities.csv").open("w", newline="", encoding="utf-8") as handle:
            fields = ["fold", "patient_id", "capture_id", "true_label", "P(BE)", "P(BM)", "P(ME)", "P(MM)"]
            writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
            writer.writerows([{field: row[field] for field in fields} for row in be])
    (ROOT / "probability_analysis.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    index = (len(values) - 1) * p / 100
    lo, hi = int(index), min(int(index) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (index - lo)


def _reports(summaries: list[dict[str, Any]]) -> None:
    baseline = summaries[0]["summary"]
    rows = []
    deltas: dict[str, Any] = {}
    for summary in summaries:
        name = summary["experiment"]["name"]; metrics = summary["summary"]
        keys = ("macro_f1", "BE_f1", "BE_recall", "MM_f1", "binary_macro_f1",
                "BM_f1", "ME_f1", "accuracy")
        deltas[name] = {key: metrics[key]["mean"] - baseline[key]["mean"] for key in keys}
        rows.append({"Experiment": name, **{
            key: f'{metrics[key]["mean"]:.4f} ± {metrics[key]["std"]:.4f}'
            for key in ("macro_f1", "BE_f1", "BE_recall", "BM_f1", "ME_f1", "MM_f1", "binary_macro_f1", "accuracy")
        }})
    (ROOT / "metric_deltas.json").write_text(json.dumps(deltas, indent=2), encoding="utf-8")
    (ROOT / "experiment_comparison.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    with (ROOT / "experiment_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    lines = ["Controlled experiment report (all deltas use exp1_ce_normal)", ""]
    for row in rows:
        lines.append(" | ".join(f"{key}: {value}" for key, value in row.items()))
    (ROOT / "experiment_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    summaries = []
    for config in DEFAULT_EXPERIMENTS:
        summaries.append(run_experiment(config))
    _probability_report(summaries)
    best = _select_best(summaries)
    best_config = best["experiment"]
    run_experiment(ExperimentConfig(
        name="exp5_best_plus_spectral_aug", loss_name=best_config["loss_name"],
        sampling_mode=best_config["sampling_mode"], augmentation=True,
        focal_gamma=best_config["focal_gamma"], seed=best_config["seed"],
    ))
    # Include the augmentation follow-up in the comparison, while keeping
    # probability analysis limited to the prespecified experiments 1--4.
    summaries.append(json.loads(
        (ROOT / "exp5_best_plus_spectral_aug" / "metrics.json").read_text(encoding="utf-8")
    ))
    _reports(summaries)


if __name__ == "__main__":
    main()
