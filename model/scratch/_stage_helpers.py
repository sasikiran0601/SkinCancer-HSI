"""Shared, non-training helpers for the numbered controlled-stage scripts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.experiment_runner import (
    DEFAULT_EXPERIMENTS, ExperimentConfig, _check_fold_integrity, _read_folds,
    _stats, run_experiment,
)

ROOT = Path("scratch/experiments")


def run_named(name: str) -> dict[str, Any]:
    config = next(c for c in DEFAULT_EXPERIMENTS if c.name == name)
    return run_experiment(config)


def load_metrics(name: str) -> dict[str, Any]:
    return json.loads((ROOT / name / "metrics.json").read_text(encoding="utf-8"))


def verify_locked_folds(data_dir: str = "../preprocessing/outputs") -> dict[str, Any]:
    data, digest = _read_folds(data_dir)
    for key, fold in data["folds"].items():
        _check_fold_integrity(fold, key)
    return {"kfold_sha256": digest, "folds": sorted(data["folds"])}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")
