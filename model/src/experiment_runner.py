"""Controlled patient-fold experiments (stages 6--17).

This module deliberately has no test-set code.  It reads the existing
``kfold_splits.json`` but never writes it, and requires CUDA before any
training work is attempted.
"""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
)

from .augmentations import SpectralAugmenter
from .cli import get_model
from .finetuning import freeze_encoder, get_layer_wise_lr, unfreeze_encoder
from .hsi_dataset import HSIPixelDataset, make_dataloader
from .pretrain import load_pretrained_encoder
from .training import train_epoch, validate
from .utils import EarlyStopping, FocalLoss, require_cuda

_PID_RE = re.compile(r"^P(\d+)_", re.IGNORECASE)
CLASS_NAMES = ("BE", "BM", "ME", "MM")


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    loss_name: str
    sampling_mode: str
    augmentation: bool = False
    focal_gamma: float = 1.5
    seed: int = 42


DEFAULT_EXPERIMENTS = (
    ExperimentConfig("exp1_ce_normal", "ce", "normal"),
    ExperimentConfig("exp2_ce_patient_balanced", "ce", "patient_balanced"),
    ExperimentConfig("exp3_weighted_ce_patient_balanced", "weighted_ce", "patient_balanced"),
    ExperimentConfig("exp4_focal_patient_balanced", "focal", "patient_balanced"),
)


def patient_id(sample_id: str) -> int | str:
    match = _PID_RE.match(str(sample_id))
    return int(match.group(1)) if match else str(sample_id)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _read_folds(data_dir: str | Path) -> tuple[dict[str, Any], str]:
    path = Path(data_dir) / "splits" / "kfold_splits.json"
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle), digest


def _check_fold_integrity(fold: dict[str, Any], fold_name: str) -> None:
    train = set(fold["train"])
    val = set(fold["val"])
    train_patients = {patient_id(s) for s in train}
    val_patients = {patient_id(s) for s in val}
    overlap = train_patients & val_patients
    if overlap:
        raise RuntimeError(f"{fold_name}: patient leakage detected: {sorted(overlap)!r}")
    if not train or not val:
        raise RuntimeError(f"{fold_name}: train and validation captures must be non-empty")
    print(f"{fold_name}: Train patients: {len(train_patients)}; "
          f"Validation patients: {len(val_patients)}; Overlap: 0")


def _train_weights(dataset: HSIPixelDataset, n_classes: int = 4) -> torch.Tensor:
    """Inverse-frequency weights computed strictly from this fold's train pixels."""
    counts = np.bincount(dataset.labels, minlength=n_classes).astype(np.float64)
    if np.any(counts == 0):
        raise ValueError(f"Training fold has no samples for classes: {np.where(counts == 0)[0].tolist()}")
    return torch.tensor(len(dataset) / (n_classes * counts), dtype=torch.float32)


def _criterion(config: ExperimentConfig, weights: torch.Tensor | None) -> nn.Module:
    if config.loss_name == "ce":
        return nn.CrossEntropyLoss()
    if config.loss_name == "weighted_ce":
        if weights is None:
            raise ValueError("weighted_ce requires fold training weights")
        return nn.CrossEntropyLoss(weight=weights)
    if config.loss_name == "focal":
        return FocalLoss(gamma=config.focal_gamma, weight=None, label_smoothing=0.0)
    raise ValueError(f"Unknown loss_name={config.loss_name!r}")


def _metric_record(y_true: Iterable[int], y_pred: Iterable[int]) -> dict[str, Any]:
    true = np.asarray(list(y_true))
    pred = np.asarray(list(y_pred))
    cm = confusion_matrix(true, pred, labels=list(range(4)))
    per_p = precision_score(true, pred, labels=list(range(4)), average=None, zero_division=0)
    per_r = recall_score(true, pred, labels=list(range(4)), average=None, zero_division=0)
    per_f = f1_score(true, pred, labels=list(range(4)), average=None, zero_division=0)
    true_b = (true >= 2).astype(int)
    pred_b = (pred >= 2).astype(int)
    bcm = confusion_matrix(true_b, pred_b, labels=[0, 1])
    tn, fp, fn, tp = bcm.ravel()
    return {
        "macro_f1": float(f1_score(true, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(true, pred, average="weighted", zero_division=0)),
        "accuracy": float(accuracy_score(true, pred)),
        "per_class": {
            name: {"precision": float(per_p[i]), "recall": float(per_r[i]), "f1": float(per_f[i])}
            for i, name in enumerate(CLASS_NAMES)
        },
        "binary": {
            "accuracy": float(accuracy_score(true_b, pred_b)),
            "macro_f1": float(f1_score(true_b, pred_b, average="macro", zero_division=0)),
            "recall": float(recall_score(true_b, pred_b, zero_division=0)),
            "precision": float(precision_score(true_b, pred_b, zero_division=0)),
            "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        },
        "confusion_matrix": cm.tolist(),
        "normalized_confusion_matrix": (cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)).tolist(),
    }


def _stats(values: list[float]) -> dict[str, float]:
    return {"mean": float(np.mean(values)), "std": float(np.std(values)),
            "min": float(np.min(values)), "max": float(np.max(values))}


def _write_predictions(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["fold", "patient_id", "capture_id", "true_label", "predicted_label",
              "P(BE)", "P(BM)", "P(ME)", "P(MM)"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_experiment(
    config: ExperimentConfig,
    *,
    data_dir: str = "../preprocessing/outputs",
    pretrained_path: str = "outputs/pretrained_encoder_best.pth",
    output_root: str = "scratch/experiments",
    n_folds: int = 5,
    phase1_epochs: int = 5,
    phase2_epochs: int = 15,
    batch_size: int = 64,
    device: str = "cuda",
) -> dict[str, Any]:
    """Run one reproducible experiment and persist config, metrics and predictions."""
    device = require_cuda(device)
    if config.sampling_mode not in {"normal", "patient_balanced"}:
        raise ValueError("Controlled stages allow only normal or patient_balanced sampling")
    _seed_everything(config.seed)
    folds_data, split_hash = _read_folds(data_dir)
    folds = folds_data["folds"]
    if len(folds) != n_folds:
        raise ValueError(f"Expected {n_folds} immutable folds, found {len(folds)}")

    out = Path(output_root) / config.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "experiment_config.json").write_text(json.dumps({
        **asdict(config), "device": device, "data_dir": data_dir,
        "pretrained_path": pretrained_path, "n_folds": n_folds,
        "phase1_epochs": phase1_epochs, "phase2_epochs": phase2_epochs,
        "batch_size": batch_size, "kfold_sha256": split_hash,
        "test_set_used": False,
    }, indent=2), encoding="utf-8")

    fold_results: list[dict[str, Any]] = []
    for fold_idx in range(n_folds):
        fold_name = f"fold_{fold_idx}"
        fold = folds[fold_name]
        _check_fold_integrity(fold, fold_name)
        fold_out = out / fold_name
        fold_out.mkdir(exist_ok=True)
        train_ds = HSIPixelDataset(data_dir, f"{fold_name}_train", "multi", "per_pixel")
        train_weights = _train_weights(train_ds)
        criterion = _criterion(config, train_weights if config.loss_name == "weighted_ce" else None).to(device)
        train_loader = make_dataloader(
            f"{fold_name}_train", batch_size, "multi", "per_pixel", data_dir,
            use_sampler=False, sampling_mode=config.sampling_mode,
        )
        val_loader = make_dataloader(
            f"{fold_name}_val", batch_size, "multi", "per_pixel", data_dir,
            use_sampler=False, sampling_mode="normal",
        )
        model = get_model("self_attention", dropout=0.4, attn_dim=128)
        model = load_pretrained_encoder(model, pretrained_path, device=device).to(device)
        augmenter = SpectralAugmenter() if config.augmentation else None
        best_score, best_state, best_epoch = -float("inf"), None, 0

        freeze_encoder(model)
        opt = optim.AdamW(model.out.parameters(), lr=0.0005, weight_decay=5e-5)
        sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=phase1_epochs, eta_min=5e-6)
        for epoch in range(phase1_epochs):
            train_epoch(model, train_loader, opt, criterion, device, augmenter=augmenter, accumulation_steps=2)
            vm, _, _ = validate(model, val_loader, criterion, device)
            sch.step()
            if vm["f1_macro"] > best_score:
                best_score, best_state, best_epoch = vm["f1_macro"], copy.deepcopy(model.state_dict()), epoch + 1

        unfreeze_encoder(model)
        opt = optim.AdamW(get_layer_wise_lr(model, base_lr=0.0005, encoder_lr=0.00005), weight_decay=5e-5)
        sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=phase2_epochs, eta_min=5e-7)
        stopping = EarlyStopping(patience=8, mode="max", min_delta=0.001)
        for epoch in range(phase2_epochs):
            train_epoch(model, train_loader, opt, criterion, device, augmenter=augmenter, accumulation_steps=2)
            vm, _, _ = validate(model, val_loader, criterion, device)
            sch.step()
            if vm["f1_macro"] > best_score:
                best_score, best_state, best_epoch = vm["f1_macro"], copy.deepcopy(model.state_dict()), phase1_epochs + epoch + 1
            if stopping(vm["f1_macro"]):
                break
        if best_state is None:
            raise RuntimeError(f"{fold_name}: no finite validation checkpoint was produced")
        model.load_state_dict(best_state)
        checkpoint = fold_out / "best_model.pth"
        torch.save(model.state_dict(), checkpoint)

        labels: list[int] = []
        preds: list[int] = []
        rows: list[dict[str, Any]] = []
        model.eval()
        with torch.no_grad():
            for x, y, ids in val_loader:
                logits = model(x.to(device))
                probs = torch.softmax(logits.float(), dim=1).cpu().numpy()
                pp = probs.argmax(axis=1)
                labels.extend(y.numpy().tolist()); preds.extend(pp.tolist())
                for sid, truth, guess, prob in zip(ids, y.numpy(), pp, probs):
                    sid = str(sid)
                    rows.append({"fold": fold_idx, "patient_id": patient_id(sid),
                                 "capture_id": sid, "true_label": CLASS_NAMES[int(truth)],
                                 "predicted_label": CLASS_NAMES[int(guess)],
                                 **{f"P({name})": float(prob[i]) for i, name in enumerate(CLASS_NAMES)}})
        metrics = _metric_record(labels, preds)
        metrics.update({"fold": fold_idx, "best_epoch": best_epoch,
                        "best_validation_macro_f1": float(best_score),
                        "class_weights_train_only": train_weights.tolist(),
                        "checkpoint": str(checkpoint)})
        (fold_out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        _write_predictions(fold_out / "validation_predictions.csv", rows)
        fold_results.append(metrics)

    metric_names = ["macro_f1", "weighted_f1", "accuracy"]
    summary = {"experiment": asdict(config), "kfold_sha256": split_hash,
               "fold_metrics": fold_results, "summary": {}}
    for name in metric_names:
        summary["summary"][name] = _stats([r[name] for r in fold_results])
    for name in ("accuracy", "macro_f1"):
        summary["summary"][f"binary_{name}"] = _stats([r["binary"][name] for r in fold_results])
    for class_name in CLASS_NAMES:
        for metric in ("precision", "recall", "f1"):
            summary["summary"][f"{class_name}_{metric}"] = _stats(
                [r["per_class"][class_name][metric] for r in fold_results])
    (out / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "per_fold_metrics.json").write_text(
        json.dumps(fold_results, indent=2), encoding="utf-8"
    )
    (out / "confusion_matrices.json").write_text(json.dumps({
        str(r["fold"]): {
            "raw": r["confusion_matrix"],
            "normalized": r["normalized_confusion_matrix"],
        } for r in fold_results
    }, indent=2), encoding="utf-8")
    return summary
