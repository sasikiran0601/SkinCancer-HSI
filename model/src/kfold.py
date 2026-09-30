"""
Patient-Stratified 5-Fold Cross-Validation for HSI Skin Cancer Classification.

Evaluates model generalization across all 5 patient-stratified folds:
- Zero patient leakage across folds
- All 4 classes represented in both train and val for every fold
- Computes Mean ± Std of metrics and aggregate Out-Of-Fold (OOF) evaluation
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.optim as optim
from torch.amp import autocast
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

from .models import (
    BaselineCNN,
    DepthwiseSeparableCNN,
    SelfAttentionCNN,
)
from .augmentations import SpectralAugmenter
from .utils import compute_metrics, EarlyStopping, FocalLoss, require_cuda
from .training import train_epoch, validate
from .hsi_dataset import make_kfold_dataloaders, load_class_weights


def get_model(name: str, **kwargs) -> torch.nn.Module:
    models = {
        "baseline": BaselineCNN,
        "depthwise_separable": DepthwiseSeparableCNN,
        "self_attention": SelfAttentionCNN,
    }
    if name not in models:
        raise ValueError(f"Unknown model architecture: {name!r}. Available: {list(models.keys())}")
    return models[name](**kwargs)


def run_kfold_cross_validation(
    model_name: str = "self_attention",
    n_splits: int = 5,
    config: Optional[Dict[str, Any]] = None,
    save_dir: str = "outputs/kfold",
    device: str = "auto",
) -> Dict[str, Any]:
    """
    Execute complete K-Fold cross-validation across all folds.

    Parameters
    ----------
    model_name : str
        Architecture to evaluate: 'self_attention' | 'depthwise_separable' | 'baseline'
    n_splits : int
        Number of cross-validation folds (default=5).
    config : dict or None
        Training hyperparameter overrides.
    save_dir : str
        Directory to save fold checkpoints and evaluation reports.
    device : str
        'cuda', 'cpu', or 'auto'

    Returns
    -------
    summary_report : dict with fold metrics, aggregate stats, and OOF evaluation.
    """
    config = config or {}
    epochs = config.get("epochs", 50)
    lr = config.get("lr", 0.00695)
    weight_decay = config.get("weight_decay", 0.0000019)
    patience = config.get("patience", 15)
    batch_size = config.get("batch_size", 32)
    dropout = config.get("dropout", 0.5)
    mixup_prob = config.get("mixup_prob", 0.058)
    augment = config.get("augment", True)
    data_dir = config.get("data_dir", "../preprocessing/outputs")
    normalization = config.get("normalization", "per_pixel")
    label_mode = config.get("label_mode", "multi")

    device = require_cuda(device)

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = save_path / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"PATIENT-STRATIFIED {n_splits}-FOLD CROSS-VALIDATION")
    print(f"Model: {model_name} | Device: {device} | Epochs: {epochs} | Batch size: {batch_size}")
    print("=" * 70)

    class_weights = load_class_weights(label_mode=label_mode, data_dir=data_dir, power=2.0).to(device)
    criterion = FocalLoss(gamma=2.5, weight=class_weights)

    fold_results: List[Dict[str, Any]] = []
    oof_all_labels: List[int] = []
    oof_all_preds: List[int] = []

    for fold_idx in range(n_splits):
        print(f"\n>>> Starting Fold {fold_idx + 1}/{n_splits} ...")

        train_loader, val_loader = make_kfold_dataloaders(
            fold_idx=fold_idx,
            batch_size=batch_size,
            label_mode=label_mode,
            normalization=normalization,
            data_dir=data_dir,
            mode="pixel",
        )

        model = get_model(
            model_name,
            num_bands=config.get("num_bands", 116),
            num_classes=config.get("num_classes", 4),
            dropout=dropout,
        ).to(device)

        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        early_stopping = EarlyStopping(patience=patience, mode="min")
        augmenter = SpectralAugmenter() if augment else None

        best_val_f1 = -float("inf")
        best_state = None
        best_epoch = 0

        for epoch in range(epochs):
            train_metrics = train_epoch(
                model, train_loader, optimizer, criterion, device,
                augmenter=augmenter, mixup_prob=mixup_prob,
            )
            val_metrics, _, _ = validate(model, val_loader, criterion, device)
            scheduler.step()

            if val_metrics["f1_macro"] > best_val_f1 and np.isfinite(val_metrics["loss"]):
                best_val_f1 = val_metrics["f1_macro"]
                best_epoch = epoch + 1
                best_state = copy.deepcopy(model.state_dict())

            if early_stopping(val_metrics["loss"]):
                print(f"  [Fold {fold_idx}] Early stopping triggered at epoch {epoch + 1}.")
                break

        if best_state is not None:
            model.load_state_dict(best_state)

        # Save fold checkpoint
        ckpt_file = checkpoints_dir / f"best_{model_name}_fold_{fold_idx}.pth"
        torch.save(model.state_dict(), ckpt_file)

        # Evaluate fold out-of-fold validation set
        model.eval()
        fold_preds, fold_labels = [], []
        with torch.no_grad():
            for batch_data in val_loader:
                x, y = batch_data[0].to(device), batch_data[1].to(device)
                with autocast(device_type=torch.device(device).type, dtype=torch.bfloat16):
                    outputs = model(x)
                _, preds = torch.max(outputs, 1)
                fold_preds.extend(preds.cpu().numpy())
                fold_labels.extend(y.cpu().numpy())

        raw_metrics = compute_metrics(fold_labels, fold_preds)
        fold_metrics: Dict[str, Any] = {
            "accuracy": raw_metrics["accuracy"],
            "f1_macro": raw_metrics["f1_macro"],
            "precision_macro": raw_metrics["precision_macro"],
            "recall_macro": raw_metrics["recall_macro"],
            "fold": fold_idx,
            "best_epoch": best_epoch,
            "checkpoint": str(ckpt_file),
        }
        fold_results.append(fold_metrics)

        oof_all_labels.extend(fold_labels)
        oof_all_preds.extend(fold_preds)

        print(
            f"  [Fold {fold_idx} Finished] Best Epoch: {best_epoch} | "
            f"Val Macro-F1: {fold_metrics['f1_macro']:.4f} | "
            f"Val Accuracy: {fold_metrics['accuracy']:.4f}"
        )

    # Compute aggregate statistics (Mean +- Std)
    f1_scores = [r["f1_macro"] for r in fold_results]
    accuracies = [r["accuracy"] for r in fold_results]
    precisions = [r["precision_macro"] for r in fold_results]
    recalls = [r["recall_macro"] for r in fold_results]

    mean_f1, std_f1 = float(np.mean(f1_scores)), float(np.std(f1_scores))
    mean_acc, std_acc = float(np.mean(accuracies)), float(np.std(accuracies))
    mean_prec, std_prec = float(np.mean(precisions)), float(np.std(precisions))
    mean_rec, std_rec = float(np.mean(recalls)), float(np.std(recalls))

    # Aggregate Out-Of-Fold (OOF) evaluation
    oof_cm = confusion_matrix(oof_all_labels, oof_all_preds).tolist()
    class_names = ["BE", "BM", "ME", "MM"]
    oof_report = classification_report(
        oof_all_labels, oof_all_preds, target_names=class_names, output_dict=True, zero_division=0
    )

    summary_report = {
        "model_name": model_name,
        "n_splits": n_splits,
        "summary": {
            "f1_macro_mean": mean_f1,
            "f1_macro_std": std_f1,
            "accuracy_mean": mean_acc,
            "accuracy_std": std_acc,
            "precision_macro_mean": mean_prec,
            "precision_macro_std": std_prec,
            "recall_macro_mean": mean_rec,
            "recall_macro_std": std_rec,
        },
        "fold_results": fold_results,
        "oof_confusion_matrix": oof_cm,
        "oof_classification_report": oof_report,
    }

    report_file = save_path / f"kfold_{model_name}_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2)

    print("\n" + "=" * 70)
    print(f"K-FOLD CROSS-VALIDATION SUMMARY ({n_splits} Folds, Model: {model_name})")
    print("=" * 70)
    for r in fold_results:
        print(f"  Fold {r['fold']}: F1={r['f1_macro']:.4f} | Acc={r['accuracy']:.4f} | Epoch={r['best_epoch']}")
    print("-" * 70)
    print(f"  Macro-F1    : {mean_f1:.4f} +- {std_f1:.4f}")
    print(f"  Accuracy    : {mean_acc:.4f} +- {std_acc:.4f}")
    print(f"  Precision   : {mean_prec:.4f} +- {std_prec:.4f}")
    print(f"  Recall      : {mean_rec:.4f} +- {std_rec:.4f}")
    print("=" * 70)
    print(f"Detailed report saved -> {report_file}\n")

    return summary_report
