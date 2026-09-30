import json
import os
import re
import sys
import copy
from pathlib import Path
from collections import Counter
import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from tqdm import tqdm

# Allow running from scratch/ subdirectory or from the model root
_MODEL_ROOT = Path(__file__).resolve().parent.parent
if str(_MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(_MODEL_ROOT))

from src.cli import get_model  # type: ignore[import]
from src.pretrain import load_pretrained_encoder  # type: ignore[import]
from src.hsi_dataset import make_dataloader  # type: ignore[import]
from src.finetuning import freeze_encoder, unfreeze_encoder, get_layer_wise_lr  # type: ignore[import]
from src.training import train_epoch, validate, EarlyStopping  # type: ignore[import]
from src.utils import FocalLoss  # type: ignore[import]

patient_re = re.compile(r"^P(\d+)_C\d+$", re.IGNORECASE)
def get_pid(sid):
    m = patient_re.match(sid.strip())
    return int(m.group(1)) if m else sid

def run_kfold_cv(n_folds=5, phase1_epochs=5, phase2_epochs=15, batch_size=64):
    print("=" * 80)
    print(f"STAGE 3: PATIENT-LEVEL {n_folds}-FOLD STRATIFIED CROSS-VALIDATION")
    print(f"Config: Phase 1 = {phase1_epochs} epochs, Phase 2 = {phase2_epochs} epochs, Batch Size = {batch_size}")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pretrained_path = "outputs/pretrained_encoder_best.pth"
    assert os.path.exists(pretrained_path), f"Pretrained weights missing: {pretrained_path}"

    with open("../preprocessing/outputs/splits/kfold_splits.json", "r") as f:
        kfold_data = json.load(f)

    folds = kfold_data["folds"]
    class_names = ["BE", "BM", "ME", "MM"]

    fold_metrics_list = []

    for fold_idx in range(n_folds):
        fold_key = f"fold_{fold_idx}"
        fold_info = folds[fold_key]
        train_sids = fold_info["train"]
        val_sids = fold_info["val"]

        train_pids = sorted(list({get_pid(s) for s in train_sids}))
        val_pids = sorted(list({get_pid(s) for s in val_sids}))

        # Collect class labels for val
        val_labels_dict = {}
        for s in val_sids:
            with open(f"../preprocessing/outputs/processed/per_pixel/{s}/{s}__meta.json") as mf:
                val_labels_dict[s] = json.load(mf)["multiLabel"]

        val_dist = Counter(val_labels_dict.values())
        # Reporting only: captures from one patient must not be printed repeatedly.
        be_val_pts = sorted({get_pid(s) for s, c in val_labels_dict.items() if c == "BE"})

        print("\n" + "=" * 70)
        print(f"RUNNING {fold_key.upper()} (Validation BE Patients: {be_val_pts})")
        print(f"Train captures: {len(train_sids)} ({len(train_pids)} pts) | Val captures: {len(val_sids)} ({len(val_pids)} pts)")
        print(f"Val Class Distribution: {dict(val_dist)}")
        print("=" * 70)

        # Create data loaders
        train_loader = make_dataloader(
            split=f"{fold_key}_train",
            batch_size=batch_size,
            label_mode="multi",
            normalization="per_pixel",
            data_dir="../preprocessing/outputs",
            use_sampler=True
        )
        val_loader = make_dataloader(
            split=f"{fold_key}_val",
            batch_size=batch_size,
            label_mode="multi",
            normalization="per_pixel",
            data_dir="../preprocessing/outputs",
            use_sampler=False
        )

        # Initialize fresh model from pretrained encoder
        model = get_model("self_attention", dropout=0.4, attn_dim=128)
        model = load_pretrained_encoder(model, pretrained_path, device=device)
        model.to(device)

        # Criterion: gentle focal loss
        criterion = FocalLoss(gamma=1.5, weight=None, label_smoothing=0.0)

        # ----------------------------------------------------
        # Phase 1: Frozen Encoder Warmup
        # ----------------------------------------------------
        freeze_encoder(model)
        head_params = list(model.out.parameters())
        optimizer_p1 = optim.AdamW(head_params, lr=0.0005, weight_decay=5e-5)
        scheduler_p1 = optim.lr_scheduler.CosineAnnealingLR(optimizer_p1, T_max=phase1_epochs, eta_min=5e-6)

        best_val_f1 = -1.0
        best_state = None

        for ep in range(phase1_epochs):
            train_epoch(model, train_loader, optimizer_p1, criterion, device, accumulation_steps=2)
            vm, vl, vp = validate(model, val_loader, criterion, device)
            scheduler_p1.step()
            star = " ★" if vm["f1_macro"] > best_val_f1 else ""
            print(f"  [Phase1 Ep {ep+1}/{phase1_epochs}] Val Macro F1={vm['f1_macro']:.4f} | Acc={vm['accuracy']*100:.1f}%{star}")
            if vm["f1_macro"] > best_val_f1:
                best_val_f1 = vm["f1_macro"]
                best_state = copy.deepcopy(model.state_dict())

        if best_state is not None:
            model.load_state_dict(best_state)

        # ----------------------------------------------------
        # Phase 2: Full Fine-Tuning
        # ----------------------------------------------------
        unfreeze_encoder(model)
        param_groups = get_layer_wise_lr(model, base_lr=0.0005, encoder_lr=0.00005)
        optimizer_p2 = optim.AdamW(param_groups, weight_decay=5e-5)
        scheduler_p2 = optim.lr_scheduler.CosineAnnealingLR(optimizer_p2, T_max=phase2_epochs, eta_min=5e-7)
        early_stopping = EarlyStopping(patience=8, mode="max", min_delta=0.001)

        for ep in range(phase2_epochs):
            train_epoch(model, train_loader, optimizer_p2, criterion, device, accumulation_steps=2)
            vm, vl, vp = validate(model, val_loader, criterion, device)
            scheduler_p2.step()
            star = " ★" if vm["f1_macro"] > best_val_f1 else ""
            print(f"  [Phase2 Ep {ep+1}/{phase2_epochs}] Val Macro F1={vm['f1_macro']:.4f} | Acc={vm['accuracy']*100:.1f}%{star}")
            if vm["f1_macro"] > best_val_f1:
                best_val_f1 = vm["f1_macro"]
                best_state = copy.deepcopy(model.state_dict())

            if early_stopping(vm["f1_macro"]):
                print(f"  [Early Stop] No improvement for {early_stopping.patience} epochs — stopping Phase 2.")
                break

        # Final evaluation on this fold's validation set using best weights
        if best_state is not None:
            model.load_state_dict(best_state)

        vm, vl, vp = validate(model, val_loader, criterion, device)
        y_true = np.array(vl)
        y_pred = np.array(vp)

        # 4-class metrics
        macro_f1 = f1_score(y_true, y_pred, average="macro")
        weighted_f1 = f1_score(y_true, y_pred, average="weighted")
        acc = accuracy_score(y_true, y_pred)
        bal_acc = balanced_accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])

        prec_per: np.ndarray = np.atleast_1d(precision_score(y_true, y_pred, average=None, labels=[0, 1, 2, 3], zero_division=0))  # type: ignore[assignment]
        rec_per: np.ndarray = np.atleast_1d(recall_score(y_true, y_pred, average=None, labels=[0, 1, 2, 3], zero_division=0))  # type: ignore[assignment]
        f1_per: np.ndarray = np.atleast_1d(f1_score(y_true, y_pred, average=None, labels=[0, 1, 2, 3], zero_division=0))  # type: ignore[assignment]

        # Binary clinical metrics (Benign: BE+BM vs Malignant: ME+MM)
        y_true_bin = (y_true >= 2).astype(int)
        y_pred_bin = (y_pred >= 2).astype(int)

        bin_acc = accuracy_score(y_true_bin, y_pred_bin)
        bin_bal_acc = balanced_accuracy_score(y_true_bin, y_pred_bin)
        bin_macro_f1 = f1_score(y_true_bin, y_pred_bin, average="macro")
        bin_cm = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1])
        tn, fp, fn, tp = bin_cm.ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        fold_res = {
            "fold": fold_key,
            "val_patients": val_pids,
            "be_val_patients": be_val_pts,
            "macro_f1": float(macro_f1),
            "weighted_f1": float(weighted_f1),
            "accuracy": float(acc),
            "balanced_accuracy": float(bal_acc),
            "per_class": {
                cname: {
                    "f1": float(f1_per[i]),
                    "recall": float(rec_per[i]),
                    "precision": float(prec_per[i])
                } for i, cname in enumerate(class_names)
            },
            "cm": cm.tolist(),
            "binary": {
                "accuracy": float(bin_acc),
                "balanced_acc": float(bin_bal_acc),
                "macro_f1": float(bin_macro_f1),
                "sensitivity": float(sens),
                "specificity": float(spec),
                "cm": bin_cm.tolist()
            }
        }
        fold_metrics_list.append(fold_res)

        print(f"\n[{fold_key.upper()} RESULT]")
        print(f"  Macro F1: {macro_f1:.4f} | Weighted F1: {weighted_f1:.4f} | Accuracy: {acc*100:.2f}%")
        for i, cn in enumerate(class_names):
            print(f"    {cn:>3}: F1={f1_per[i]:.4f}, Recall={rec_per[i]:.4f}, Precision={prec_per[i]:.4f}")
        print(f"  Binary Accuracy: {bin_acc*100:.2f}% | Sens: {sens*100:.2f}% | Spec: {spec*100:.2f}% | Bin Macro F1: {bin_macro_f1:.4f}")
        print("  Confusion Matrix:")
        print(cm)

    # ----------------------------------------------------
    # Summary across folds
    # ----------------------------------------------------
    print("\n" + "=" * 80)
    print("STAGE 3: 5-FOLD CROSS-VALIDATION SUMMARY REPORT")
    print("=" * 80)

    macro_f1s = [f["macro_f1"] for f in fold_metrics_list]
    weighted_f1s = [f["weighted_f1"] for f in fold_metrics_list]
    accs = [f["accuracy"] for f in fold_metrics_list]
    bin_accs = [f["binary"]["accuracy"] for f in fold_metrics_list]
    bin_f1s = [f["binary"]["macro_f1"] for f in fold_metrics_list]

    be_f1s = [f["per_class"]["BE"]["f1"] for f in fold_metrics_list]
    bm_f1s = [f["per_class"]["BM"]["f1"] for f in fold_metrics_list]
    me_f1s = [f["per_class"]["ME"]["f1"] for f in fold_metrics_list]
    mm_f1s = [f["per_class"]["MM"]["f1"] for f in fold_metrics_list]

    be_recs = [f["per_class"]["BE"]["recall"] for f in fold_metrics_list]
    bm_recs = [f["per_class"]["BM"]["recall"] for f in fold_metrics_list]
    me_recs = [f["per_class"]["ME"]["recall"] for f in fold_metrics_list]
    mm_recs = [f["per_class"]["MM"]["recall"] for f in fold_metrics_list]

    def stats(arr):
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr))
        }

    summary = {
        "macro_f1": stats(macro_f1s),
        "weighted_f1": stats(weighted_f1s),
        "accuracy": stats(accs),
        "binary_accuracy": stats(bin_accs),
        "binary_macro_f1": stats(bin_f1s),
        "per_class_f1": {
            "BE": stats(be_f1s),
            "BM": stats(bm_f1s),
            "ME": stats(me_f1s),
            "MM": stats(mm_f1s)
        },
        "per_class_recall": {
            "BE": stats(be_recs),
            "BM": stats(bm_recs),
            "ME": stats(me_recs),
            "MM": stats(mm_recs)
        },
        "folds": fold_metrics_list
    }

    print(f"{'Metric':<20} | {'Mean ± Std':<20} | {'Min':<8} | {'Max':<8}")
    print("-" * 62)
    print(f"{'Macro F1':<20} | {summary['macro_f1']['mean']:.4f} ± {summary['macro_f1']['std']:.4f}     | {summary['macro_f1']['min']:.4f}   | {summary['macro_f1']['max']:.4f}")
    print(f"{'Weighted F1':<20} | {summary['weighted_f1']['mean']:.4f} ± {summary['weighted_f1']['std']:.4f}     | {summary['weighted_f1']['min']:.4f}   | {summary['weighted_f1']['max']:.4f}")
    print(f"{'Accuracy':<20} | {summary['accuracy']['mean']*100:.2f}% ± {summary['accuracy']['std']*100:.2f}%   | {summary['accuracy']['min']*100:.2f}% | {summary['accuracy']['max']*100:.2f}%")
    print(f"{'Binary Accuracy':<20} | {summary['binary_accuracy']['mean']*100:.2f}% ± {summary['binary_accuracy']['std']*100:.2f}%   | {summary['binary_accuracy']['min']*100:.2f}% | {summary['binary_accuracy']['max']*100:.2f}%")
    print(f"{'Binary Macro F1':<20} | {summary['binary_macro_f1']['mean']:.4f} ± {summary['binary_macro_f1']['std']:.4f}     | {summary['binary_macro_f1']['min']:.4f}   | {summary['binary_macro_f1']['max']:.4f}")
    print("-" * 62)
    print("Per-Class F1 across Folds:")
    for cn in class_names:
        st = summary["per_class_f1"][cn]
        print(f"  {cn:>3} F1: {st['mean']:.4f} ± {st['std']:.4f} (Min: {st['min']:.4f}, Max: {st['max']:.4f})")
    print("-" * 62)
    print("Per-Class Recall across Folds:")
    for cn in class_names:
        st = summary["per_class_recall"][cn]
        print(f"  {cn:>3} Recall: {st['mean']:.4f} ± {st['std']:.4f} (Min: {st['min']:.4f}, Max: {st['max']:.4f})")

    with open("scratch/kfold_cv_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\n[OK] 5-Fold Cross-Validation complete! Saved to scratch/kfold_cv_summary.json")

if __name__ == "__main__":
    run_kfold_cv()
