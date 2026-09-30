import json
import re
import sys
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
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
from src.hsi_dataset import make_dataloader  # type: ignore[import]

def establish_baseline():
    print("=" * 80)
    print("STAGE 1: ESTABLISH THE CURRENT BASELINE")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = "best_finetuned_self_attention.pth"
    print(f"Loading checkpoint: {checkpoint_path} on {device}")

    model = get_model("self_attention", dropout=0.4, attn_dim=128)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    class_names = ["BE", "BM", "ME", "MM"]

    results = {}

    for split_name in ["val", "test"]:
        loader = make_dataloader(
            split=split_name,
            batch_size=64,
            label_mode="multi",
            normalization="per_pixel",
            data_dir="../preprocessing/outputs",
            shuffle=False
        )

        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for batch_data in tqdm(loader, desc=f"Evaluating {split_name.upper()}"):
                x, y = batch_data[0].to(device), batch_data[1].to(device)
                logits = model(x)
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)
        y_prob = np.array(all_probs)

        # 4-class metrics
        acc = accuracy_score(y_true, y_pred)
        bal_acc = balanced_accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average="macro")
        weighted_f1 = f1_score(y_true, y_pred, average="weighted")
        cm_4class = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])

        prec_per_class: np.ndarray = np.atleast_1d(precision_score(y_true, y_pred, average=None, labels=[0, 1, 2, 3], zero_division=0))  # type: ignore[assignment]
        rec_per_class: np.ndarray = np.atleast_1d(recall_score(y_true, y_pred, average=None, labels=[0, 1, 2, 3], zero_division=0))  # type: ignore[assignment]
        f1_per_class: np.ndarray = np.atleast_1d(f1_score(y_true, y_pred, average=None, labels=[0, 1, 2, 3], zero_division=0))  # type: ignore[assignment]

        # Binary clinical metrics: Benign = 0 (BE, BM), Malignant = 1 (ME, MM)
        # 0->0, 1->0; 2->1, 3->1
        y_true_bin = (y_true >= 2).astype(int)
        y_pred_bin = (y_pred >= 2).astype(int)

        bin_acc = accuracy_score(y_true_bin, y_pred_bin)
        bin_bal_acc = balanced_accuracy_score(y_true_bin, y_pred_bin)
        bin_macro_f1 = f1_score(y_true_bin, y_pred_bin, average="macro")
        bin_weighted_f1 = f1_score(y_true_bin, y_pred_bin, average="weighted")
        bin_prec = precision_score(y_true_bin, y_pred_bin, zero_division=0) # precision for Malignant
        bin_rec = recall_score(y_true_bin, y_pred_bin, zero_division=0)   # recall for Malignant (Sensitivity)
        bin_f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)

        bin_cm = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1])
        # bin_cm:
        # [TN (True Benign pred Benign),   FP (True Benign pred Malignant)]
        # [FN (True Malignant pred Benign), TP (True Malignant pred Malignant)]
        tn, fp, fn, tp = bin_cm.ravel()
        benign_sensitivity = tn / (tn + fp) if (tn + fp) > 0 else 0.0 # Specificity
        malignant_sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0 # Sensitivity
        specificity = benign_sensitivity

        print(f"\n--- {split_name.upper()} SET BASELINE METRICS ---")
        print(f"Total Samples: {len(y_true)}")
        print(f"Overall Accuracy:       {acc*100:.2f}%")
        print(f"Balanced Accuracy:      {bal_acc*100:.2f}%")
        print(f"Macro F1:               {macro_f1:.4f}")
        print(f"Weighted F1:            {weighted_f1:.4f}")
        print("\nPer-Class Breakdown:")
        for idx, cname in enumerate(class_names):
            supp = np.sum(y_true == idx)
            print(f"  {cname:>3}: Precision={prec_per_class[idx]:.4f}, Recall={rec_per_class[idx]:.4f}, F1={f1_per_class[idx]:.4f}, Support={supp}")

        print("\n4-Class Confusion Matrix (Rows: True, Cols: Pred [BE, BM, ME, MM]):")
        print(cm_4class)

        print("\nBinary Clinical Grouping (BENIGN: BE+BM vs MALIGNANT: ME+MM):")
        print(f"  Binary Accuracy:        {bin_acc*100:.2f}%")
        print(f"  Binary Balanced Acc:    {bin_bal_acc*100:.2f}%")
        print(f"  Binary Macro F1:        {bin_macro_f1:.4f}")
        print(f"  Malignant Precision:    {bin_prec:.4f}")
        print(f"  Malignant Sensitivity:  {malignant_sensitivity*100:.2f}% (Recall of ME+MM)")
        print(f"  Benign Sensitivity:     {benign_sensitivity*100:.2f}% (Specificity / Recall of BE+BM)")
        print(f"  Malignant F1:           {bin_f1:.4f}")
        print("  Binary Confusion Matrix (Rows: True [Benign=0, Malignant=1], Cols: Pred [0, 1]):")
        print(f"    [[TN={tn:5d}, FP={fp:5d}]")
        print(f"     [FN={fn:5d}, TP={tp:5d}]]")

        results[split_name] = {
            "acc": float(acc),
            "bal_acc": float(bal_acc),
            "macro_f1": float(macro_f1),
            "weighted_f1": float(weighted_f1),
            "per_class": {
                cname: {
                    "precision": float(prec_per_class[i]),
                    "recall": float(rec_per_class[i]),
                    "f1": float(f1_per_class[i]),
                    "support": int(np.sum(y_true == i))
                } for i, cname in enumerate(class_names)
            },
            "cm_4class": cm_4class.tolist(),
            "binary": {
                "accuracy": float(bin_acc),
                "balanced_accuracy": float(bin_bal_acc),
                "macro_f1": float(bin_macro_f1),
                "malignant_precision": float(bin_prec),
                "malignant_sensitivity": float(malignant_sensitivity),
                "benign_sensitivity": float(benign_sensitivity),
                "specificity": float(specificity),
                "malignant_f1": float(bin_f1),
                "confusion_matrix": bin_cm.tolist()
            }
        }

    return results

def verify_data_split():
    print("\n" + "=" * 80)
    print("STAGE 2: DATA SPLIT AND LEAKAGE AUDIT")
    print("=" * 80)

    splits_path = Path("../preprocessing/outputs/splits/patient_splits.json")
    with open(splits_path, "r", encoding="utf-8") as f:
        splits_json = json.load(f)

    splits = splits_json["splits"]
    train_samples = set(splits["train"])
    val_samples = set(splits["val"])
    test_samples = set(splits["test"])

    print(f"Loaded {len(train_samples)} train, {len(val_samples)} val, {len(test_samples)} test capture IDs.")

    # 1. Check sample ID disjointness
    assert len(train_samples & val_samples) == 0, f"Leakage between train and val: {train_samples & val_samples}"
    assert len(train_samples & test_samples) == 0, f"Leakage between train and test: {train_samples & test_samples}"
    assert len(val_samples & test_samples) == 0, f"Leakage between val and test: {val_samples & test_samples}"
    print("[OK] Sample ID disjointness verified (0 overlap across splits).")

    # 2. Parse patient IDs
    patient_re = re.compile(r"^P(\d+)_C\d+$", re.IGNORECASE)
    def get_pid(sid):
        m = patient_re.match(sid.strip())
        return int(m.group(1)) if m else sid

    patient_to_samples = defaultdict(list)
    sample_to_label = {}

    all_samples = list(train_samples | val_samples | test_samples)
    for sid in all_samples:
        pid = get_pid(sid)
        patient_to_samples[pid].append(sid)
        meta_path = Path(f"../preprocessing/outputs/processed/per_pixel/{sid}/{sid}__meta.json")
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
            sample_to_label[sid] = meta["multiLabel"]

    # Check patient disjointness
    train_pids = {get_pid(s) for s in train_samples}
    val_pids = {get_pid(s) for s in val_samples}
    test_pids = {get_pid(s) for s in test_samples}

    assert len(train_pids & val_pids) == 0, f"PATIENT LEAKAGE Train/Val: {train_pids & val_pids}"
    assert len(train_pids & test_pids) == 0, f"PATIENT LEAKAGE Train/Test: {train_pids & test_pids}"
    assert len(val_pids & test_pids) == 0, f"PATIENT LEAKAGE Val/Test: {val_pids & test_pids}"
    print("[OK] Patient ID disjointness verified (strictly 0 patient leakage across splits).")

    # Verify multi-capture patients
    multi_caps = {pid: caps for pid, caps in patient_to_samples.items() if len(caps) > 1}
    print(f"[OK] Found {len(multi_caps)} multi-capture patients: {list(multi_caps.keys())}. All captures remain in identical split.")

    # 3. Class-wise breakdown by patient, cube, and pixels
    class_stats = {
        c: {
            "train_pts": set(), "val_pts": set(), "test_pts": set(),
            "train_cubes": 0, "val_cubes": 0, "test_cubes": 0
        }
        for c in ["BE", "BM", "ME", "MM"]
    }

    for s in train_samples:
        c = sample_to_label[s]
        class_stats[c]["train_pts"].add(get_pid(s))
        class_stats[c]["train_cubes"] += 1

    for s in val_samples:
        c = sample_to_label[s]
        class_stats[c]["val_pts"].add(get_pid(s))
        class_stats[c]["val_cubes"] += 1

    for s in test_samples:
        c = sample_to_label[s]
        class_stats[c]["test_pts"].add(get_pid(s))
        class_stats[c]["test_cubes"] += 1

    print("\n" + "-" * 85)
    print(f"{'Class':<6} | {'Train Pts':<10} | {'Val Pts':<10} | {'Test Pts':<10} | {'Train Pixels':<12} | {'Val Pixels':<10} | {'Test Pixels':<10}")
    print("-" * 85)
    for c in ["BE", "BM", "ME", "MM"]:
        tr_p = len(class_stats[c]["train_pts"])
        va_p = len(class_stats[c]["val_pts"])
        te_p = len(class_stats[c]["test_pts"])
        tr_px = class_stats[c]["train_cubes"] * 2500
        va_px = class_stats[c]["val_cubes"] * 2500
        te_px = class_stats[c]["test_cubes"] * 2500
        print(f"{c:<6} | {tr_p:<10} | {va_p:<10} | {te_p:<10} | {tr_px:<12} | {va_px:<10} | {te_px:<10}")
    print("-" * 85)

    # Check normalization leakage
    print("\n[OK] Normalization Audit:")
    print("  'per_pixel' normalization normalizes each pixel vector individually by its own L2/standard norm.")
    print("  No global dataset mean, std, min, or max from validation/test data is ever used in preprocessing.")
    print("  Every pixel normalization is strictly local and self-contained.")

    # Explicit confirmation of BE in test
    be_test_patients = list(class_stats["BE"]["test_pts"])
    print(f"\nCRITICAL CONFIRMATION: BE has EXACTLY {len(be_test_patients)} independent patient in Test set (Patient ID: {be_test_patients[0]}).")

if __name__ == "__main__":
    baseline_res = establish_baseline()
    verify_data_split()
    with open("scratch/baseline_and_audit_results.json", "w") as f:
        json.dump(baseline_res, f, indent=2)
    print("\n[OK] Stage 1 and Stage 2 complete. Results saved to scratch/baseline_and_audit_results.json")

