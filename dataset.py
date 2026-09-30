"""
PyTorch Dataset / DataLoader for the Hyperspectral Skin Cancer Dataset
=========================================================================
Loads ONLY from the finished preprocessing outputs:

    preprocessing/outputs/splits/patient_splits.json
    preprocessing/outputs/processed/{per_pixel|per_band}/{SAMPLE_ID}/{SAMPLE_ID}__processed.npy
    preprocessing/outputs/processed/{per_pixel|per_band}/{SAMPLE_ID}/{SAMPLE_ID}__meta.json
    preprocessing/outputs/reports/label_checks_report.json

Does NOT touch extracted_dataset/ or any .mat file directly -- that stage
is finished and sealed.

No model code here -- this file only produces batches of (X, y) tensors
ready to be fed into whatever architecture you build next.
"""

import json
import os
from pathlib import Path

import numpy as np
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
from torch.utils.data import Dataset, DataLoader
# ============================================================
# CONFIGURATION
# ============================================================
PREPROCESSING_OUTPUT_DIR = "./preprocessing/outputs"

# Fixed, consistent label -> integer mappings (must never change order
# once training starts, or checkpoints/reports become incomparable)
BINARY_LABEL_MAP = {"B": 0, "M": 1}
MULTI_LABEL_MAP = {"BE": 0, "BM": 1, "ME": 2, "MM": 3}
# ============================================================


class HyperspectralSkinDataset(Dataset):
    """
    One sample = one processed hyperspectral cube + its label.

    Parameters
    ----------
    split : str
        One of "train", "val", "test" -- looked up in patient_splits.json.
    label_mode : str
        "binary"  -> uses binaryLabel  ("B"/"M")
        "multi"   -> uses multiLabel   ("BE"/"BM"/"ME"/"MM")
    normalization : str
        "per_pixel" or "per_band" -- selects which processed-output
        subfolder to load from (see Phase 5 ablation study note).
    output_dir : str
        Root of the preprocessing outputs (defaults to module constant).
    transform : callable, optional
        Optional function applied to the loaded numpy cube before it's
        converted to a tensor (e.g. data augmentation). Left as a hook --
        no augmentation is applied by default.
    """

    def __init__(
        self,
        split,
        label_mode="multi",
        normalization="per_pixel",
        output_dir=PREPROCESSING_OUTPUT_DIR,
        transform=None,
    ):
        assert split in ("train", "val", "test") or split.startswith("fold_"), \
            f"Unknown split: {split}"
        assert label_mode in ("binary", "multi"), f"Unknown label_mode: {label_mode}"
        assert normalization in ("per_pixel", "per_band"), \
            f"Unknown normalization: {normalization}"

        self.output_dir = Path(output_dir)
        self.split = split
        self.label_mode = label_mode
        self.normalization = normalization
        self.transform = transform

        self.label_map = BINARY_LABEL_MAP if label_mode == "binary" else MULTI_LABEL_MAP

        # --- 1. Load the patient-independent or K-Fold split (sample ID list) ---
        if split.startswith("fold_"):
            parts = split.split("_")
            if len(parts) >= 3 and parts[-1] in ("train", "val", "test"):
                fold_key = "_".join(parts[:-1])
                subsplit = parts[-1]
            else:
                raise ValueError(f"Invalid fold split format: {split}")
            splits_path = self.output_dir / "splits" / "kfold_splits.json"
            if not splits_path.exists():
                raise FileNotFoundError(f"K-Fold splits not found: {splits_path}")
            with open(splits_path, "r", encoding="utf-8") as f:
                kdata = json.load(f)
            self.sample_ids = kdata.get("folds", kdata)[fold_key][subsplit]
        else:
            splits_path = self.output_dir / "splits" / "patient_splits.json"
            if not splits_path.exists():
                raise FileNotFoundError(f"Splits file not found: {splits_path}")

            with open(splits_path, "r", encoding="utf-8") as f:
                splits_data = json.load(f)
                
            all_splits = splits_data.get("splits", {})

            if split not in all_splits:
                raise KeyError(
                    f"Split '{split}' not found in {splits_path}. "
                    f"Available: {list(all_splits.keys())}"
                )

            self.sample_ids = all_splits[split]
        if len(self.sample_ids) == 0:
            raise ValueError(f"Split '{split}' is empty in {splits_path}")

        # --- 2. Base directory for processed cubes/labels ---
        self.processed_dir = self.output_dir / "processed" / self.normalization

        # --- 3. Pre-validate every sample has both required files ---
        # Fail loudly and early (at construction time) rather than
        # halfway through an epoch -- exactly the "verify before trusting"
        # principle from the preprocessing plan.
        missing = []
        for sample_id in self.sample_ids:
            sample_dir = self.processed_dir / sample_id
            npy_path = sample_dir / f"{sample_id}__processed.npy"
            meta_path = sample_dir / f"{sample_id}__meta.json"
            if not npy_path.exists() or not meta_path.exists():
                missing.append(sample_id)

        if missing:
            raise FileNotFoundError(
                f"{len(missing)} sample(s) in split '{split}' are missing "
                f"processed .npy or meta.json files. First few: {missing[:5]}"
            )

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, index):
        sample_id = self.sample_ids[index]
        sample_dir = self.processed_dir / sample_id

        # --- Load features (X) ---
        npy_path = sample_dir / f"{sample_id}__processed.npy"
        cube = np.load(npy_path)  # shape (116, 50, 50), float64, in [0,1]

        if self.transform is not None:
            cube = self.transform(cube)

        x = torch.from_numpy(cube).float()  # -> float32 tensor, shape (116,50,50)

        # --- Load target (y) ---
        meta_path = sample_dir / f"{sample_id}__meta.json"
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        label_key = "binaryLabel" if self.label_mode == "binary" else "multiLabel"
        raw_label = meta.get(label_key)

        if raw_label not in self.label_map:
            raise ValueError(
                f"Sample '{sample_id}' has unexpected {label_key}='{raw_label}', "
                f"expected one of {list(self.label_map.keys())}"
            )

        y = torch.tensor(self.label_map[raw_label], dtype=torch.long)

        return x, y, sample_id  # sample_id returned too -- useful for
                                 # debugging/error analysis later, harmless
                                 # to ignore in the training loop if unused


def load_class_weights(label_mode="multi", output_dir=PREPROCESSING_OUTPUT_DIR):
    """
    Loads the inverse-frequency class weights computed during
    preprocessing (Phase 7.3), returned as a torch tensor in the correct
    class-index order for direct use in:

        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    Returns
    -------
    torch.FloatTensor of shape (num_classes,)
    """
    report_path = Path(output_dir) / "reports" / "label_checks_report.json"
    if not report_path.exists():
        raise FileNotFoundError(f"Label checks report not found: {report_path}")

    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    weights_dict = report.get("class_weights")
    if weights_dict is None:
        raise KeyError(
            "'class_weights' not found in label_checks_report.json"
        )

    label_map = BINARY_LABEL_MAP if label_mode == "binary" else MULTI_LABEL_MAP

    # Build the weight tensor in the EXACT index order defined by label_map,
    # not dict iteration order (which is not guaranteed to match).
    ordered_weights = [None] * len(label_map)
    for label_name, class_idx in label_map.items():
        if label_name not in weights_dict:
            raise KeyError(
                f"Class '{label_name}' not found in class_weights "
                f"(available: {list(weights_dict.keys())})"
            )
        ordered_weights[class_idx] = weights_dict[label_name]

    return torch.tensor(ordered_weights, dtype=torch.float32)


def make_dataloader(
    split,
    batch_size=16,
    label_mode="multi",
    normalization="per_pixel",
    shuffle=None,
    num_workers=0,
    output_dir=PREPROCESSING_OUTPUT_DIR,
):
    """
    Convenience wrapper: builds the Dataset and wraps it in a DataLoader.

    shuffle defaults to True for "train" and False for "val"/"test" unless
    explicitly overridden.
    """
    dataset = HyperspectralSkinDataset(
        split=split,
        label_mode=label_mode,
        normalization=normalization,
        output_dir=output_dir,
    )

    if shuffle is None:
        shuffle = (split == "train")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
    )


if __name__ == "__main__":
    # ------------------------------------------------------------
    # Smoke test: confirms the whole loading chain actually works
    # before you build any model on top of it.
    # ------------------------------------------------------------
    print("Loading train split (multi-class, per_pixel normalization)...")
    train_loader = make_dataloader("train", batch_size=4, label_mode="multi",
                                    normalization="per_pixel")

    print(f"Number of training samples: {len(train_loader.dataset)}")  # type: ignore

    x_batch, y_batch, ids_batch = next(iter(train_loader))
    print(f"Batch X shape: {x_batch.shape}, dtype: {x_batch.dtype}")
    print(f"Batch y shape: {y_batch.shape}, values: {y_batch.tolist()}")
    print(f"Sample IDs in this batch: {ids_batch}")
    print(f"X value range: [{x_batch.min().item():.4f}, {x_batch.max().item():.4f}]")

    print("\nLoading class weights (multi-class)...")
    weights = load_class_weights(label_mode="multi")
    print(f"Class weights (order: {list(MULTI_LABEL_MAP.keys())}): {weights}")

    print("\nAll smoke tests passed -- dataset and dataloader are ready for model training.")
