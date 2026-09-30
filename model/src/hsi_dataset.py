"""
HSI Dataset loader — wired to the REAL preprocessed outputs.

Loads from:
    {data_dir}/splits/patient_splits.json
    {data_dir}/processed/{normalization}/{SAMPLE_ID}/{SAMPLE_ID}__processed.npy
    {data_dir}/processed/{normalization}/{SAMPLE_ID}/{SAMPLE_ID}__meta.json
    {data_dir}/reports/label_checks_report.json

The default data_dir is ``../preprocessing/outputs`` (relative to model/).

Two dataset modes
-----------------
- **pixel** (default): Each (116, 50, 50) cube is flattened into 2500
  individual pixel vectors of shape (116,).  All 2500 share the
  image-level label.  This is what the 1-D CNN models expect.

- **image**: Returns whole cubes (116, 50, 50) with one label per cube.
  Useful if you later build a 3-D CNN that consumes entire cubes.

- **distillation**: Returns ``(pixel_vector, patch, label)`` triples
  for knowledge-distillation training (pixel for the student,
  spatial patch for the teacher).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# ============================================================
# Label maps — MUST stay consistent with every other file
# ============================================================
BINARY_LABEL_MAP = {"B": 0, "M": 1}
MULTI_LABEL_MAP = {"BE": 0, "BM": 1, "ME": 2, "MM": 3}

DEFAULT_DATA_DIR = "../preprocessing/outputs"


def _resolve_sample_ids(data_dir: Path | str, split: str) -> list[str]:
    """Resolve sample IDs for patient splits or K-Fold splits.

    Supports:
      - Standard splits: "train", "val", "test" from patient_splits.json
      - Fold splits: "fold_0_train", "fold_0_val", etc. from kfold_splits.json
    """
    data_dir = Path(data_dir)
    if split.startswith("fold_"):
        parts = split.split("_")
        if len(parts) >= 3 and parts[-1] in ("train", "val", "test"):
            fold_key = "_".join(parts[:-1])
            subsplit = parts[-1]
        else:
            raise ValueError(
                f"Invalid fold split format {split!r}. Expected e.g. 'fold_0_train' or 'fold_0_val'"
            )
        kfold_path = data_dir / "splits" / "kfold_splits.json"
        if not kfold_path.exists():
            raise FileNotFoundError(
                f"K-Fold splits not found: {kfold_path}. Run run_kfold_splitting.py first."
            )
        with open(kfold_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        folds = data.get("folds", data)
        return folds[fold_key][subsplit]
    else:
        splits_path = data_dir / "splits" / "patient_splits.json"
        with open(splits_path, "r", encoding="utf-8") as f:
            splits_data = json.load(f)
        return splits_data["splits"][split]


# ===================================================================
# Pixel-level dataset  (for 1-D CNN models)
# ===================================================================
class HSIPixelDataset(Dataset):
    """
    Flattens each (116, 50, 50) cube into 2500 pixel vectors of (116,).
    Each pixel inherits the cube's image-level label.

    Total samples = num_cubes_in_split × 50 × 50.
    """

    def __init__(
        self,
        data_dir: str = DEFAULT_DATA_DIR,
        split: str = "train",
        label_mode: str = "multi",
        normalization: str = "per_pixel",
    ):
        self.data_dir = Path(data_dir)
        self.label_mode = label_mode
        self.normalization = normalization
        self.label_map = BINARY_LABEL_MAP if label_mode == "binary" else MULTI_LABEL_MAP

        # --- Load patient-independent or K-fold split ---
        sample_ids = _resolve_sample_ids(self.data_dir, split)
        if not sample_ids:
            raise ValueError(f"Split {split!r} is empty.")

        # --- Pre-load all cubes and labels into memory ---
        # With 76 cubes × ~2.3 MB each ≈ 175 MB — fits easily in RAM.
        self.pixels = []   # list of (116,) float32 arrays
        self.labels = []   # list of int labels
        self.sample_ids = [] # list of str sample IDs

        processed_dir = self.data_dir / "processed" / self.normalization

        for sample_id in sample_ids:
            sample_dir = processed_dir / sample_id
            npy_path = sample_dir / f"{sample_id}__processed.npy"
            meta_path = sample_dir / f"{sample_id}__meta.json"
            if not npy_path.exists() or not meta_path.exists():
                raise FileNotFoundError(
                    f"Missing processed data for sample {sample_id!r}: "
                    f"{npy_path} and/or {meta_path}"
                )

            # Load cube: (116, 50, 50) → reshape to (2500, 116)
            cube = np.load(npy_path).astype(np.float32)       # (116, 50, 50)
            pixel_vectors = cube.reshape(cube.shape[0], -1).T  # (2500, 116)

            # Load label
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            label_key = "binaryLabel" if label_mode == "binary" else "multiLabel"
            raw_label = meta.get(label_key)
            if raw_label not in self.label_map:
                raise ValueError(
                    f"Sample {sample_id!r} has invalid {label_key}={raw_label!r}."
                )
            label_int = self.label_map[raw_label]

            self.pixels.append(pixel_vectors)
            self.labels.append(np.full(pixel_vectors.shape[0], label_int, dtype=np.int64))
            self.sample_ids.append(np.full(pixel_vectors.shape[0], sample_id, dtype=object))

        # Concatenate into single arrays for O(1) indexing
        self.pixels = np.concatenate(self.pixels, axis=0)   # (N, 116)
        self.labels = np.concatenate(self.labels, axis=0)    # (N,)
        self.sample_ids = np.concatenate(self.sample_ids, axis=0) # (N,)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        x = torch.from_numpy(self.pixels[index])       # (116,) float32
        y = torch.tensor(self.labels[index], dtype=torch.long)
        sample_id = self.sample_ids[index]
        return x, y, sample_id


# ===================================================================
# Image-level dataset  (for 3-D CNN or whole-cube models)
# ===================================================================
class HSIImageDataset(Dataset):
    """
    One sample = one (116, 50, 50) cube + one label.
    Total samples = number of cubes in the split (52 train / 11 val / 13 test).
    """

    def __init__(
        self,
        data_dir: str = DEFAULT_DATA_DIR,
        split: str = "train",
        label_mode: str = "multi",
        normalization: str = "per_pixel",
    ):
        self.data_dir = Path(data_dir)
        self.label_mode = label_mode
        self.normalization = normalization
        self.label_map = BINARY_LABEL_MAP if label_mode == "binary" else MULTI_LABEL_MAP

        self.sample_ids = _resolve_sample_ids(self.data_dir, split)
        if not self.sample_ids:
            raise ValueError(f"Split {split!r} is empty.")
        self.processed_dir = self.data_dir / "processed" / self.normalization

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, index):
        sample_id = self.sample_ids[index]
        sample_dir = self.processed_dir / sample_id

        npy_path = sample_dir / f"{sample_id}__processed.npy"
        meta_path = sample_dir / f"{sample_id}__meta.json"
        if not npy_path.exists() or not meta_path.exists():
            raise FileNotFoundError(f"Missing processed data for sample {sample_id!r}.")
        cube = np.load(npy_path).astype(np.float32)
        x = torch.from_numpy(cube)  # (116, 50, 50)

        with open(sample_dir / f"{sample_id}__meta.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
        label_key = "binaryLabel" if self.label_mode == "binary" else "multiLabel"
        raw_label = meta.get(label_key)
        if raw_label not in self.label_map:
            raise ValueError(
                f"Sample {sample_id!r} has invalid {label_key}={raw_label!r}."
            )
        y = torch.tensor(self.label_map[raw_label], dtype=torch.long)

        return x, y


# ===================================================================
# Distillation dataset  (pixel + spatial patch)
# ===================================================================
class HSIDistillationDataset(Dataset):
    """
    Returns (pixel_vector, patch, label) triples.

    - pixel_vector (116,): for the 1-D student model
    - patch (116, patch_size, patch_size): for the 2-D teacher model
    - label: integer class index
    """

    def __init__(
        self,
        data_dir: str = DEFAULT_DATA_DIR,
        split: str = "train",
        label_mode: str = "multi",
        normalization: str = "per_pixel",
        patch_size: int = 3,
    ):
        self.data_dir = Path(data_dir)
        self.label_mode = label_mode
        self.normalization = normalization
        self.patch_size = patch_size
        self.label_map = BINARY_LABEL_MAP if label_mode == "binary" else MULTI_LABEL_MAP

        sample_ids = _resolve_sample_ids(self.data_dir, split)
        self.sample_ids = sample_ids

        processed_dir = self.data_dir / "processed" / self.normalization

        # Build index: (cube_idx, row, col) for every pixel
        self.cubes = []    # list of (116, 50, 50) float32 arrays
        self.labels_per_cube = []  # list of int
        self.index = []    # list of (cube_idx, h, w)

        for cube_idx, sample_id in enumerate(sample_ids):
            sample_dir = processed_dir / sample_id
            cube = np.load(sample_dir / f"{sample_id}__processed.npy").astype(np.float32)
            self.cubes.append(cube)

            with open(sample_dir / f"{sample_id}__meta.json", "r", encoding="utf-8") as f:
                meta = json.load(f)
            label_key = "binaryLabel" if label_mode == "binary" else "multiLabel"
            label_int = self.label_map[meta[label_key]]
            self.labels_per_cube.append(label_int)

            _, H, W = cube.shape
            for h in range(H):
                for w in range(W):
                    self.index.append((cube_idx, h, w))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        cube_idx, h, w = self.index[index]
        cube = self.cubes[cube_idx]         # (116, 50, 50)
        label = self.labels_per_cube[cube_idx]

        # --- Pixel vector for the student ---
        pixel = torch.from_numpy(cube[:, h, w].copy())  # (116,)

        # --- Spatial patch for the teacher ---
        _, H, W = cube.shape
        half = self.patch_size // 2
        h_start = max(0, h - half)
        h_end = min(H, h + half + 1)
        w_start = max(0, w - half)
        w_end = min(W, w + half + 1)

        patch = cube[:, h_start:h_end, w_start:w_end].copy()  # (116, ph, pw)

        # Pad if the patch is on the edge
        pad_h = self.patch_size - (h_end - h_start)
        pad_w = self.patch_size - (w_end - w_start)
        if pad_h > 0 or pad_w > 0:
            patch = np.pad(patch, ((0, 0), (0, pad_h), (0, pad_w)), mode="edge")

        patch = torch.from_numpy(patch)  # (116, patch_size, patch_size)
        y = torch.tensor(label, dtype=torch.long)

        return pixel, patch, y


# ===================================================================
# Convenience factories
# ===================================================================
def make_weighted_sampler(dataset: HSIPixelDataset, power: float = 0.75) -> WeightedRandomSampler:
    """
    Build a WeightedRandomSampler so minority classes appear more often per batch.

    Uses class-specific power values to give BE, ME, MM stronger oversampling
    while heavily suppressing BM (majority class) to near-zero weight.

    Parameters
    ----------
    power : float
        Unused (kept for backward compat). Uses hard-coded class-specific powers.

    Class-specific power values (BE, BM, ME, MM):
      - BE (7.8%): power=1.0 → weight = (1/0.078)^1.0 = 12.82
      - BM (43.1%): power=0.02 → weight = (1/0.431)^0.02 = 1.08 (heavily suppressed)
      - ME (33.3%): power=1.0 → weight = (1/0.333)^1.0 = 3.00
      - MM (15.7%): power=1.0 → weight = (1/0.157)^1.0 = 6.37

    After normalization, expected batch composition with batch_size=32:
      BE: ~8, BM: ~3, ME: ~12, MM: ~9 (roughly balanced minorities, suppressed majority)
    """
    labels = dataset.labels  # numpy int64 array of length N
    class_counts = np.bincount(labels)  # count per class index
    total = len(labels)
    class_proportions = class_counts / total

    # Inverse frequency raised to class-specific power
    # BE: power=0.75 (strong but not over-dominant — reduces catch-all effect)
    # BM: power=0.02 (HEAVILY suppressed — essentially 1.08 weight vs minorities)
    # ME: power=1.0 (full inverse frequency)
    # MM: power=1.0 (full inverse frequency)
    class_power = np.array([0.75, 0.02, 1.0, 1.0], dtype=np.float32)  # BE, BM, ME, MM

    # Weight = (1 / proportion) ^ power
    inv_freq = 1.0 / np.maximum(class_proportions, 0.01)
    class_weights = inv_freq ** class_power

    sample_weights = torch.from_numpy(class_weights[labels]).float()
    return WeightedRandomSampler(
        weights=sample_weights,  # type: ignore[arg-type]
        num_samples=len(sample_weights),
        replacement=True,
    )


def make_patient_balanced_sampler(dataset: HSIPixelDataset) -> WeightedRandomSampler:
    """Sample pixels so every patient contributes equal total epoch weight."""
    import re

    patient_pattern = re.compile(r"^P(\d+)_", re.IGNORECASE)

    def patient_id(sample_id):
        match = patient_pattern.match(str(sample_id))
        return int(match.group(1)) if match else str(sample_id)

    patient_ids = np.asarray([patient_id(sid) for sid in dataset.sample_ids], dtype=object)
    unique_patients, counts = np.unique(patient_ids, return_counts=True)
    count_by_patient = dict(zip(unique_patients, counts))
    sample_weights = np.asarray(
        [1.0 / count_by_patient[patient] for patient in patient_ids],
        dtype=np.float32,
    )
    return WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
    )


def make_dataloader(
    split: str,
    batch_size: int = 64,
    label_mode: str = "multi",
    normalization: str = "per_pixel",
    data_dir: str = DEFAULT_DATA_DIR,
    mode: str = "pixel",
    patch_size: int = 3,
    shuffle: bool | None = None,
    num_workers: int = 0,
    use_sampler: bool = False,
    sampling_mode: str = "normal",
) -> DataLoader:
    """
    Build a DataLoader for the given split.

    Parameters
    ----------
    mode : str
        'pixel'        → HSIPixelDataset   → returns (x, y)
        'image'        → HSIImageDataset   → returns (cube, y)
        'distillation' → HSIDistillationDataset → returns (pixel, patch, y)
    use_sampler : bool
        Backward-compatible switch for the legacy class-balanced sampler.
        It does not mean patient-balanced sampling.  Use
        ``sampling_mode="patient_balanced"`` explicitly for equal patient
        contribution.  Validation and test splits never receive a sampler.
    """
    if mode == "pixel":
        dataset = HSIPixelDataset(data_dir, split, label_mode, normalization)
    elif mode == "image":
        dataset = HSIImageDataset(data_dir, split, label_mode, normalization)
    elif mode == "distillation":
        dataset = HSIDistillationDataset(data_dir, split, label_mode, normalization, patch_size)
    else:
        raise ValueError(f"Unknown mode: {mode!r}")

    is_train = split == "train" or split.endswith("_train")

    if sampling_mode not in ("normal", "class_balanced", "patient_balanced"):
        raise ValueError(
            f"Unknown sampling_mode={sampling_mode!r}; expected normal, "
            "class_balanced, or patient_balanced."
        )
    # Historical API: use_sampler=True selected the class-weighted sampler.
    # Keep that behavior; patient balancing is an explicit new mode.
    if use_sampler and sampling_mode == "normal":
        sampling_mode = "class_balanced"

    sampler = None
    if use_sampler and is_train and isinstance(dataset, HSIPixelDataset):
        sampler = make_weighted_sampler(dataset)
    elif is_train and isinstance(dataset, HSIPixelDataset):
        if sampling_mode == "class_balanced":
            sampler = make_weighted_sampler(dataset)
        elif sampling_mode == "patient_balanced":
            sampler = make_patient_balanced_sampler(dataset)
    if sampler is not None:
        shuffle = False  # sampler is mutually exclusive with shuffle
    else:
        if shuffle is None:
            shuffle = is_train

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        drop_last=False,
    )


def make_kfold_dataloaders(
    fold_idx: int,
    batch_size: int = 64,
    label_mode: str = "multi",
    normalization: str = "per_pixel",
    data_dir: str = DEFAULT_DATA_DIR,
    mode: str = "pixel",
    patch_size: int = 3,
    num_workers: int = 0,
    use_sampler: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """Build (train_loader, val_loader) for a specific K-fold index (0..n_splits-1).

    use_sampler : bool
        Attach a WeightedRandomSampler to the training fold to balance BE/MM.
        Defaults to True — this is the primary fix for BE class collapse.
    """
    train_loader = make_dataloader(
        split=f"fold_{fold_idx}_train",
        batch_size=batch_size,
        label_mode=label_mode,
        normalization=normalization,
        data_dir=data_dir,
        mode=mode,
        patch_size=patch_size,
        shuffle=not use_sampler,  # sampler and shuffle are mutually exclusive
        num_workers=num_workers,
        use_sampler=use_sampler,
    )
    val_loader = make_dataloader(
        split=f"fold_{fold_idx}_val",
        batch_size=batch_size,
        label_mode=label_mode,
        normalization=normalization,
        data_dir=data_dir,
        mode=mode,
        patch_size=patch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader


def load_class_weights(
    label_mode: str = "multi",
    data_dir: str = DEFAULT_DATA_DIR,
    power: float | None = None,
) -> torch.Tensor:
    """
    Load class weights with class-specific power scaling.

    Returns a tensor in the exact index order defined by the label map
    (BE=0, BM=1, ME=2, MM=3) for direct use in FocalLoss / CrossEntropyLoss.

    For multi-class, computes inverse frequency from class proportions with uniform
    power=0.3 so loss weights remain gentle/secondary to the WeightedRandomSampler.

    Parameters
    ----------
    power : float or None
        If None and label_mode='multi', uses uniform power=0.3.
        Otherwise applies uniform power to (1 / proportion) ** power.
    """
    report_path = Path(data_dir) / "reports" / "label_checks_report.json"
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    # For multi-class with default power, compute from class proportions
    if power is None and label_mode == "multi":
        label_map = MULTI_LABEL_MAP
        dist = report.get("class_distribution", {})

        # Extract proportions in label map order
        proportions = np.zeros(len(label_map), dtype=np.float32)
        for name, class_idx in label_map.items():
            proportions[class_idx] = float(dist[name]["actual"]) if name in dist else 0.25

        # Inverse frequency: weight = 1 / proportion
        inv_freq = 1.0 / np.maximum(proportions, 0.01)

        # Uniform gentle power for FocalLoss alpha weights:
        # The sampler handles the primary rebalancing. Loss weighting is gentle/secondary
        # to avoid compounding over-suppression of majority or over-boosting of minority.
        class_power = 0.3
        w = inv_freq ** class_power

        # Diagnostic print of effective signal (sampler batch frequency x normalized loss weight)
        samp_power = np.array([0.75, 0.02, 1.0, 1.0], dtype=np.float32)
        samp_w = inv_freq ** samp_power
        samp_freq = samp_w / samp_w.sum()
        w_norm = w / w.sum()
        eff_signal = samp_freq * w_norm
        eff_signal_pct = eff_signal / eff_signal.sum() * 100

        print(f"\n{'='*75}")
        print("DIAGNOSTIC: EFFECTIVE TRAINING SIGNAL (Sampler Freq × FocalLoss Weight)")
        print(f"{'='*75}")
        print(f"{'Class':<6} {'Data %':>8} {'Samp Freq%':>12} {'Loss Wt (Norm)':>16} {'Eff Signal%':>14} {'Balance':>12}")
        print(f"{'-'*75}")
        for name, idx in label_map.items():
            balanced = "OK (~5-50%)" if 4.0 <= eff_signal_pct[idx] <= 50.0 else "CHECK"
            print(f"{name:<6} {proportions[idx]*100:>7.1f}% {samp_freq[idx]*100:>11.1f}% {w_norm[idx]:>16.4f} {eff_signal_pct[idx]:>13.1f}% {balanced:>12}")
        print(f"{'='*75}\n")
    else:
        # Fallback: use pre-computed weights from report with uniform power
        weights_dict = report["class_weights"]
        label_map = BINARY_LABEL_MAP if label_mode == "binary" else MULTI_LABEL_MAP

        ordered = np.zeros(len(label_map), dtype=np.float32)
        for name, class_idx in label_map.items():
            ordered[class_idx] = weights_dict[name]

        power = power or 2.0
        w = ordered ** power

    w = w / w.mean()  # normalize to keep overall loss scale stable
    return torch.tensor(w, dtype=torch.float32)
