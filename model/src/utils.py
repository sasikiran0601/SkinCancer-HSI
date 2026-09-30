"""
Shared utilities: metrics, early stopping, etc.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)


def require_cuda(device: str | torch.device = "cuda") -> str:
    """Require a working CUDA device for every training operation."""
    requested = torch.device(device)
    if requested.type != "cuda":
        raise RuntimeError(
            f"GPU training is mandatory; received device {requested!s}. "
            "Set device: cuda in the training configuration."
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "GPU training is mandatory, but CUDA is unavailable. "
            "Install a CUDA-enabled PyTorch build and verify the NVIDIA driver."
        )
    return str(requested)


class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss — drops in for nn.CrossEntropyLoss.

    Focal loss down-weights well-classified examples (typically the majority
    class) so the model focuses harder on the rare, hard-to-classify classes
    (BE and MM in your case).

    Parameters
    ----------
    gamma : float
        Focusing parameter. gamma=0 reduces to standard CrossEntropy.
        gamma=2 is the value from the original paper (Lin et al. 2017).
    weight : Tensor or None
        Per-class weights, same as nn.CrossEntropyLoss(weight=...).
    reduction : str
        'mean' or 'sum'.
    label_smoothing : float
        Smoothing factor (0.0 = hard labels, 0.1 recommended).
        Prevents over-confidence on the majority class (BM).
    """

    def __init__(self, gamma: float = 2.0, weight=None, reduction: str = "mean",
                 label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # 1. Compute unweighted cross-entropy so p_t is the true probability:
        #    p_t = exp(-CE_unweighted) = P(target_class)
        ce_loss = F.cross_entropy(
            inputs, targets,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        p_t = torch.exp(-ce_loss)

        # 2. Focal modulation factor: (1 - p_t)^gamma down-weights easy examples
        focal_loss = ((1.0 - p_t) ** self.gamma) * ce_loss

        # 3. Apply class weights alpha_t externally (as in Lin et al. 2017)
        if self.weight is not None:
            focal_loss = self.weight[targets] * focal_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


def compute_metrics(y_true, y_pred):
    """
    Compute classification metrics from label arrays.

    Returns
    -------
    dict with keys: accuracy, f1_macro, precision_macro, recall_macro
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
    }


def specificity_score(y_true, y_pred, labels=None):
    """
    Per-class specificity = TN / (TN + FP).

    Returns
    -------
    dict mapping class index -> specificity
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    specificities = {}
    for i, label in enumerate(labels):
        # True negatives = total - (row_i + col_i - diag_i)
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        denom = tn + fp
        specificities[label] = float(tn / denom) if denom > 0 else 0.0
    return specificities


class EarlyStopping:
    """
    Tracks a monitored metric and returns True when patience is exhausted.

    Parameters
    ----------
    patience : int
        Number of epochs to wait without improvement before stopping.
    mode : str
        'min' — improvement means the metric decreased.
        'max' — improvement means the metric increased.
    min_delta : float
        Minimum change to qualify as an improvement.
        For F1 monitoring (mode='max'), use min_delta=0.002 so small
        noise does not reset the counter.
    """

    def __init__(self, patience=10, mode="min", min_delta=0.0):
        assert mode in ("min", "max")
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best = None
        self.counter = 0

    def __call__(self, metric):
        if not np.isfinite(metric):
            self.counter += 1
            return self.counter >= self.patience

        if self.best is None:
            self.best = metric
            return False

        if self.mode == "min":
            improved = metric < self.best - self.min_delta
        else:
            improved = metric > self.best + self.min_delta

        if improved:
            self.best = metric
            self.counter = 0
        else:
            self.counter += 1

        return self.counter >= self.patience
