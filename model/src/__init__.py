"""
HSI Skin Cancer Classification — Model Training Package.
"""
from .models import (
    BaselineCNN,
    DepthwiseSeparableCNN,
    SelfAttentionCNN,
    SpatialTeacherCNN,
    MaskedAutoencoder,
)
from .augmentations import SpectralAugmenter
from .utils import compute_metrics, specificity_score, EarlyStopping, FocalLoss
from .training import train_model, train_epoch, validate
from .distillation import train_distillation, distillation_loss
from .uncertainty import mc_dropout_predict
from .explainability import compute_band_importance
from .federated import federated_averaging
from .hsi_dataset import (
    HSIPixelDataset,
    HSIImageDataset,
    HSIDistillationDataset,
    make_dataloader,
    make_kfold_dataloaders,
    make_weighted_sampler,
    load_class_weights,
)

__all__ = [
    "BaselineCNN", "DepthwiseSeparableCNN", "SelfAttentionCNN",
    "SpatialTeacherCNN", "MaskedAutoencoder",
    "SpectralAugmenter",
    "compute_metrics", "specificity_score", "EarlyStopping", "FocalLoss",
    "train_model", "train_epoch", "validate",
    "train_distillation", "distillation_loss",
    "mc_dropout_predict",
    "compute_band_importance",
    "federated_averaging",
    "HSIPixelDataset", "HSIImageDataset", "HSIDistillationDataset",
    "make_dataloader", "make_kfold_dataloaders", "make_weighted_sampler", "load_class_weights",
]
