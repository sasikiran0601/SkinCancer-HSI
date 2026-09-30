"""
Training loop with early stopping, class-weighted loss,
mixed-precision training, gradient accumulation, and optional MLflow logging.
"""
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast
from tqdm import tqdm
from sklearn.metrics import classification_report

from .utils import compute_metrics, EarlyStopping, FocalLoss, require_cuda
from .augmentations import SpectralAugmenter
from .contrastive import ContrastiveAugmentor, NTXentLoss


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_epoch(model, loader, optimizer, criterion, device,
                augmenter=None, mixup_prob=0.0, accumulation_steps=1, scheduler=None):
    """
    Training epoch with gradient accumulation support.

    Parameters
    ----------
    accumulation_steps : int
        Number of batches to accumulate before stepping optimizer
        (effective batch size = batch_size × accumulation_steps)
    scheduler : LR scheduler or None
        Not used here (ReduceLROnPlateau is stepped per-epoch in train_model)
    """
    device = require_cuda(device)
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for batch_idx, batch_data in enumerate(tqdm(loader, desc="Training", leave=False)):
        # Unpack robustly to handle both (x, y) and (x, y, sample_ids)
        x, y = batch_data[0].to(device), batch_data[1].to(device)
        sample_ids = batch_data[2] if len(batch_data) > 2 else "unknown"

        use_mixup = False
        y_a, y_b, lam = None, None, None
        if augmenter is not None:
            # Stronger augmentation: increased probabilities
            if torch.rand(1).item() < 0.7:
                x = augmenter.add_noise(x)
            if torch.rand(1).item() < 0.7:
                x = augmenter.band_dropout(x)
            if torch.rand(1).item() < 0.3:
                x = augmenter.spectral_shift(x)
            if torch.rand(1).item() < 0.4:
                x = augmenter.spectrum_scale(x)
            if torch.rand(1).item() < mixup_prob:
                x, (y_a, y_b, lam) = augmenter.mixup(x, y)
                use_mixup = True

        # Use bfloat16 to prevent overflow on Ampere GPUs
        with autocast(device_type=torch.device(device).type, dtype=torch.bfloat16):
            outputs = model(x)
            if use_mixup and y_a is not None and y_b is not None and lam is not None:
                loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
            else:
                loss = criterion(outputs, y)

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite loss detected! Sample IDs in this batch: {sample_ids}. "
                f"Loss value: {loss.item()}. Stopping immediately for debugging."
            )

        # Scale loss for gradient accumulation (outside autocast)
        loss = loss / accumulation_steps
        loss.backward()

        # Accumulation step: update weights every accumulation_steps batches
        if (batch_idx + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        running_loss += loss.item() * accumulation_steps
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    # Flush any remaining accumulated gradients at end of epoch
    if len(loader) % accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

    metrics = compute_metrics(all_labels, all_preds)
    metrics["loss"] = running_loss / len(loader)
    return metrics


def validate(model, loader, criterion, device):
    device = require_cuda(device)
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch_data in tqdm(loader, desc="Validation", leave=False):
            x, y = batch_data[0].to(device), batch_data[1].to(device)
            sample_ids = batch_data[2] if len(batch_data) > 2 else "unknown"

            with autocast(device_type=torch.device(device).type, dtype=torch.bfloat16):
                outputs = model(x)
                loss = criterion(outputs, y)

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite VAL loss detected! Sample IDs in this batch: {sample_ids}. "
                    f"Loss value: {loss.item()}. Batch shape: {x.shape}."
                )

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    metrics = compute_metrics(all_labels, all_preds)
    metrics["loss"] = running_loss / len(loader)
    return metrics, all_labels, all_preds


def train_epoch_with_contrastive(
    model,
    loader,
    optimizer,
    criterion,
    device,
    contrastive_weight=0.1,
    contrastive_head=None,
    contrastive_loss=None,
    contrastive_augmentor=None,
):
    """Train classification with an optional spectral contrastive auxiliary loss.

    The classifier remains the primary objective.  The auxiliary term is
    computed from two augmented views of each input and therefore never uses
    labels from validation or test data.  CUDA is mandatory, consistent with
    the rest of the training API.
    """
    device = require_cuda(device)
    if contrastive_weight < 0:
        raise ValueError("contrastive_weight must be non-negative")
    if contrastive_weight and (contrastive_head is None or contrastive_loss is None):
        raise ValueError("contrastive_head and contrastive_loss are required when enabled")
    augmenter = contrastive_augmentor or ContrastiveAugmentor()
    model.train()
    if contrastive_head is not None:
        contrastive_head.train()
    optimizer.zero_grad()
    losses = []
    predictions, labels = [], []
    for batch_data in tqdm(loader, desc="Training", leave=False):
        x, y = batch_data[0].to(device), batch_data[1].to(device)
        view_a, view_b = augmenter.augment_v1(x), augmenter.augment_v2(x)
        with autocast(device_type=torch.device(device).type, dtype=torch.bfloat16):
            logits = model(x)
            total_loss = criterion(logits, y)
            if contrastive_weight:
                if not hasattr(model, "get_embedding"):
                    raise TypeError(
                        "Contrastive training requires a model.get_embedding(x) method"
                    )
                features_a = model.get_embedding(view_a)
                features_b = model.get_embedding(view_b)
                auxiliary = contrastive_loss(
                    contrastive_head(features_a), contrastive_head(features_b)
                )
                total_loss = total_loss + contrastive_weight * auxiliary
        if not torch.isfinite(total_loss):
            raise RuntimeError("Non-finite combined classification/contrastive loss")
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(contrastive_head.parameters()), max_norm=1.0
        )
        optimizer.step()
        optimizer.zero_grad()
        losses.append(float(total_loss.detach().cpu()))
        predictions.extend(logits.argmax(1).detach().cpu().numpy())
        labels.extend(y.detach().cpu().numpy())
    metrics = compute_metrics(labels, predictions)
    metrics["loss"] = float(np.mean(losses)) if losses else 0.0
    return metrics


def train_model(model, train_loader, val_loader, test_loader,
                epochs=100, lr=1e-3, weight_decay=5e-5,
                device="cuda", patience=25, log_mlflow=False,
                augmenter=None, mixup_prob=0.15, class_weights=None,
                accumulation_steps=4, label_smoothing=0.1):
    """
    Train with gradient accumulation, OneCycleLR scheduler, and F1-based early stopping.

    Parameters
    ----------
    accumulation_steps : int
        Gradient accumulation factor (effective batch = batch_size × accumulation_steps)
    label_smoothing : float
        Label smoothing in loss function
    patience : int
        Early stopping patience (now monitored on F1, not loss)
    """
    device = require_cuda(device)
    model.to(device)
    # Use class weights if provided
    if class_weights is not None:
        class_weights = class_weights.to(device)

    # FocalLoss WITHOUT label smoothing — smoothing breaks the focal weight computation:
    # p_t = exp(-CE_smoothed) != p(correct_class), making (1-p_t)^gamma incorrect.
    # Class weights + focal gamma alone are sufficient for minority class focus.
    criterion = FocalLoss(gamma=2.5, weight=class_weights, label_smoothing=0.0)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # ReduceLROnPlateau: Reduces LR when val_f1_macro plateaus for 3 epochs
    # Most stable for class-imbalanced training — monitors actual metric we care about
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',           # Maximize F1
        factor=0.5,           # Multiply LR by 0.5 when triggered
        patience=3,           # Wait 3 epochs before reducing
        min_lr=lr / 1000,     # Don't go below 1/1000 of initial LR
    )

    # Early stopping now monitors F1 (not loss) with min_delta for noise tolerance
    early_stopping = EarlyStopping(patience=patience, mode="max", min_delta=0.002)

    best_val_f1 = -np.inf
    best_model_state = None

    # Start MLflow run if requested
    if log_mlflow:
        import mlflow
        mlflow.start_run()

    try:
        # Diagnostic: inspect first batch to verify sampler is working
        if epochs > 0:
            first_batch = next(iter(train_loader))
            batch_labels = first_batch[1].numpy() if hasattr(first_batch[1], 'numpy') else first_batch[1]
            unique, counts = np.unique(batch_labels, return_counts=True)
            print(f"Batch class distribution: {dict(zip(unique, counts))}")
            print(f"Expected: roughly balanced (sampler active)" if len(unique) > 2 else "Warning: batch has < 3 classes")

        for epoch in range(epochs):
            train_metrics = train_epoch(
                model, train_loader, optimizer, criterion, device,
                augmenter=augmenter, mixup_prob=mixup_prob,
                accumulation_steps=accumulation_steps,
                scheduler=scheduler,
            )
            val_metrics, val_labels, val_preds = validate(model, val_loader, criterion, device)

            if log_mlflow:
                import mlflow
                mlflow.log_metrics({
                    "train_loss": train_metrics["loss"],
                    "train_accuracy": train_metrics["accuracy"],
                    "train_f1": train_metrics["f1_macro"],
                    "val_loss": val_metrics["loss"],
                    "val_accuracy": val_metrics["accuracy"],
                    "val_f1": val_metrics["f1_macro"],
                }, step=epoch)

            print(
                f"Epoch {epoch + 1}/{epochs}: "
                f"Train Loss={train_metrics['loss']:.4f}, "
                f"Train F1={train_metrics['f1_macro']:.4f}, "
                f"Val Loss={val_metrics['loss']:.4f}, "
                f"Val F1={val_metrics['f1_macro']:.4f}"
            )

            # Print per-class metrics to monitor minority classes (BE, MM)
            print("  Per-class validation metrics:")
            class_report = classification_report(
                val_labels, val_preds,
                target_names=["BE", "BM", "ME", "MM"],
                zero_division=0,
                digits=3
            )
            for line in class_report.split('\n')[:6]:  # First 6 lines: per-class metrics
                print(f"    {line}")

            # Save best model when F1 improves
            if val_metrics["f1_macro"] > best_val_f1 and np.isfinite(val_metrics["loss"]):
                best_val_f1 = val_metrics["f1_macro"]
                best_model_state = copy.deepcopy(model.state_dict())

            # Step scheduler based on validation F1 (for ReduceLROnPlateau)
            scheduler.step(val_metrics["f1_macro"])

            # Early stopping on F1 improvement (not loss)
            if early_stopping(val_metrics["f1_macro"]):
                print(f"Early stopping triggered at epoch {epoch + 1}. Best F1: {best_val_f1:.4f}")
                break
    finally:
        if log_mlflow:
            import mlflow
            mlflow.end_run()

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    if test_loader is not None:
        test_metrics, test_labels, test_preds = validate(model, test_loader, criterion, device)
        print(f"\nTest Metrics: {test_metrics}")
        print("Per-class test metrics:")
        class_report = classification_report(
            test_labels, test_preds,
            target_names=["BE", "BM", "ME", "MM"],
            zero_division=0,
            digits=3
        )
        print(class_report)
    else:
        test_metrics = {"f1_macro": best_val_f1, "best_val_f1": best_val_f1}

    return model, test_metrics
