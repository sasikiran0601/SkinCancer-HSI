"""
Fine-tuning module for transfer learning from pre-trained encoder.

Two-phase fine-tuning:
  Phase 1: Freeze encoder, train classification head (10 epochs)
  Phase 2: Unfreeze encoder, fine-tune full model (40 epochs)
"""
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report

from .training import train_epoch, validate, EarlyStopping
from .utils import compute_metrics, require_cuda


def freeze_encoder(model):
    """Freeze all encoder parameters while keeping classification head (self.out) trainable."""
    for param in model.parameters():
        param.requires_grad = False

    if hasattr(model, 'out'):
        for param in model.out.parameters():
            param.requires_grad = True
    else:
        for name, param in model.named_parameters():
            if 'out' in name or 'fc' in name or 'linear' in name:
                param.requires_grad = True


def unfreeze_encoder(model, freeze_early=False):
    """
    Unfreeze encoder parameters for fine-tuning.

    Parameters
    ----------
    model : nn.Module
        Model with frozen encoder
    freeze_early : bool
        If True, only freeze early layers, unfreeze later layers
    """
    for param in model.parameters():
        param.requires_grad = True

    # Optionally freeze early layers to preserve learned representations
    if freeze_early:
        for name, param in model.named_parameters():
            if 'block1' in name or 'block2' in name:
                param.requires_grad = False


def get_layer_wise_lr(model, base_lr=0.001, encoder_lr=0.0001):
    """
    Create parameter groups with different learning rates.

    Encoder uses lower LR to prevent catastrophic forgetting.
    Classification head uses higher LR for faster adaptation.

    Parameters
    ----------
    model : nn.Module
        Model with encoder + classification head
    base_lr : float
        Learning rate for classification head
    encoder_lr : float
        Learning rate for encoder

    Returns
    -------
    list of dict
        Parameter groups for optimizer
    """
    param_groups = []

    # Classification head (high LR): self.out
    if hasattr(model, 'out'):
        head_params = list(model.out.parameters())
    else:
        head_params = [p for n, p in model.named_parameters() if 'out' in n or 'fc' in n or 'linear' in n]
    head_param_ids = set(id(p) for p in head_params)

    # Encoder (low LR): all other pre-trained layers
    encoder_params = [p for p in model.parameters() if id(p) not in head_param_ids]

    if head_params:
        param_groups.append({'params': head_params, 'lr': base_lr})

    if encoder_params:
        param_groups.append({'params': encoder_params, 'lr': encoder_lr})

    return param_groups


def finetune_model(model, train_loader, val_loader, test_loader,
                   phase1_epochs=10, phase2_epochs=40,
                   base_lr=0.001, encoder_lr=0.0001,
                   weight_decay=5e-5, device="cuda", patience=25,
                   augmenter=None, mixup_prob=0.15, class_weights=None,
                   accumulation_steps=4):
    """
    Two-phase fine-tuning with encoder transfer learning.

    Phase 1: Freeze encoder, train classification head (phase1_epochs)
    Phase 2: Unfreeze encoder, fine-tune full model (phase2_epochs)

    Parameters
    ----------
    model : nn.Module
        Pre-trained model (encoder already loaded)
    train_loader : DataLoader
        Labeled training data
    val_loader : DataLoader
        Labeled validation data
    test_loader : DataLoader
        Labeled test data
    phase1_epochs : int
        Epochs for phase 1 (frozen encoder)
    phase2_epochs : int
        Epochs for phase 2 (unfrozen encoder)
    base_lr : float
        Learning rate for classification head
    encoder_lr : float
        Learning rate for encoder (lower to prevent forgetting)
    weight_decay : float
        L2 regularization
    device : str
        Device to train on
    patience : int
        Early stopping patience
    augmenter : Augmenter
        Data augmentation
    mixup_prob : float
        Mixup probability
    class_weights : Tensor
        Class weights for loss function
    accumulation_steps : int
        Gradient accumulation steps

    Returns
    -------
    tuple of (model, metrics)
        Fine-tuned model and test metrics
    """
    import os
    from .utils import FocalLoss

    device = require_cuda(device)
    model.to(device)
    if class_weights is not None:
        class_weights = class_weights.to(device)

    # gamma=1.5 for fine-tuning: softer focal focus than pre-training (gamma=2.0).
    # High gamma in fine-tuning causes instability when the pre-trained encoder
    # already provides good features — moderate focusing is sufficient.
    criterion = FocalLoss(gamma=1.5, weight=class_weights, label_smoothing=0.0)

    best_val_f1 = -np.inf
    best_model_state = None
    checkpoint_dir = "outputs/finetune_checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print("TWO-PHASE FINE-TUNING FROM PRE-TRAINED ENCODER")
    print(f"{'='*70}")
    print(f"Phase 1 (Head warmup, frozen encoder): {phase1_epochs} epochs | Head LR: {base_lr:.6f}")
    print(f"Phase 2 (Full fine-tuning, unfrozen):  {phase2_epochs} epochs | Encoder LR: {encoder_lr:.6f}, Head LR: {base_lr:.6f}")
    print(f"Early stopping patience: {patience} epochs (monitors Macro Val F1)")
    print(f"{'='*70}\n")

    # ====================================================================
    # PHASE 1: Frozen Encoder — Train Classification Head Only
    # ====================================================================
    print(f"\n{'='*70}")
    print(f"PHASE 1: FROZEN ENCODER — Training Classification Head ({phase1_epochs} Epochs)")
    print(f"{'='*70}")
    freeze_encoder(model)

    if hasattr(model, 'out'):
        head_params = list(model.out.parameters())
    else:
        head_params = [p for n, p in model.named_parameters() if 'out' in n or 'fc' in n or 'linear' in n]

    optimizer_phase1 = optim.AdamW(head_params, lr=base_lr, weight_decay=float(weight_decay))
    scheduler_phase1 = optim.lr_scheduler.CosineAnnealingLR(
        optimizer_phase1,
        T_max=phase1_epochs,
        eta_min=base_lr / 100,
    )

    print("Encoder is FROZEN. Only classification head (self.out) weights are being updated.")
    print(f"Head LR: {base_lr:.6f} with CosineAnnealingLR (T_max={phase1_epochs}, eta_min={base_lr/100:.2e})")
    print(f"{'='*70}\n")

    for epoch in range(phase1_epochs):
        train_metrics = train_epoch(
            model, train_loader, optimizer_phase1, criterion, device,
            augmenter=augmenter, mixup_prob=mixup_prob,
            accumulation_steps=accumulation_steps, scheduler=None
        )

        val_metrics, val_labels, val_preds = validate(model, val_loader, criterion, device)

        print(
            f"Phase 1 - Epoch {epoch + 1}/{phase1_epochs}: "
            f"Train F1={train_metrics['f1_macro']:.4f}, "
            f"Val F1={val_metrics['f1_macro']:.4f}"
        )

        # Print per-class metrics
        print("  Per-class validation metrics:")
        class_report = str(
            classification_report(
                val_labels, val_preds,
                target_names=["BE", "BM", "ME", "MM"],
                zero_division=0, digits=3
            )
        )
        for line in class_report.split('\n')[:6]:
            print(f"    {line}")

        # Save best model across both phases
        if val_metrics["f1_macro"] > best_val_f1 and np.isfinite(val_metrics["loss"]):
            best_val_f1 = val_metrics["f1_macro"]
            best_model_state = copy.deepcopy(model.state_dict())
            torch.save(best_model_state, os.path.join(checkpoint_dir, "best_finetune.pth"))
            print(f"  ✓ New best Val F1={best_val_f1:.4f} — checkpoint saved")

        # Periodic checkpoint every 5 epochs in Phase 1
        if (epoch + 1) % 5 == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"finetune_phase1_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), ckpt_path)

        scheduler_phase1.step()

    # Load best weights achieved during Phase 1 before starting Phase 2
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\nLoaded best Phase 1 checkpoint (Val F1={best_val_f1:.4f}) before starting Phase 2.")

    # ====================================================================
    # PHASE 2: Unfrozen Encoder — Fine-Tune Full Model
    # ====================================================================
    print(f"\n{'='*70}")
    print(f"PHASE 2: FULL FINE-TUNING — Unfrozen Encoder + Classification Head ({phase2_epochs} Epochs)")
    print(f"{'='*70}")
    unfreeze_encoder(model)

    param_groups = get_layer_wise_lr(model, base_lr=base_lr, encoder_lr=encoder_lr)
    optimizer_phase2 = optim.AdamW(param_groups, weight_decay=float(weight_decay))

    scheduler_phase2 = optim.lr_scheduler.CosineAnnealingLR(
        optimizer_phase2,
        T_max=phase2_epochs,
        eta_min=encoder_lr / 100,
    )

    early_stopping = EarlyStopping(patience=patience, mode="max", min_delta=0.001)
    if best_val_f1 > -np.inf:
        early_stopping.best = best_val_f1

    print("Encoder is UNFROZEN. Training all layers with layer-wise learning rates.")
    print(f"Encoder LR: {encoder_lr:.6f}, Head LR: {base_lr:.6f}")
    print(f"Max Phase 2 Epochs: {phase2_epochs}, Early stopping patience: {patience}")
    print(f"Scheduler: CosineAnnealingLR (T_max={phase2_epochs}, eta_min={encoder_lr/100:.2e})")
    print(f"{'='*70}\n")

    for epoch in range(phase2_epochs):
        train_metrics = train_epoch(
            model, train_loader, optimizer_phase2, criterion, device,
            augmenter=augmenter, mixup_prob=mixup_prob,
            accumulation_steps=accumulation_steps, scheduler=None
        )

        val_metrics, val_labels, val_preds = validate(model, val_loader, criterion, device)

        print(
            f"Phase 2 - Epoch {epoch + 1}/{phase2_epochs}: "
            f"Train F1={train_metrics['f1_macro']:.4f}, "
            f"Val F1={val_metrics['f1_macro']:.4f}"
        )

        # Print per-class metrics
        print("  Per-class validation metrics:")
        class_report = str(
            classification_report(
                val_labels, val_preds,
                target_names=["BE", "BM", "ME", "MM"],
                zero_division=0, digits=3
            )
        )
        for line in class_report.split('\n')[:6]:
            print(f"    {line}")

        # Save best model continuously across both phases
        if val_metrics["f1_macro"] > best_val_f1 and np.isfinite(val_metrics["loss"]):
            best_val_f1 = val_metrics["f1_macro"]
            best_model_state = copy.deepcopy(model.state_dict())
            torch.save(best_model_state, os.path.join(checkpoint_dir, "best_finetune.pth"))
            print(f"  ✓ New best Val F1={best_val_f1:.4f} — checkpoint saved")

        # Periodic checkpoint every 5 epochs in Phase 2
        if (epoch + 1) % 5 == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"finetune_phase2_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), ckpt_path)

        # Step scheduler
        scheduler_phase2.step()

        # Early stopping check
        if early_stopping(val_metrics["f1_macro"]):
            print(f"Early stopping at Phase 2 epoch {epoch + 1}. Best F1: {best_val_f1:.4f}")
            break

    # ====================================================================
    # EVALUATION ON TEST SET
    # ====================================================================
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    test_metrics, test_labels, test_preds = validate(model, test_loader, criterion, device)

    print(f"\n{'='*70}")
    print("TEST METRICS")
    print(f"{'='*70}")
    print(f"Test F1 (Macro): {test_metrics['f1_macro']:.4f}")
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print()
    print("Per-class test metrics:")
    class_report = classification_report(
        test_labels, test_preds,
        target_names=["BE", "BM", "ME", "MM"],
        zero_division=0, digits=3
    )
    print(class_report)

    return model, test_metrics
