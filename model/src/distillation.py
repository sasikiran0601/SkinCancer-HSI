"""
Knowledge distillation: spatial teacher (2-D CNN) → pixel student (1-D CNN).

The training loader must yield (pixel_vector, patch, label) triples
(use ``make_dataloader(..., mode='distillation')``).
"""
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast
from tqdm import tqdm

from .utils import compute_metrics, EarlyStopping, FocalLoss, require_cuda
from .training import validate


def distillation_loss(student_logits, teacher_logits, labels,
                      criterion=None, temperature=4.0, alpha=0.5):
    soft_targets = F.softmax(teacher_logits / temperature, dim=1)
    student_soft = F.log_softmax(student_logits / temperature, dim=1)
    kd_loss = F.kl_div(student_soft, soft_targets, reduction="batchmean") * (temperature ** 2)
    if criterion is not None:
        hard_loss = criterion(student_logits, labels)
    else:
        hard_loss = F.cross_entropy(student_logits, labels)
    return alpha * hard_loss + (1 - alpha) * kd_loss


def train_distillation(student, teacher, train_loader, val_loader, test_loader,
                       temperature=4.0, alpha=0.5, lr=1e-3, weight_decay=1e-5,
                       epochs=100, patience=10, device="cuda", class_weights=None):
    """
    Train a pixel-level student by distilling from a spatial teacher.

    ``train_loader`` must yield ``(x_pixel, x_patch, y)`` triples.
    ``val_loader`` / ``test_loader`` must yield ``(x, y)`` pairs (pixel mode).
    """
    device = require_cuda(device)
    teacher.eval()
    student.to(device)
    teacher.to(device)
    if class_weights is not None:
        class_weights = class_weights.to(device)
    criterion = FocalLoss(gamma=2.5, weight=class_weights)  # matches training.py gamma
    optimizer = optim.AdamW(student.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    early_stopping = EarlyStopping(patience=patience, mode="min")

    best_val_f1 = -float("inf")
    best_state = None

    for epoch in range(epochs):
        student.train()
        running_loss = 0.0
        all_preds, all_labels = [], []
        for batch_data in tqdm(train_loader, desc="Distillation", leave=False):
            x_pixel = batch_data[0].to(device)
            x_patch = batch_data[1].to(device)
            y = batch_data[2].to(device)
            sample_ids = batch_data[3] if len(batch_data) > 3 else "unknown"

            with torch.no_grad():
                teacher_logits = teacher(x_patch)
            optimizer.zero_grad()
            with autocast(device_type=torch.device(device).type, dtype=torch.bfloat16):
                student_logits = student(x_pixel)
                loss = distillation_loss(
                    student_logits, teacher_logits, y,
                    criterion=criterion, temperature=temperature, alpha=alpha
                )

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite distillation loss detected! Samples: {sample_ids}. "
                    f"Loss: {loss.item()}."
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(student_logits, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

        scheduler.step()
        train_metrics = compute_metrics(all_labels, all_preds)
        train_metrics["loss"] = running_loss / len(train_loader)

        val_metrics, _, _ = validate(student, val_loader, criterion, device)

        print(f"Epoch {epoch + 1}: Train Loss={train_metrics['loss']:.4f}, "
              f"Val Loss={val_metrics['loss']:.4f}, Val F1={val_metrics['f1_macro']:.4f}")

        if val_metrics["f1_macro"] > best_val_f1 and np.isfinite(val_metrics["loss"]):
            best_val_f1 = val_metrics["f1_macro"]
            best_state = copy.deepcopy(student.state_dict())

        if early_stopping(val_metrics["loss"]):
            print("Early stopping triggered.")
            break

    if best_state is not None:
        student.load_state_dict(best_state)
    test_metrics, _, _ = validate(student, test_loader, criterion, device)
    print(f"Distilled Student Test Metrics: {test_metrics}")
    return student, test_metrics
