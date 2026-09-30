"""
Optuna hyperparameter search.

Requires: ``pip install optuna``
"""
import torch

from .models import DepthwiseSeparableCNN
from .training import train_model
from .hsi_dataset import make_dataloader, load_class_weights
from .utils import require_cuda


def objective(trial, data_dir, device):
    device = require_cuda(device)
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    mixup_prob = trial.suggest_float("mixup_prob", 0.0, 0.5)

    train_loader = make_dataloader("train", batch_size=batch_size, data_dir=data_dir, use_sampler=True)
    val_loader = make_dataloader("val", batch_size=batch_size, data_dir=data_dir)
    class_weights = load_class_weights(data_dir=data_dir, power=2.0)

    model = DepthwiseSeparableCNN(dropout=dropout)
    _, val_metrics = train_model(
        model, train_loader, val_loader, test_loader=None,
        epochs=50,
        lr=lr, weight_decay=weight_decay,
        device=device, patience=7, log_mlflow=False,
        augmenter=None, mixup_prob=mixup_prob,
        class_weights=class_weights,
    )
    return val_metrics["best_val_f1"]


def run_optuna(data_dir, n_trials=20, device="cuda"):
    import optuna

    device = require_cuda(device)
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, data_dir, device), n_trials=n_trials)
    print("Best trial:")
    print(f"  Value (val F1): {study.best_value}")
    print("  Params:", study.best_params)
    return study.best_params
