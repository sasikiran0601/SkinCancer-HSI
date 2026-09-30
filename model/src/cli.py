"""
CLI entry-point for HSI Skin Cancer Classification.

Usage:
    python -m src.cli train  --config configs/default.yaml
    python -m src.cli check
    python -m src.cli predict image.npy --model best.pth
    python -m src.cli explain image.npy --model best.pth
    python -m src.cli export --model best.pth
    python -m src.cli federate --clients 3 --rounds 10
    python -m src.cli optuna --trials 30
    python -m src.cli kfold --model self_attention --folds 5
"""
import argparse
import sys
from pathlib import Path

# Allow direct script execution (`python src/cli.py`) as well as module execution (`python -m src.cli`)
if __name__ == "__main__" and (__package__ is None or __package__ == ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "src"

from .models import (
    BaselineCNN,
    DepthwiseSeparableCNN,
    SelfAttentionCNN,
    SpatialTeacherCNN,
    MaskedAutoencoder,
)
from .augmentations import SpectralAugmenter
from .training import train_model
from .distillation import train_distillation
from .federated import federated_averaging
from .explainability import compute_band_importance
from .hsi_dataset import make_dataloader, load_class_weights, HSIPixelDataset
from .utils import require_cuda

import torch
import yaml
from torch.utils.data import Subset


def get_model(name, **kwargs):
    models = {
        "baseline": BaselineCNN,
        "depthwise_separable": DepthwiseSeparableCNN,
        "self_attention": SelfAttentionCNN,
        "teacher": SpatialTeacherCNN,
        "autoencoder": MaskedAutoencoder,
    }
    if name not in models:
        raise ValueError(f"Unknown model: {name!r}. Available: {list(models.keys())}")
    return models[name](**kwargs)


def main():
    parser = argparse.ArgumentParser(description="HSI Skin Classification CLI")
    subparsers = parser.add_subparsers(dest="command")

    # train
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("--config", type=str, default="configs/default.yaml")
    train_parser.add_argument("--model", type=str, help="Model type override")
    train_parser.add_argument("--epochs", type=int)
    train_parser.add_argument("--lr", type=float)

    # kfold
    kfold_parser = subparsers.add_parser("kfold", help="Run Stratified Group K-Fold Cross-Validation")
    kfold_parser.add_argument("--config", type=str, default="configs/default.yaml")
    kfold_parser.add_argument("--model", type=str, default="self_attention")
    kfold_parser.add_argument("--folds", type=int, default=5)
    kfold_parser.add_argument("--epochs", type=int)
    kfold_parser.add_argument("--lr", type=float)
    kfold_parser.add_argument("--output_dir", type=str, default="outputs/kfold")

    # distill
    distill_parser = subparsers.add_parser("distill", help="Train student via knowledge distillation")
    distill_parser.add_argument("--config", type=str, default="configs/default.yaml")
    distill_parser.add_argument("--teacher_model", type=str, default="best_teacher.pth")
    distill_parser.add_argument("--epochs", type=int)
    distill_parser.add_argument("--temperature", type=float, default=4.0)
    distill_parser.add_argument("--alpha", type=float, default=0.5)

    # predict
    pred_parser = subparsers.add_parser("predict", help="Run inference")
    pred_parser.add_argument("image", type=str)
    pred_parser.add_argument("--model", type=str, required=True)
    pred_parser.add_argument("--arch", type=str, default="self_attention")
    pred_parser.add_argument("--output", type=str, default="prediction.npy")

    # explain
    exp_parser = subparsers.add_parser("explain", help="Explain prediction")
    exp_parser.add_argument("image", type=str)
    exp_parser.add_argument("--model", type=str, required=True)
    exp_parser.add_argument("--arch", type=str, default="self_attention")
    exp_parser.add_argument("--class", type=int, default=3, dest="target_class")
    exp_parser.add_argument("--output", type=str, default="importance.png")

    # export
    export_parser = subparsers.add_parser("export", help="Export model to ONNX")
    export_parser.add_argument("--model", type=str, required=True)
    export_parser.add_argument("--arch", type=str, default="self_attention")
    export_parser.add_argument("--output", type=str, default="model.onnx", help="Path to output .onnx file")
    export_parser.add_argument("--format", type=str, default="onnx", choices=["onnx"])

    # federate
    fed_parser = subparsers.add_parser("federate", help="Run federated learning")
    fed_parser.add_argument("--clients", type=int, default=3)
    fed_parser.add_argument("--rounds", type=int, default=10)
    fed_parser.add_argument("--config", type=str, default="configs/default.yaml")

    # optuna
    opt_parser = subparsers.add_parser("optuna", help="Run hyperparameter search")
    opt_parser.add_argument("--trials", type=int, default=20)
    opt_parser.add_argument("--config", type=str, default="configs/default.yaml")

    # pretrain (self-supervised)
    pretrain_parser = subparsers.add_parser("pretrain", help="Self-supervised pre-training with contrastive learning")
    pretrain_parser.add_argument("--config", type=str, default="configs/pretrain.yaml")
    pretrain_parser.add_argument("--epochs", type=int, help="Override epochs from config")
    pretrain_parser.add_argument("--lr", type=float, help="Override learning rate from config")
    pretrain_parser.add_argument("--batch_size", type=int, help="Override batch size from config")

    # finetune (transfer learning)
    finetune_parser = subparsers.add_parser("finetune", help="Fine-tune pre-trained encoder on labeled data")
    finetune_parser.add_argument("--config", type=str, default="configs/finetune.yaml")
    finetune_parser.add_argument("--pretrained", type=str, default="outputs/pretrained_encoder.pth",
                                 help="Path to pre-trained encoder weights")
    finetune_parser.add_argument("--phase1_epochs", type=int, help="Override phase 1 epochs from config")
    finetune_parser.add_argument("--phase2_epochs", type=int, help="Override phase 2 epochs from config")
    finetune_parser.add_argument("--base_lr", type=float, help="Override base learning rate from config")
    finetune_parser.add_argument("--encoder_lr", type=float, help="Override encoder learning rate from config")

    # check
    subparsers.add_parser("check", help="Check environment and GPU")

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(0)

    # ==================================================================
    # TRAIN
    # ==================================================================
    if args.command == "train":
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
        if args.model:
            config["model"] = args.model
        if args.epochs:
            config["epochs"] = args.epochs
        if args.lr:
            config["lr"] = args.lr

        device = require_cuda(config.get("device", "cuda"))

        # Print device info
        print(f"\n=== TRAINING CONFIG ===")
        print(f"Device: {device}")
        if device == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"Epochs: {config['epochs']}")
        print(f"Batch Size: {config['batch_size']}")
        print(f"Learning Rate: {config['lr']}")
        print("=======================\n")

        data_dir = config.get("data_dir", "../preprocessing/outputs")
        batch_size = config["batch_size"]

        train_loader = make_dataloader("train", batch_size=batch_size, data_dir=data_dir,
                                       label_mode=config.get("label_mode", "multi"),
                                       normalization=config.get("normalization", "per_pixel"),
                                       use_sampler=True)  # WeightedRandomSampler for BE/MM
        val_loader = make_dataloader("val", batch_size=batch_size, data_dir=data_dir,
                                     label_mode=config.get("label_mode", "multi"),
                                     normalization=config.get("normalization", "per_pixel"))
        test_loader = make_dataloader("test", batch_size=batch_size, data_dir=data_dir,
                                      label_mode=config.get("label_mode", "multi"),
                                      normalization=config.get("normalization", "per_pixel"))
        class_weights = load_class_weights(
            label_mode=config.get("label_mode", "multi"), data_dir=data_dir,
            power=None,  # None = uniform gentle power=0.3 (secondary to sampler rebalancing)
        )

        model = get_model(
            config["model"],
            num_bands=config.get("num_bands", 116),
            num_classes=config.get("num_classes", 4),
            dropout=config.get("dropout", 0.4),
            attn_dim=config.get("attn_dim", 128),  # wider attention for self_attention arch
        )

        augmenter = SpectralAugmenter() if config.get("augment", True) else None

        # Print computed class weights so you can verify BE/MM receive large weights
        print(f"Class weights (BE, BM, ME, MM): {class_weights.tolist()}")

        trained_model, test_metrics = train_model(
            model, train_loader, val_loader, test_loader,
            epochs=config["epochs"],
            lr=config["lr"],
            weight_decay=config["weight_decay"],
            device=device,
            patience=config["patience"],
            log_mlflow=config.get("log_mlflow", False),
            augmenter=augmenter,
            mixup_prob=config.get("mixup_prob", 0.15),
            class_weights=class_weights,
            accumulation_steps=config.get("accumulation_steps", 4),
            label_smoothing=config.get("label_smoothing", 0.0),
        )

        # Save best model
        save_path = f"best_{config['model']}.pth"
        torch.save(trained_model.state_dict(), save_path)
        print(f"Best model saved to {save_path}")

    # ==================================================================
    # KFOLD CROSS-VALIDATION
    # ==================================================================
    elif args.command == "kfold":
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
        if args.epochs:
            config["epochs"] = args.epochs
        if args.lr:
            config["lr"] = args.lr
        from .kfold import run_kfold_cross_validation
        run_kfold_cross_validation(
            model_name=args.model,
            n_splits=args.folds,
            config=config,
            save_dir=args.output_dir,
            device=config.get("device", "auto"),
        )

    # ==================================================================
    # DISTILLATION
    # ==================================================================
    elif args.command == "distill":
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
        device = require_cuda(config.get("device", "cuda"))
        data_dir = config.get("data_dir", "../preprocessing/outputs")
        batch_size = config["batch_size"]
        epochs = args.epochs if args.epochs else config.get("epochs", 50)
        train_loader = make_dataloader("train", batch_size=batch_size, data_dir=data_dir, mode="distillation")
        val_loader = make_dataloader("val", batch_size=batch_size, data_dir=data_dir, mode="pixel")
        test_loader = make_dataloader("test", batch_size=batch_size, data_dir=data_dir, mode="pixel")
        class_weights = load_class_weights(data_dir=data_dir)
        student = get_model("baseline")
        teacher = get_model("teacher")
        if Path(args.teacher_model).exists():
            teacher.load_state_dict(torch.load(args.teacher_model, map_location=device))
        student, test_metrics = train_distillation(
            student, teacher, train_loader, val_loader, test_loader,
            temperature=args.temperature, alpha=args.alpha,
            lr=config.get("lr", 1e-3),
            weight_decay=config.get("weight_decay", 1e-5),
            epochs=epochs, patience=config.get("patience", 10),
            device=device, class_weights=class_weights,
        )
        torch.save(student.state_dict(), "best_distilled_student.pth")
        print("Distilled student saved to best_distilled_student.pth")

    # ==================================================================
    # PREDICT
    # ==================================================================
    elif args.command == "predict":
        import numpy as np
        image = np.load(args.image)  # shape (H, W, bands)
        model = get_model(args.arch)
        model.load_state_dict(torch.load(args.model, map_location="cpu"))
        model.eval()
        H, W, _ = image.shape
        predictions = np.zeros((H, W), dtype=int)
        with torch.no_grad():
            for h in range(H):
                for w in range(W):
                    pixel = torch.tensor(image[h, w], dtype=torch.float32).unsqueeze(0)
                    logits = model(pixel)
                    pred = torch.argmax(logits, dim=1).item()
                    predictions[h, w] = pred
        np.save(args.output, predictions)
        print(f"Prediction saved to {args.output}")

    # ==================================================================
    # EXPLAIN
    # ==================================================================
    elif args.command == "explain":
        import numpy as np
        import matplotlib.pyplot as plt
        image = np.load(args.image)
        center_pixel = torch.tensor(
            image[image.shape[0] // 2, image.shape[1] // 2],
            dtype=torch.float32,
        ).unsqueeze(0)
        model = get_model(args.arch)
        model.load_state_dict(torch.load(args.model, map_location="cpu"))
        importances = compute_band_importance(model, center_pixel, target_class=args.target_class)
        plt.figure(figsize=(12, 4))
        plt.bar(range(importances.shape[1]), importances[0].numpy())
        plt.xlabel("Band index")
        plt.ylabel("Importance (gradient magnitude)")
        plt.title("Spectral Band Importance")
        plt.tight_layout()
        plt.savefig(args.output, dpi=150)
        print(f"Importance plot saved to {args.output}")

    # ==================================================================
    # EXPORT
    # ==================================================================
    elif args.command == "export":
        import os
        output_path = getattr(args, "output", "model.onnx")
        model = get_model(args.arch)
        model.load_state_dict(torch.load(args.model, map_location="cpu"))
        model.eval()
        dummy_input = torch.randn(1, 116)
        print(f"Exporting '{args.arch}' architecture to ONNX ({output_path})...")
        torch.onnx.export(
            model,
            (dummy_input,),
            output_path,
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            dynamo=False,
        )
        print(f"Model exported successfully to {output_path}")

        try:
            import onnx  # type: ignore[import-not-found, import-untyped]
            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)
            size_kb = os.path.getsize(output_path) / 1024
            print(f"ONNX graph verified! Model file size: {size_kb:.1f} KB")
        except Exception as e:
            print(f"ONNX check warning: {e}")

        try:
            import onnxruntime as ort  # type: ignore[import-not-found, import-untyped]
            session = ort.InferenceSession(output_path)
            input_name = session.get_inputs()[0].name
            test_batch = torch.randn(4, 116).numpy()
            ort_outs = session.run(None, {input_name: test_batch})
            out_array = torch.as_tensor(ort_outs[0])
            print(f"ONNX Runtime test verification passed! Test batch (4, 116) -> output shape: {tuple(out_array.shape)}")
        except Exception as e:
            print(f"ONNX Runtime verification warning: {e}")

    # ==================================================================
    # FEDERATE
    # ==================================================================
    elif args.command == "federate":
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
        data_dir = config.get("data_dir", "../preprocessing/outputs")
        device = require_cuda("cuda")

        train_dataset = HSIPixelDataset(data_dir, "train")
        n_clients = args.clients
        total = len(train_dataset)
        client_sizes = [total // n_clients] * n_clients
        client_sizes[-1] += total - sum(client_sizes)
        client_datasets = []
        idx = 0
        for size in client_sizes:
            client_datasets.append(Subset(train_dataset, range(idx, idx + size)))
            idx += size
        global_model = get_model("depthwise_separable",
                                  num_classes=config.get("num_classes", 4))
        federated_averaging(global_model, client_datasets,
                            rounds=args.rounds, device=device)

    # ==================================================================
    # OPTUNA
    # ==================================================================
    elif args.command == "optuna":
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
        data_dir = config.get("data_dir", "../preprocessing/outputs")
        from .optuna_search import run_optuna
        run_optuna(data_dir, n_trials=args.trials)

    # ==================================================================
    # SELF-SUPERVISED PRE-TRAINING
    # ==================================================================
    elif args.command == "pretrain":
        from .pretrain import ContrastiveDataset, ContrastiveAugmentor, pretrain_model
        from torch.utils.data import DataLoader

        with open(args.config, "r") as f:
            config = yaml.safe_load(f)

        # Override config with command-line args
        if args.epochs:
            config["epochs"] = args.epochs
        if args.lr:
            config["lr"] = args.lr
        if args.batch_size:
            config["batch_size"] = args.batch_size

        device = require_cuda(config.get("device", "cuda"))

        # Print config
        print(f"\n{'='*70}")
        print("SELF-SUPERVISED PRE-TRAINING")
        print(f"{'='*70}")
        print(f"Device: {device}")
        print(f"GPU: {torch.cuda.get_device_name(0) if device == 'cuda' else 'None'}")
        print(f"Epochs: {config['epochs']}")
        print(f"Batch Size: {config['batch_size']}")
        print(f"Learning Rate: {config['lr']}")
        print(f"{'='*70}\n")

        # Create contrastive augmentor
        augmentor = ContrastiveAugmentor(
            noise_prob=config.get("augmentation", {}).get("noise_prob", 0.6),
            noise_std=config.get("augmentation", {}).get("noise_std", 0.05),
            dropout_prob=config.get("augmentation", {}).get("dropout_prob", 0.5),
            shift_prob=config.get("augmentation", {}).get("shift_prob", 0.4),
            scale_prob=config.get("augmentation", {}).get("scale_prob", 0.5),
        )

        # Create contrastive dataset (all splits combined for pre-training)
        data_dir = config.get("data_dir", "../preprocessing/outputs")
        train_dataset = ContrastiveDataset(
            data_dir=data_dir, split="train",
            normalization=config.get("normalization", "per_pixel"),
            augmentor=augmentor
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=0,
            drop_last=True
        )

        # Create model
        model = get_model(
            config["model"],
            num_bands=config.get("num_bands", 116),
            num_classes=config.get("num_classes", 4),
            dropout=config.get("dropout", 0.4),
            attn_dim=config.get("attn_dim", 128),
        )

        # Pre-train
        model = pretrain_model(
            model, train_loader,
            epochs=config["epochs"],
            lr=config["lr"],
            weight_decay=config.get("weight_decay", 5e-5),
            device=device,
            save_dir=config.get("save_dir", "outputs")
        )

    # ==================================================================
    # FINE-TUNING FROM PRE-TRAINED ENCODER
    # ==================================================================
    elif args.command == "finetune":
        from .finetuning import finetune_model
        from .pretrain import load_pretrained_encoder

        with open(args.config, "r") as f:
            config = yaml.safe_load(f)

        # Override config with command-line args
        if args.phase1_epochs:
            config["phase1_epochs"] = args.phase1_epochs
        if args.phase2_epochs:
            config["phase2_epochs"] = args.phase2_epochs
        if args.base_lr:
            config["base_lr"] = args.base_lr
        if args.encoder_lr:
            config["encoder_lr"] = args.encoder_lr

        device = require_cuda(config.get("device", "cuda"))

        # Print config
        print(f"\n{'='*70}")
        print("FINE-TUNING FROM PRE-TRAINED ENCODER")
        print(f"{'='*70}")
        print(f"Device: {device}")
        print(f"Pre-trained weights: {args.pretrained}")
        print(f"Phase 1 epochs: {config.get('phase1_epochs', 10)}")
        print(f"Phase 2 epochs: {config.get('phase2_epochs', 40)}")
        print(f"Base LR: {config.get('base_lr', 0.001)}")
        print(f"Encoder LR: {config.get('encoder_lr', 0.0001)}")
        print(f"{'='*70}\n")

        data_dir = config.get("data_dir", "../preprocessing/outputs")
        batch_size = config.get("batch_size", 32)

        # Create data loaders
        train_loader = make_dataloader(
            "train", batch_size=batch_size, data_dir=data_dir,
            label_mode=config.get("label_mode", "multi"),
            normalization=config.get("normalization", "per_pixel"),
            use_sampler=True
        )
        val_loader = make_dataloader(
            "val", batch_size=batch_size, data_dir=data_dir,
            label_mode=config.get("label_mode", "multi"),
            normalization=config.get("normalization", "per_pixel")
        )
        test_loader = make_dataloader(
            "test", batch_size=batch_size, data_dir=data_dir,
            label_mode=config.get("label_mode", "multi"),
            normalization=config.get("normalization", "per_pixel")
        )

        # Create model
        model = get_model(
            config["model"],
            num_bands=config.get("num_bands", 116),
            num_classes=config.get("num_classes", 4),
            dropout=config.get("dropout", 0.4),
            attn_dim=config.get("attn_dim", 128),
        )

        # Load pre-trained encoder
        model = load_pretrained_encoder(model, args.pretrained, device=device)

        # Load class weights
        class_weights = load_class_weights(
            label_mode=config.get("label_mode", "multi"), data_dir=data_dir,
            power=None
        )

        # Create augmenter
        augmenter = SpectralAugmenter() if config.get("augment", True) else None

        # Fine-tune
        model, test_metrics = finetune_model(
            model, train_loader, val_loader, test_loader,
            phase1_epochs=config.get("phase1_epochs", 10),
            phase2_epochs=config.get("phase2_epochs", 40),
            base_lr=config.get("base_lr", 0.001),
            encoder_lr=config.get("encoder_lr", 0.0001),
            weight_decay=config.get("weight_decay", 5e-5),
            device=device,
            patience=config.get("patience", 25),
            augmenter=augmenter,
            mixup_prob=config.get("mixup_prob", 0.15),
            class_weights=class_weights,
            accumulation_steps=config.get("accumulation_steps", 4)
        )

        # Save fine-tuned model
        save_path = f"best_finetuned_{config['model']}.pth"
        torch.save(model.state_dict(), save_path)
        print(f"\nFine-tuned model saved to {save_path}")
        print(f"Test F1 (Macro): {test_metrics['f1_macro']:.4f}")

    # ==================================================================
    # CHECK
    # ==================================================================
    elif args.command == "check":
        print("Python version:", sys.version)
        print("PyTorch version:", torch.__version__)
        print("CUDA available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("GPU device:", torch.cuda.get_device_name(0))
            print("GPU memory:", f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("Running on CPU.")
        for pkg in ["mlflow", "optuna", "captum"]:
            try:
                __import__(pkg)
                print(f"{pkg}: installed [OK]")
            except ImportError:
                print(f"{pkg}: NOT installed (optional)")


if __name__ == "__main__":
    main()
