"""
Self-supervised pre-training using contrastive learning (SimCLR-style).

Pre-trains encoder on unlabeled HSI data using NT-Xent loss.
Learns to recognize same spectrum under different augmentations.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from .models import SelfAttentionCNN
from .contrastive import NTXentLoss, ContrastiveHead, ContrastiveAugmentor
from .utils import require_cuda


class ContrastiveDataset(Dataset):
    """
    Dataset for contrastive pre-training (self-supervised).

    Returns pairs of augmented views from same spectrum.
    No labels required — learns from the spectrum structure alone.
    """

    def __init__(self, data_dir="../preprocessing/outputs", split="train",
                 normalization="per_pixel", augmentor=None):
        """
        Parameters
        ----------
        data_dir : str
            Path to preprocessed data
        split : str
            'train', 'val', or 'test'
        normalization : str
            Normalization method ('per_pixel', etc.)
        augmentor : ContrastiveAugmentor
            Augmentation pipeline for creating positive pairs
        """
        from .hsi_dataset import HSIPixelDataset

        # Load pixel dataset (no labels needed)
        self.pixel_dataset = HSIPixelDataset(
            data_dir=data_dir, split=split, label_mode="multi",
            normalization=normalization
        )

        self.augmentor = augmentor or ContrastiveAugmentor()

    def __len__(self):
        return len(self.pixel_dataset)

    def __getitem__(self, index):
        """
        Returns two augmented views of the same spectrum (positive pair).

        Returns
        -------
        tuple of Tensor
            (view_1, view_2), each shape (116,)
        """
        x, _, _ = self.pixel_dataset[index]  # Ignore labels

        # Create two different augmented versions
        v1, v2 = self.augmentor.create_pair(x)

        return v1, v2


def pretrain_epoch(model, contrastive_head, train_loader, optimizer,
                   criterion, device, augmentor=None):
    """
    One epoch of self-supervised pre-training.

    Parameters
    ----------
    model : nn.Module
        Encoder model (SelfAttentionCNN)
    contrastive_head : nn.Module
        Projection head for contrastive space
    train_loader : DataLoader
        Contrastive dataset loader
    optimizer : optim.Optimizer
        Optimizer for encoder + head
    criterion : nn.Module
        NT-Xent loss
    device : str
        Device to train on ('cuda' or 'cpu')
    augmentor : ContrastiveAugmentor
        Optional augmentor (can be None if already in DataLoader)

    Returns
    -------
    dict
        Metrics: {'loss': average_loss}
    """
    device = require_cuda(device)
    model.train()
    contrastive_head.train()

    total_loss = 0.0
    num_batches = 0

    for batch_idx, batch_data in enumerate(tqdm(train_loader, desc="Pre-training", leave=False)):
        v1, v2 = batch_data
        v1, v2 = v1.to(device), v2.to(device)

        # Forward pass: encode both views
        with torch.autocast(device_type=torch.device(device).type, dtype=torch.bfloat16):
            z1 = model(v1)  # (batch_size, embedding_dim)
            z2 = model(v2)

            # Project to contrastive space
            p1 = contrastive_head(z1)  # (batch_size, projection_dim)
            p2 = contrastive_head(z2)

            # Compute contrastive loss
            loss = criterion(p1, p2)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(model.parameters()) + list(contrastive_head.parameters()),
            max_norm=1.0,
        )
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches
    return {"loss": avg_loss}


def pretrain_model(model, train_loader, epochs=100, lr=0.001,
                   weight_decay=5e-5, device="cuda", save_dir="outputs"):
    """
    Pre-train encoder using self-supervised contrastive learning.

    Parameters
    ----------
    model : nn.Module
        Encoder to pre-train (SelfAttentionCNN)
    train_loader : DataLoader
        Contrastive dataset loader
    epochs : int
        Number of pre-training epochs
    lr : float
        Learning rate
    weight_decay : float
        Weight decay regularization
    device : str
        Device to train on
    save_dir : str
        Directory to save checkpoints

    Returns
    -------
    nn.Module
        Pre-trained encoder (ready for fine-tuning)
    """
    device = require_cuda(device)
    import os
    os.makedirs(save_dir, exist_ok=True)

    model.to(device)

    # Remove classification head to get encoder-only model
    # SelfAttentionCNN outputs logits via self.out, but we need embeddings from fc2
    original_out = model.out
    model.out = nn.Identity()  # Replace with identity to get fc2 output (128 dims)

    # Add contrastive head
    contrastive_head = ContrastiveHead(input_dim=128, hidden_dim=128, projection_dim=128)
    contrastive_head.to(device)

    # Optimizer for encoder + head
    params = list(model.parameters()) + list(contrastive_head.parameters())
    optimizer = optim.AdamW(params, lr=float(lr), weight_decay=float(weight_decay))

    # NT-Xent loss
    batch_size = train_loader.batch_size
    if batch_size is None or batch_size < 2:
        raise ValueError("Pretraining requires a DataLoader batch_size of at least 2.")
    criterion = NTXentLoss(batch_size=batch_size, temperature=0.07, device=device)

    # Pre-training loop
    print(f"\n{'='*60}")
    print("SELF-SUPERVISED PRE-TRAINING")
    print(f"{'='*60}")
    print(f"Encoder: {type(model).__name__}")
    print(f"Epochs: {epochs}")
    print(f"Batch Size: {batch_size}")
    print(f"Learning Rate: {lr}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")

    best_loss = float('inf')
    best_path = os.path.join(save_dir, "pretrained_encoder_best.pth")
    for epoch in range(epochs):
        metrics = pretrain_epoch(
            model, contrastive_head, train_loader, optimizer,
            criterion, device
        )

        loss = metrics["loss"]

        print(f"Epoch {epoch + 1}/{epochs}: Loss={loss:.4f}")

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            ckpt_path = os.path.join(save_dir, f"pretrained_epoch_{epoch + 1}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"  Checkpoint saved: {ckpt_path}")

        # Save best model
        if loss < best_loss:
            best_loss = loss
            best_path = os.path.join(save_dir, "pretrained_encoder_best.pth")
            torch.save(model.state_dict(), best_path)

    # Final save
    final_path = os.path.join(save_dir, "pretrained_encoder.pth")
    torch.save(model.state_dict(), final_path)
    print(f"\nPre-training complete!")
    print(f"Final model saved: {final_path}")
    print(f"Best model saved: {best_path}")

    # Restore original classification head
    model.out = original_out

    return model


def load_pretrained_encoder(model, pretrained_path, device="cuda"):
    """
    Load pre-trained encoder weights into model, skipping the classification head.

    Parameters
    ----------
    model : nn.Module
        Model to load weights into
    pretrained_path : str
        Path to pre-trained weights
    device : str
        Device to load on

    Returns
    -------
    nn.Module
        Model with pre-trained encoder weights (head untouched)
    """
    print(f"\nLoading pre-trained encoder from: {pretrained_path}")
    state_dict = torch.load(pretrained_path, map_location=device)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    if not isinstance(state_dict, dict):
        raise ValueError("Pre-trained checkpoint must contain a model state dictionary.")

    # Remove classification head weights (they were Identity during pretraining)
    # Keep only encoder weights (block*, se*, attention*, fc1, fc2, etc.)
    filtered_state_dict = {
        k: v for k, v in state_dict.items()
        if not (k == "out" or k.startswith("out."))
    }
    incompatible = model.load_state_dict(filtered_state_dict, strict=False)
    unexpected = [key for key in incompatible.unexpected_keys if not key.startswith("out.")]
    if unexpected:
        raise RuntimeError(
            "Pre-trained checkpoint is incompatible with the requested model. "
            f"Unexpected encoder keys: {unexpected[:5]}"
        )
    print("Pre-trained encoder loaded successfully!")
    print(f"  Loaded {len(filtered_state_dict)} encoder weights")
    print(f"  Classification head (self.out) kept as initialized")
    return model
