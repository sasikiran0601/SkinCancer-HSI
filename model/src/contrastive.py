"""
Contrastive learning components for self-supervised pre-training.

NT-Xent loss (Normalized Temperature-scaled Cross Entropy) for SimCLR-style training.
Contrastive augmentation pipeline for creating positive pairs.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveHead(nn.Module):
    """
    Projection head for contrastive learning.
    Maps encoder output to a latent space where contrastive loss is computed.

    Architecture: input_dim → hidden_dim → projection_dim
    Removed during fine-tuning (only encoder is kept).
    """

    def __init__(self, input_dim=256, hidden_dim=256, projection_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_dim, projection_dim)
        self.bn2 = nn.BatchNorm1d(projection_dim)

    def forward(self, x):
        """
        Parameters
        ----------
        x : Tensor
            Encoder output, shape (batch_size, input_dim)

        Returns
        -------
        Tensor
            Projected representation, shape (batch_size, projection_dim)
        """
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        return x


class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy (NT-Xent) loss.

    Computes contrastive loss between positive pairs (same spectrum, different augmentations)
    and negative pairs (different spectra).

    Formula:
        loss = -log(exp(sim(z_i, z_j) / tau) / sum_k(exp(sim(z_i, z_k) / tau)))

    where:
        z_i, z_j: embeddings of augmented views from same sample (positive pair)
        z_k: embeddings from other samples in batch (negative pairs)
        tau: temperature parameter (default 0.07)
        sim: cosine similarity
    """

    def __init__(self, batch_size, temperature=0.07, device='cuda'):
        """
        Parameters
        ----------
        batch_size : int
            Batch size (number of positive pairs)
        temperature : float
            Temperature parameter for scaling similarities (default 0.07)
        device : str
            Device to run on ('cuda' or 'cpu')
        """
        super().__init__()
        if batch_size < 2:
            raise ValueError("NTXentLoss requires a batch size of at least 2.")
        if temperature <= 0:
            raise ValueError("temperature must be greater than zero.")
        self.batch_size = batch_size
        self.temperature = temperature

    def forward(self, z_i, z_j):
        """
        Compute NT-Xent loss for a batch of positive pairs.

        Parameters
        ----------
        z_i : Tensor
            Embeddings from first augmented view, shape (batch_size, embedding_dim)
        z_j : Tensor
            Embeddings from second augmented view, shape (batch_size, embedding_dim)

        Returns
        -------
        float
            Contrastive loss value
        """
        if z_i.ndim != 2 or z_j.ndim != 2:
            raise ValueError("NTXentLoss inputs must have shape (batch, embedding_dim).")
        if z_i.shape != z_j.shape:
            raise ValueError("Positive contrastive views must have identical shapes.")
        batch_size = z_i.shape[0]
        if batch_size < 2:
            raise ValueError("NTXentLoss requires at least two samples per batch.")

        # Concatenate embeddings: (2*batch_size, embedding_dim)
        z = torch.cat([z_i, z_j], dim=0)

        # Normalize embeddings (unit norm in embedding space)
        z = F.normalize(z, dim=1)

        # Compute pairwise cosine similarity: (2N, 2N)
        similarity = torch.mm(z, z.t()) / self.temperature

        total = 2 * batch_size
        diagonal = torch.eye(total, dtype=torch.bool, device=z.device)
        similarity = similarity.masked_fill(diagonal, float("-inf"))
        positive_indices = torch.cat([
            torch.arange(batch_size, total, device=z.device),
            torch.arange(0, batch_size, device=z.device),
        ])
        return F.cross_entropy(similarity, positive_indices)


class ContrastiveAugmentor:
    """
    Augmentation pipeline for creating positive pairs in contrastive learning.

    Applies two different augmentations to the same spectrum to create views
    that should be recognized as similar (positive pair).
    """

    def __init__(self, noise_prob=0.6, noise_std=0.05,
                 dropout_prob=0.5, shift_prob=0.4, scale_prob=0.5):
        """
        Parameters
        ----------
        noise_prob : float
            Probability of adding Gaussian noise
        noise_std : float
            Standard deviation of Gaussian noise
        dropout_prob : float
            Probability of band dropout
        shift_prob : float
            Probability of spectral shift
        scale_prob : float
            Probability of spectrum scaling
        """
        self.noise_prob = noise_prob
        self.noise_std = noise_std
        self.dropout_prob = dropout_prob
        self.shift_prob = shift_prob
        self.scale_prob = scale_prob

    def augment_v1(self, x):
        """
        Augmentation view 1: Noise + Spectral Shift

        Parameters
        ----------
        x : Tensor
            Input spectrum, shape (116,)

        Returns
        -------
        Tensor
            Augmented spectrum, shape (116,)
        """
        x = x.clone()

        # Add Gaussian noise
        if torch.rand(1).item() < self.noise_prob:
            noise = torch.randn_like(x) * self.noise_std
            x = x + noise
            x = torch.clamp(x, 0, 1)

        # Spectral shift (roll bands)
        if torch.rand(1).item() < self.shift_prob:
            shift = torch.randint(-3, 4, (1,)).item()
            x = torch.roll(x, shift)

        return x

    def augment_v2(self, x):
        """
        Augmentation view 2: Band Dropout + Spectrum Scale

        Parameters
        ----------
        x : Tensor
            Input spectrum, shape (116,)

        Returns
        -------
        Tensor
            Augmented spectrum, shape (116,)
        """
        x = x.clone()

        # Band dropout (randomly zero out consecutive bands)
        if torch.rand(1).item() < self.dropout_prob:
            num_bands_to_drop = torch.randint(5, 20, (1,)).item()
            start_idx = torch.randint(0, 116 - num_bands_to_drop, (1,)).item()
            x[start_idx:start_idx + num_bands_to_drop] = 0

        # Spectrum scaling
        if torch.rand(1).item() < self.scale_prob:
            scale_factor = torch.empty(1).uniform_(0.85, 1.15).item()
            x = x * scale_factor
            x = torch.clamp(x, 0, 1)

        return x

    def create_pair(self, x):
        """
        Create a positive pair from a single spectrum.

        Parameters
        ----------
        x : Tensor
            Input spectrum, shape (116,)

        Returns
        -------
        tuple of Tensor
            (augmented_view_1, augmented_view_2), each shape (116,)
        """
        v1 = self.augment_v1(x)
        v2 = self.augment_v2(x)
        return v1, v2
