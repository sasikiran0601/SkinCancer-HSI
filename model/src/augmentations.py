"""
Spectral augmentation strategies for 1-D pixel vectors.

All methods operate on tensors of shape (batch, num_bands) and
return the same shape.
"""
import torch


class SpectralAugmenter:
    """
    Lightweight augmenter applied *inside* the training loop.

    Usage:
        aug = SpectralAugmenter()
        x = aug.add_noise(x)
        x = aug.band_dropout(x)
        x, (y_a, y_b, lam) = aug.mixup(x, y)
    """

    def __init__(self, noise_sigma=0.01, drop_prob=0.1, mixup_alpha=0.2, max_shift=3, scale_range=(0.85, 1.15)):
        self.noise_sigma = noise_sigma
        self.drop_prob = drop_prob
        self.mixup_alpha = mixup_alpha
        self.max_shift = max_shift
        self.scale_range = scale_range

    # ------------------------------------------------------------------
    def add_noise(self, x, sigma=None):
        """Add zero-mean Gaussian noise to each pixel vector."""
        sigma = sigma or self.noise_sigma
        noise = torch.randn_like(x) * sigma
        return (x + noise).clamp(0.0, 1.0)

    # ------------------------------------------------------------------
    def band_dropout(self, x, drop_prob=None):
        """Randomly zero-out entire spectral bands across the batch."""
        drop_prob = drop_prob or self.drop_prob
        # mask shape: (1, num_bands) — same bands dropped for whole batch
        mask = (torch.rand(1, x.shape[1], device=x.device) > drop_prob).float()
        return x * mask

    # ------------------------------------------------------------------
    def mixup(self, x, y, alpha=None):
        """
        Standard mixup augmentation (Zhang et al., 2018).

        Returns
        -------
        x_mixed : (batch, num_bands)
        (y_a, y_b, lam) : tuple used by mixup_criterion
        """
        alpha = alpha or self.mixup_alpha
        lam = torch.distributions.Beta(alpha, alpha).sample().item() if alpha > 0 else 1.0
        batch_size = x.size(0)
        index = torch.randperm(batch_size, device=x.device)
        x_mixed = lam * x + (1 - lam) * x[index]
        return x_mixed, (y, y[index], lam)

    # ------------------------------------------------------------------
    def spectral_shift(self, x, max_shift=None):
        """
        Randomly shift the spectrum left or right by up to max_shift bands.
        Simulates sensor calibration drift or wavelength registration errors.

        Parameters
        ----------
        x : Tensor of shape (batch, num_bands)
        max_shift : int or None
            Max shift in bands (default from __init__)

        Returns
        -------
        Tensor of same shape with rolled bands
        """
        max_shift = max_shift or self.max_shift
        # Random shift per sample
        shift = torch.randint(-max_shift, max_shift + 1, (x.size(0),), device=x.device)
        # Apply shift per sample in batch
        x_shifted = torch.zeros_like(x)
        for i in range(x.size(0)):
            x_shifted[i] = torch.roll(x[i], shifts=shift[i].item(), dims=-1)
        return x_shifted

    # ------------------------------------------------------------------
    def spectrum_scale(self, x, scale_range=None):
        """
        Randomly scale the entire spectrum by a uniform factor.
        Simulates illumination variation or sensor gain drift.

        Parameters
        ----------
        x : Tensor of shape (batch, num_bands)
        scale_range : tuple (min, max) or None
            Scale factor range (default from __init__)

        Returns
        -------
        Tensor of same shape, clamped to [0, 1]
        """
        scale_range = scale_range or self.scale_range
        # Per-sample scale factor
        scale = torch.empty(x.size(0), 1, device=x.device).uniform_(*scale_range)
        return (x * scale).clamp(0.0, 1.0)
