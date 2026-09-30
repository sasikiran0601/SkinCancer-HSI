"""
Gradient-based spectral band importance (saliency).

Uses vanilla gradients by default.  If ``captum`` is installed the user
can swap in Integrated Gradients via the CLI.
"""
import torch
import torch.nn as nn


def compute_band_importance(model, x, target_class=None):
    """
    Compute per-band saliency via input gradients.

    Parameters
    ----------
    model : nn.Module
        Trained classifier expecting input (batch, num_bands).
    x : Tensor, shape (batch, num_bands)
        Pixel vectors.  Will be cloned with requires_grad=True.
    target_class : int or None
        If None, uses the predicted class for each sample.

    Returns
    -------
    importances : Tensor, shape (batch, num_bands)
        Absolute gradient magnitude per band.
    """
    model.eval()
    x_input = x.clone().detach().requires_grad_(True)

    logits = model(x_input)

    if target_class is not None:
        target = torch.full(
            (x_input.shape[0],), target_class, dtype=torch.long, device=x_input.device
        )
    else:
        target = logits.argmax(dim=1)

    # Sum of target-class logits (one per sample) → scalar
    loss = logits.gather(1, target.unsqueeze(1)).sum()
    loss.backward()

    # Absolute gradient magnitude per band
    importances = x_input.grad.abs()
    return importances.detach()
