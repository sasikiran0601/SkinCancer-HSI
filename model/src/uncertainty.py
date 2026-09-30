"""
Monte Carlo Dropout uncertainty estimation.

Only Dropout layers are set to train mode; BatchNorm stays frozen
in eval mode to avoid unstable batch-statistics during inference.
"""
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import chi2


def mc_dropout_predict(model, x, num_samples=30):
    """
    Per-pixel uncertainty via Monte Carlo dropout.

    Parameters
    ----------
    model : nn.Module
        Must contain at least one nn.Dropout layer.
    x : Tensor, shape (batch, num_bands)
    num_samples : int
        Number of stochastic forward passes.

    Returns
    -------
    mean_probs : ndarray, shape (batch, num_classes)
    variance   : ndarray, shape (batch, num_classes)
    entropy    : ndarray, shape (batch,)
    """
    # Only enable Dropout, keep BatchNorm in eval mode
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()

    predictions = []
    for _ in range(num_samples):
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            predictions.append(probs.cpu().numpy())

    # Restore full eval mode
    model.eval()

    predictions = np.array(predictions)  # (samples, batch, classes)
    mean_probs = predictions.mean(axis=0)
    variance = predictions.var(axis=0)
    entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-8), axis=1)

    return mean_probs, variance, entropy


class TemperatureScaler(nn.Module):
    """Fit one validation-set temperature for post-hoc logit calibration."""

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits):
        return logits / self.temperature.clamp(min=0.05)

    def calibrate(self, logits, labels, lr=0.01, max_iter=200):
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        criterion = nn.CrossEntropyLoss()

        def closure():
            optimizer.zero_grad()
            loss = criterion(self(logits), labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        return float(self.temperature.item())


class OODDetector:
    """MSP and class-conditional Mahalanobis OOD detector."""

    def __init__(self, msp_threshold=0.50):
        self.msp_threshold = msp_threshold
        self.class_means = {}
        self.precision_matrix = None

    def fit(self, embeddings, labels):
        embeddings = np.asarray(embeddings, dtype=np.float64)
        labels = np.asarray(labels)
        classes = np.unique(labels)
        pooled_covariance = np.zeros((embeddings.shape[1], embeddings.shape[1]))
        for cls in classes:
            class_embeddings = embeddings[labels == cls]
            mean = class_embeddings.mean(axis=0)
            self.class_means[int(cls)] = mean
            centered = class_embeddings - mean
            pooled_covariance += centered.T @ centered
        pooled_covariance /= max(len(labels) - len(classes), 1)
        pooled_covariance += np.eye(embeddings.shape[1]) * 1e-6
        self.precision_matrix = np.linalg.pinv(pooled_covariance)

    def mahalanobis_score(self, embedding):
        if self.precision_matrix is None or not self.class_means:
            raise RuntimeError("OODDetector.fit must be called before scoring.")
        embedding = np.asarray(embedding, dtype=np.float64)
        return min(
            float(
                (embedding - mean)
                @ self.precision_matrix
                @ (embedding - mean)
            )
            for mean in self.class_means.values()
        )

    def is_ood_msp(self, softmax_probs):
        return float(np.max(softmax_probs)) < self.msp_threshold

    def is_ood_mahalanobis(self, embedding, alpha=0.01):
        return self.mahalanobis_score(embedding) > chi2.ppf(
            1.0 - alpha, df=len(embedding)
        )
