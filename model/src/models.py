"""
Model architectures for Hyperspectral Skin Cancer Classification.

All 1-D models expect input shape (batch, num_bands) and internally
unsqueeze to (batch, 1, num_bands) for Conv1d.

SpatialTeacherCNN expects (batch, bands, H, W) for 2-D convolutions.
MaskedAutoencoder expects (batch, num_bands) for self-supervised pretraining.
"""
import torch
import torch.nn as nn


def _get_flattened_size(model, input_shape):
    """Compute flattened feature size after convolutions."""
    with torch.no_grad():
        dummy = torch.zeros(1, *input_shape)
        out = model(dummy)
        return out.flatten(1).shape[1]


# ===================================================================
# Squeeze-and-Excitation (SE) Block for 1D
# ===================================================================
class SEBlock1d(nn.Module):
    """
    Channel attention mechanism: rescales channels based on global
    context. Applied after each depthwise-separable block.

    Parameters
    ----------
    channels : int
        Number of input channels
    reduction : int
        Reduction factor for the bottleneck (default 8)
    """
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.fc1 = nn.Linear(channels, max(channels // reduction, 1))
        self.fc2 = nn.Linear(max(channels // reduction, 1), channels)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (batch, channels, length)
        # Global average pooling: (batch, channels)
        s = x.mean(dim=-1)
        s = self.relu(self.fc1(s))
        s = self.sigmoid(self.fc2(s))
        # Channel-wise rescaling: (batch, channels, 1)
        return x * s.unsqueeze(-1)


# ===================================================================
# 1. Baseline 1-D CNN
# ===================================================================
class BaselineCNN(nn.Module):
    def __init__(self, num_bands=116, num_classes=4, dropout=0.2):
        super().__init__()
        # 1 input channel (the spectral vector) and num_bands sequence length
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)
        self.conv3 = nn.Conv1d(64, 32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(32)
        self.conv4 = nn.Conv1d(32, 32, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(32)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)

        # Compute flattened size without storing the dummy as a submodule
        dummy_seq = nn.Sequential(
            self.conv1, self.bn1, self.relu, self.pool1,
            self.conv2, self.bn2, self.relu, self.pool2,
            self.conv3, self.bn3, self.relu,
            self.conv4, self.bn4, self.relu
        )
        self.flattened_size = _get_flattened_size(dummy_seq, (1, num_bands))
        self.fc1 = nn.Linear(self.flattened_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.out = nn.Linear(32, num_classes)

    def forward(self, x):
        # x: (batch, num_bands) -> unsqueeze to (batch, 1, num_bands)
        x = x.unsqueeze(1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = x.flatten(1)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        return self.out(x)


# ===================================================================
# 2. Depthwise-Separable Conv1d block + CNN
# ===================================================================
class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__()
        self.depthwise = nn.Conv1d(in_ch, in_ch, kernel_size, padding=padding, groups=in_ch)
        self.pointwise = nn.Conv1d(in_ch, out_ch, 1)
        self.bn = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.relu(x)


class DepthwiseSeparableCNN(nn.Module):
    def __init__(self, num_bands=116, num_classes=4, dropout=0.2):
        super().__init__()
        self.block1 = DepthwiseSeparableConv1d(1, 32)   # 1 input channel
        self.pool1 = nn.MaxPool1d(2)
        self.block2 = DepthwiseSeparableConv1d(32, 64)
        self.pool2 = nn.MaxPool1d(2)
        self.block3 = DepthwiseSeparableConv1d(64, 32)
        self.block4 = DepthwiseSeparableConv1d(32, 32)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)

        dummy_seq = nn.Sequential(
            self.block1, self.pool1,
            self.block2, self.pool2,
            self.block3, self.block4
        )
        self.flattened_size = _get_flattened_size(dummy_seq, (1, num_bands))
        self.fc1 = nn.Linear(self.flattened_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.out = nn.Linear(32, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.block1(x)
        x = self.pool1(x)
        x = self.block2(x)
        x = self.pool2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = x.flatten(1)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        return self.out(x)


# ===================================================================
# 3. Self-Attention CNN
# ===================================================================
class LightweightSelfAttention(nn.Module):
    def __init__(self, embed_dim=64, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.ln = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: (batch, embed_dim, bands) -> (batch, bands, embed_dim)
        x = x.permute(0, 2, 1)
        attn_out, _ = self.attn(x, x, x)
        # Residual connection + layer norm
        x = self.ln(x + attn_out)
        return x.permute(0, 2, 1)


class SelfAttentionCNN(nn.Module):
    def __init__(self, num_bands=116, num_classes=4, dropout=0.4, attn_dim=128):
        super().__init__()
        self.block1 = DepthwiseSeparableConv1d(1, attn_dim)
        self.se1 = SEBlock1d(attn_dim)
        self.pool1 = nn.MaxPool1d(2)

        # Multi-head attention (4 heads instead of 1)
        self.attention1 = LightweightSelfAttention(embed_dim=attn_dim, num_heads=4)

        self.block2 = DepthwiseSeparableConv1d(attn_dim, 64)
        self.se2 = SEBlock1d(64)
        self.pool2 = nn.MaxPool1d(2)

        self.block3 = DepthwiseSeparableConv1d(64, 32)
        self.se3 = SEBlock1d(32)

        # Second attention layer for deeper reasoning
        self.attention2 = LightweightSelfAttention(embed_dim=32, num_heads=4)

        self.block4 = DepthwiseSeparableConv1d(32, 32)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)

        # Compute flattened size: we'll use both flatten and GAP
        dummy_seq = nn.Sequential(
            self.block1, self.se1, self.pool1,
            self.block2, self.se2, self.pool2,
            self.block3, self.se3, self.block4
        )
        conv_out_size = _get_flattened_size(dummy_seq, (1, num_bands))

        # Global Average Pooling gives us attn_dim features
        gap_size = 32  # from block4 output channels

        # FC layers: concatenate flatten + GAP, then deeper layers
        fc_input_size = conv_out_size + gap_size
        self.fc1 = nn.Linear(fc_input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.out = nn.Linear(128, num_classes)

    def forward(self, x):
        # Input: (batch, num_bands)
        x = x.unsqueeze(1)  # (batch, 1, num_bands)

        # Block 1 + SE + Attention
        x = self.block1(x)
        x = self.se1(x)
        x = self.attention1(x)  # residual + layernorm inside
        x = self.pool1(x)

        # Block 2 + SE
        x = self.block2(x)
        x = self.se2(x)
        x = self.pool2(x)

        # Block 3 + SE + Attention
        x = self.block3(x)
        x = self.se3(x)
        x = self.attention2(x)  # second attention layer

        # Block 4
        x = self.block4(x)

        # Global Average Pooling branch
        x_gap = x.mean(dim=-1)  # (batch, channels=32)

        # Flatten branch
        x_flat = x.flatten(1)

        # Concatenate both
        x = torch.cat([x_flat, x_gap], dim=1)

        # Deeper FC layers
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        return self.out(x)

    def get_embedding(self, x):
        """Return the 128-dimensional representation immediately before ``out``."""
        x = x.unsqueeze(1)
        x = self.block1(x)
        x = self.se1(x)
        x = self.attention1(x)
        x = self.pool1(x)
        x = self.block2(x)
        x = self.se2(x)
        x = self.pool2(x)
        x = self.block3(x)
        x = self.se3(x)
        x = self.attention2(x)
        x = self.block4(x)
        x_gap = x.mean(dim=-1)
        x = torch.cat([x.flatten(1), x_gap], dim=1)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        return self.relu(self.fc2(x))


# ===================================================================
# 4. Spatial Teacher CNN (2-D, for knowledge distillation)
# ===================================================================
class SpatialTeacherCNN(nn.Module):
    """2D CNN that processes spatial patches (e.g., 3x3 neighborhoods)."""
    def __init__(self, patch_size=3, in_bands=116, num_classes=4, **kwargs):
        super().__init__()
        self.conv1 = nn.Conv2d(in_bands, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2) if patch_size >= 4 else nn.Identity()
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.AdaptiveAvgPool2d((1, 1))
        self.relu = nn.ReLU(inplace=True)

        self.flattened_size = 128
        self.fc1 = nn.Linear(self.flattened_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.out = nn.Linear(64, num_classes)

    def forward(self, x):
        # x: (batch, bands, H, W)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = x.flatten(1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.out(x)


# ===================================================================
# 5. Masked Autoencoder (self-supervised pretraining)
# ===================================================================
class MaskedAutoencoder(nn.Module):
    def __init__(self, in_dim=116, latent_dim=32, mask_ratio=0.3, **kwargs):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, in_dim),
        )

    def forward(self, x):
        mask = torch.rand_like(x, dtype=torch.float32) < self.mask_ratio
        x_masked = x.clone()
        x_masked[mask] = 0.0
        latent = self.encoder(x_masked)
        reconstructed = self.decoder(latent)
        return reconstructed, mask

    def encode(self, x):
        return self.encoder(x)


class HierarchicalClassifier(nn.Module):
    """Benign/malignant gate with class-specific benign and malignant heads."""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.gate_head = nn.Linear(128, 2)
        self.benign_head = nn.Linear(128, 2)
        self.malignant_head = nn.Linear(128, 2)

    def forward(self, x, inference=True):
        embedding = self.encoder.get_embedding(x)
        gate_logits = self.gate_head(embedding)
        benign_logits = self.benign_head(embedding)
        malignant_logits = self.malignant_head(embedding)
        if not inference:
            return gate_logits, benign_logits, malignant_logits

        gate_probs = torch.softmax(gate_logits, dim=1)
        benign_probs = torch.softmax(benign_logits, dim=1)
        malignant_probs = torch.softmax(malignant_logits, dim=1)
        return torch.stack(
            (
                gate_probs[:, 0] * benign_probs[:, 0],
                gate_probs[:, 0] * benign_probs[:, 1],
                gate_probs[:, 1] * malignant_probs[:, 0],
                gate_probs[:, 1] * malignant_probs[:, 1],
            ),
            dim=1,
        )
