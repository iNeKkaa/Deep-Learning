import torch.nn as nn
from net.utility.blocks import ConvBlock1D


class MultivariateCNN(nn.Module):
    """
    1D-CNN feature extractor for 12-lead ECG signals.

    Input:  (batch_size, 12, signal_length)
    Output: (batch_size, feature_dim)
    """

    def __init__(self, in_channels=12, feature_dim=128):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock1D(in_channels, 32, kernel_size=7, pool=True, dropout=0.05),
            ConvBlock1D(32, 64, kernel_size=5, pool=True, dropout=0.05),
            ConvBlock1D(64, 128, kernel_size=3, pool=True, dropout=0.10),
            ConvBlock1D(128, feature_dim, kernel_size=3, pool=False, dropout=0.10),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )

    def forward(self, x):
        return self.features(x)
