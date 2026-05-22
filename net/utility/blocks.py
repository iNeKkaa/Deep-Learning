import torch.nn as nn


class ConvBlock1D(nn.Module):
    """Small reusable block for 1D ECG convolution."""

    def __init__(self, in_channels, out_channels, kernel_size, pool=True, dropout=0.0):
        super().__init__()
        padding = kernel_size // 2

        layers = [
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        ]

        if pool:
            layers.append(nn.MaxPool1d(kernel_size=2))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class LinearBlock(nn.Module):
    """Basic linear block used in the metadata branch and classifier."""

    def __init__(self, in_features, out_features, dropout=0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.block(x)
