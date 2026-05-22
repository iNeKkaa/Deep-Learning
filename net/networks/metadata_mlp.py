import torch.nn as nn
from net.utility.blocks import LinearBlock


class MetadataMLP(nn.Module):
    """Feature extractor for static clinical metadata."""

    def __init__(self, metadata_dim, feature_dim=32):
        super().__init__()
        if metadata_dim <= 0:
            raise ValueError("metadata_dim must be positive for MetadataMLP.")

        self.network = nn.Sequential(
            LinearBlock(metadata_dim, 64, dropout=0.10),
            LinearBlock(64, feature_dim, dropout=0.10),
        )

    def forward(self, x):
        return self.network(x)
