import torch
import torch.nn as nn

from net.networks.multivariate_cnn import MultivariateCNN
from net.networks.metadata_mlp import MetadataMLP
from net.utility.blocks import LinearBlock


class FusionModel(nn.Module):
    """End-to-end fusion model for ECG signal and metadata."""

    def __init__(self, in_channels=12, metadata_dim=0, num_labels=5,
                 ecg_feature_dim=128, metadata_feature_dim=32, use_metadata=True):
        super().__init__()
        self.use_metadata = use_metadata and metadata_dim > 0

        self.ecg_branch = MultivariateCNN(in_channels=in_channels, feature_dim=ecg_feature_dim)

        if self.use_metadata:
            self.metadata_branch = MetadataMLP(metadata_dim=metadata_dim, feature_dim=metadata_feature_dim)
            classifier_input_dim = ecg_feature_dim + metadata_feature_dim
        else:
            self.metadata_branch = None
            classifier_input_dim = ecg_feature_dim

        self.classifier = nn.Sequential(
            LinearBlock(classifier_input_dim, 128, dropout=0.30),
            nn.Linear(128, num_labels),
        )

    def forward(self, ecg, metadata=None):
        ecg_features = self.ecg_branch(ecg)

        if self.use_metadata:
            metadata_features = self.metadata_branch(metadata)
            features = torch.cat([ecg_features, metadata_features], dim=1)
        else:
            features = ecg_features

        return self.classifier(features)


class ECGOnlyModel(nn.Module):
    """Baseline using only the 12-lead ECG signal."""

    def __init__(self, in_channels=12, num_labels=5, ecg_feature_dim=128):
        super().__init__()
        self.ecg_branch = MultivariateCNN(in_channels=in_channels, feature_dim=ecg_feature_dim)
        self.classifier = nn.Sequential(
            LinearBlock(ecg_feature_dim, 128, dropout=0.30),
            nn.Linear(128, num_labels),
        )

    def forward(self, ecg):
        return self.classifier(self.ecg_branch(ecg))


class MetadataOnlyModel(nn.Module):
    """Baseline using only static metadata."""

    def __init__(self, metadata_dim, num_labels=5, metadata_feature_dim=32):
        super().__init__()
        self.metadata_branch = MetadataMLP(metadata_dim=metadata_dim, feature_dim=metadata_feature_dim)
        self.classifier = nn.Sequential(
            LinearBlock(metadata_feature_dim, 64, dropout=0.20),
            nn.Linear(64, num_labels),
        )

    def forward(self, metadata):
        return self.classifier(self.metadata_branch(metadata))
