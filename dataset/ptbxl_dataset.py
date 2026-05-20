from pathlib import Path

import torch
from torch.utils.data import Dataset

from preprocessing.signal_processing import load_ecg_signal


class PTBXLDataset(Dataset):
    """
    PyTorch dataset for PTB-XL.

    Each sample contains:
    - a 12-lead ECG signal
    - a metadata vector
    - a multi-label target vector

    The ECG is loaded from the WFDB files listed in the PTB-XL dataframe.
    """

    def __init__(
        self,
        dataframe,
        data_dir,
        metadata_columns,
        label_columns,
        sampling_rate=100,
        use_metadata=True,
    ):
        self.dataframe = dataframe.reset_index(drop=True)
        self.data_dir = Path(data_dir)
        self.metadata_columns = metadata_columns
        self.label_columns = label_columns
        self.sampling_rate = sampling_rate
        self.use_metadata = use_metadata

        if self.sampling_rate == 100:
            self.filename_column = "filename_lr"
        elif self.sampling_rate == 500:
            self.filename_column = "filename_hr"
        else:
            raise ValueError("sampling_rate must be either 100 or 500.")

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]

        # Load the 12-lead ECG signal.
        # Expected output shape: (12, signal_length)
        relative_path = row[self.filename_column]
        signal_path = self.data_dir / relative_path
        ecg_signal = load_ecg_signal(signal_path)

        ecg_signal = torch.tensor(ecg_signal, dtype=torch.float32)

        # Load static metadata features.
        if self.use_metadata:
            metadata_values = row[self.metadata_columns].values.astype("float32")
            metadata = torch.tensor(metadata_values, dtype=torch.float32)
        else:
            metadata = torch.empty(0, dtype=torch.float32)

        # Multi-label target vector.
        # Example: [1, 0, 1, 0, 0]
        label_values = row[self.label_columns].values.astype("float32")
        labels = torch.tensor(label_values, dtype=torch.float32)

        return ecg_signal, metadata, labels
