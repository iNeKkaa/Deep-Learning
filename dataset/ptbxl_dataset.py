from pathlib import Path

import torch
from torch.utils.data import Dataset

from preprocessing.signal_processing import load_ecg_signal


class PTBXLDataset(Dataset):
    """
    Dataset that reads PTB-XL ECG files directly from disk.

    This is useful for a first check, but it can be slow because each sample
    has to read one WFDB record during training.
    """

    def __init__(
        self,
        dataframe,
        data_dir,
        metadata_columns,
        label_columns,
        sampling_rate=100,
        use_metadata=True,
        split_name="dataset",
    ):
        self.dataframe = dataframe.reset_index(drop=True)
        self.data_dir = Path(data_dir)
        self.metadata_columns = metadata_columns
        self.label_columns = label_columns
        self.sampling_rate = sampling_rate
        self.use_metadata = use_metadata
        self.split_name = split_name

        if sampling_rate == 100:
            self.filename_column = "filename_lr"
        elif sampling_rate == 500:
            self.filename_column = "filename_hr"
        else:
            raise ValueError("sampling_rate must be 100 or 500.")

        print(f"{self.split_name.capitalize()} dataset initialized with {len(self.dataframe)} samples.", flush=True)
        print(f"  Filename column: {self.filename_column}", flush=True)
        print(f"  Metadata columns: {len(self.metadata_columns)}", flush=True)
        print(f"  Label columns: {self.label_columns}", flush=True)

        if len(self.dataframe) > 0:
            first_path = self.data_dir / self.dataframe.iloc[0][self.filename_column]
            print(f"  First ECG file example: {first_path}", flush=True)
            if not first_path.with_suffix(".hea").exists() and not first_path.exists():
                print("  Warning: first ECG file was not found. Check data_dir and extracted records.", flush=True)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]

        signal_path = self.data_dir / row[self.filename_column]
        ecg_signal = load_ecg_signal(signal_path)
        ecg_signal = torch.tensor(ecg_signal, dtype=torch.float32)

        if self.use_metadata:
            metadata_values = row[self.metadata_columns].values.astype("float32")
            metadata = torch.tensor(metadata_values, dtype=torch.float32)
        else:
            metadata = torch.empty(0, dtype=torch.float32)

        label_values = row[self.label_columns].values.astype("float32")
        labels = torch.tensor(label_values, dtype=torch.float32)

        return ecg_signal, metadata, labels


class CachedPTBXLDataset(Dataset):
    """
    Dataset backed by tensors already stored in memory.

    The cache avoids reading thousands of small WFDB files at every epoch.
    It does not store a trained model, only preprocessed data tensors.
    """

    def __init__(self, ecg, metadata, labels, split_name="dataset"):
        self.ecg = ecg.float()
        self.metadata = metadata.float()
        self.labels = labels.float()
        self.split_name = split_name

        print(f"{self.split_name.capitalize()} cached dataset initialized with {len(self.labels)} samples.", flush=True)
        print(f"  ECG tensor shape: {tuple(self.ecg.shape)}", flush=True)
        print(f"  Metadata tensor shape: {tuple(self.metadata.shape)}", flush=True)
        print(f"  Label tensor shape: {tuple(self.labels.shape)}", flush=True)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.ecg[index], self.metadata[index], self.labels[index]
