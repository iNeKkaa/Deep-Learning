import numpy as np
import wfdb


def load_ecg_signal(record_path):
    """
    Load one PTB-XL ECG record.

    WFDB gives the signal as (signal_length, 12), while Conv1d expects
    (channels, signal_length). We transpose it after loading.
    """
    signal, _ = wfdb.rdsamp(str(record_path))
    signal = signal.astype("float32")
    signal = signal.T
    return normalize_per_lead(signal)


def normalize_per_lead(signal):
    """Normalize each ECG lead independently."""
    mean = signal.mean(axis=1, keepdims=True)
    std = signal.std(axis=1, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return (signal - mean) / std
