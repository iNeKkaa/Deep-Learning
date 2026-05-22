from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset.ptbxl_dataset import PTBXLDataset, CachedPTBXLDataset
from preprocessing.label_processing import add_diagnostic_superclass_labels
from preprocessing.metadata_processing import choose_metadata_columns, fit_metadata_scaler, transform_metadata
from preprocessing.signal_processing import load_ecg_signal


def split_by_ptbxl_fold(dataframe):
    """Use the official PTB-XL strat_fold split."""
    train_df = dataframe[dataframe["strat_fold"] <= 8].copy()
    val_df = dataframe[dataframe["strat_fold"] == 9].copy()
    test_df = dataframe[dataframe["strat_fold"] == 10].copy()
    return train_df, val_df, test_df


def compute_pos_weight(train_df, label_columns):
    """Compute positive class weights for BCEWithLogitsLoss."""
    labels = torch.tensor(train_df[label_columns].values, dtype=torch.float32)
    positives = labels.sum(dim=0)
    negatives = labels.shape[0] - positives
    return negatives / torch.clamp(positives, min=1.0)


def make_cache_file_name(split_name, sampling_rate, max_samples, use_metadata):
    """Create a cache file name that matches the current experiment setup."""
    sample_tag = "full" if max_samples is None else f"max{max_samples}"
    metadata_tag = "meta" if use_metadata else "nometa"
    return f"ptbxl_{sampling_rate}hz_{split_name}_{sample_tag}_{metadata_tag}.pt"


def build_or_load_cache(
    dataframe,
    data_dir,
    metadata_columns,
    label_columns,
    sampling_rate,
    use_metadata,
    cache_dir,
    split_name,
    max_samples,
    rebuild_cache=False,
):
    """
    Build or load tensor cache for one split.

    Reading PTB-XL with WFDB during every epoch is slow. This function reads the
    ECG files once, stores tensors in cache/, and then training can reuse them.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_file = cache_dir / make_cache_file_name(
        split_name=split_name,
        sampling_rate=sampling_rate,
        max_samples=max_samples,
        use_metadata=use_metadata,
    )

    if cache_file.exists() and not rebuild_cache:
        print(f"Loading {split_name} cache: {cache_file}", flush=True)
        cached = torch.load(cache_file, map_location="cpu", weights_only=False)
        return CachedPTBXLDataset(
            ecg=cached["ecg"],
            metadata=cached["metadata"],
            labels=cached["labels"],
            split_name=split_name,
        )

    print(f"Building {split_name} cache: {cache_file}", flush=True)
    print("  This step is slow only the first time because ECG files are read from disk.", flush=True)

    data_dir = Path(data_dir)
    filename_column = "filename_lr" if sampling_rate == 100 else "filename_hr"

    ecg_tensors = []
    metadata_tensors = []
    label_tensors = []

    for idx, (_, row) in enumerate(dataframe.iterrows()):
        if idx % 250 == 0:
            print(f"  {split_name}: cached {idx}/{len(dataframe)} samples", flush=True)

        signal_path = data_dir / row[filename_column]
        ecg = torch.tensor(load_ecg_signal(signal_path), dtype=torch.float32)

        if use_metadata:
            metadata_values = row[metadata_columns].values.astype("float32")
            metadata = torch.tensor(metadata_values, dtype=torch.float32)
        else:
            metadata = torch.empty(0, dtype=torch.float32)

        labels = torch.tensor(row[label_columns].values.astype("float32"), dtype=torch.float32)

        ecg_tensors.append(ecg)
        metadata_tensors.append(metadata)
        label_tensors.append(labels)

    ecg_tensor = torch.stack(ecg_tensors)
    metadata_tensor = torch.stack(metadata_tensors) if use_metadata else torch.empty((len(label_tensors), 0))
    label_tensor = torch.stack(label_tensors)

    torch.save(
        {
            "ecg": ecg_tensor,
            "metadata": metadata_tensor,
            "labels": label_tensor,
            "metadata_columns": metadata_columns,
            "label_columns": label_columns,
            "sampling_rate": sampling_rate,
        },
        cache_file,
    )
    print(f"Saved {split_name} cache to {cache_file}", flush=True)

    return CachedPTBXLDataset(ecg_tensor, metadata_tensor, label_tensor, split_name=split_name)


def get_ptbxl_loaders(
    data_dir,
    sampling_rate=100,
    batch_size=32,
    num_workers=0,
    max_samples=None,
    use_metadata=True,
    use_cache=False,
    rebuild_cache=False,
    cache_dir="cache",
):
    """Build DataLoaders for the PTB-XL experiment."""
    data_dir = Path(data_dir)
    database_path = data_dir / "ptbxl_database.csv"
    statements_path = data_dir / "scp_statements.csv"

    print(f"Checking data folder: {data_dir}", flush=True)

    if not database_path.exists():
        raise FileNotFoundError(f"Could not find {database_path}. Check --data-dir.")
    if not statements_path.exists():
        raise FileNotFoundError(f"Could not find {statements_path}. Check --data-dir.")

    print("Reading ptbxl_database.csv...", flush=True)
    dataframe = pd.read_csv(database_path)
    print(f"Raw dataframe shape: {dataframe.shape}", flush=True)

    print("Reading scp_statements.csv...", flush=True)
    scp_statements = pd.read_csv(statements_path, index_col=0)

    print("Creating diagnostic superclass labels...", flush=True)
    dataframe, label_columns = add_diagnostic_superclass_labels(dataframe, scp_statements)
    print(f"Label columns: {label_columns}", flush=True)

    metadata_columns = choose_metadata_columns(dataframe)
    print(f"Selected metadata columns: {metadata_columns}", flush=True)

    train_df, val_df, test_df = split_by_ptbxl_fold(dataframe)
    print(
        "Official fold split: "
        f"train={len(train_df)}, val={len(val_df)}, test={len(test_df)}",
        flush=True,
    )

    if max_samples is not None:
        print(f"Using subset with max_samples={max_samples}", flush=True)
        train_df = train_df.head(max_samples).copy()
        val_df = val_df.head(max(1, max_samples // 5)).copy()
        test_df = test_df.head(max(1, max_samples // 5)).copy()
        print(
            "Subset split: "
            f"train={len(train_df)}, val={len(val_df)}, test={len(test_df)}",
            flush=True,
        )

    print("Fitting metadata scaler on training split...", flush=True)
    scaler = fit_metadata_scaler(train_df, metadata_columns)
    train_df = transform_metadata(train_df, metadata_columns, scaler)
    val_df = transform_metadata(val_df, metadata_columns, scaler)
    test_df = transform_metadata(test_df, metadata_columns, scaler)

    if use_cache:
        print("Using tensor cache for datasets.", flush=True)
        train_dataset = build_or_load_cache(
            train_df,
            data_dir,
            metadata_columns,
            label_columns,
            sampling_rate,
            use_metadata,
            cache_dir,
            "train",
            max_samples,
            rebuild_cache,
        )
        val_dataset = build_or_load_cache(
            val_df,
            data_dir,
            metadata_columns,
            label_columns,
            sampling_rate,
            use_metadata,
            cache_dir,
            "validation",
            max_samples,
            rebuild_cache,
        )
        test_dataset = build_or_load_cache(
            test_df,
            data_dir,
            metadata_columns,
            label_columns,
            sampling_rate,
            use_metadata,
            cache_dir,
            "test",
            max_samples,
            rebuild_cache,
        )
    else:
        print("Creating PyTorch datasets without cache...", flush=True)
        train_dataset = PTBXLDataset(train_df, data_dir, metadata_columns, label_columns, sampling_rate, use_metadata, split_name="train")
        val_dataset = PTBXLDataset(val_df, data_dir, metadata_columns, label_columns, sampling_rate, use_metadata, split_name="validation")
        test_dataset = PTBXLDataset(test_df, data_dir, metadata_columns, label_columns, sampling_rate, use_metadata, split_name="test")

    pin_memory = torch.cuda.is_available()

    print("Creating DataLoaders...", flush=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    info = {
        "label_columns": label_columns,
        "metadata_columns": metadata_columns,
        "metadata_dim": len(metadata_columns) if use_metadata else 0,
        "pos_weight": compute_pos_weight(train_df, label_columns),
    }
    return train_loader, val_loader, test_loader, info
