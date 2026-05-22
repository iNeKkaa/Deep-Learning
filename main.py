import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn

from dataset.loaders import get_ptbxl_loaders
from net.networks.fusion_model import FusionModel, ECGOnlyModel, MetadataOnlyModel
from utility.metrics import compute_multilabel_metrics, find_best_thresholds
from utility.plots import save_loss_curves, save_metric_curves, save_pr_curves
from utility.save_checkpoint import save_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Train a PTB-XL multivariate ECG classifier")

    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/ptb-xl/1.0.3",
        help="Folder containing ptbxl_database.csv, scp_statements.csv and records100/ or records500/.",
    )
    parser.add_argument(
        "--download-data",
        action="store_true",
        help="Download PTB-XL from PhysioNet if the dataset is missing locally.",
    )
    parser.add_argument(
        "--sampling-rate",
        type=int,
        default=100,
        choices=[100, 500],
        help="Use the 100 Hz or 500 Hz PTB-XL records.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="fusion",
        choices=["fusion", "ecg_only", "metadata_only"],
        help="Model variant to train.",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="L2 regularization used by Adam. Set to 0 to disable it.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Use a small subset for quick debugging. Example: --max-samples 200.",
    )
    parser.add_argument(
        "--no-metadata",
        action="store_true",
        help="Disable metadata even when using the fusion model.",
    )
    parser.add_argument(
        "--debug-batches",
        action="store_true",
        help="Print batch progress during training and evaluation.",
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Use cached tensors instead of reading WFDB files during every epoch.",
    )
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Rebuild cached tensors even if cache files already exist.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="cache",
        help="Folder used to store preprocessed tensor caches.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=8,
        help="Early stopping patience based on validation macro-F1.",
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=1e-4,
        help="Minimum validation macro-F1 improvement required to reset early stopping.",
    )
    parser.add_argument(
        "--disable-early-stopping",
        action="store_true",
        help="Train for all epochs without early stopping.",
    )
    parser.add_argument(
        "--use-scheduler",
        action="store_true",
        help="Reduce learning rate when validation macro-F1 stops improving.",
    )
    parser.add_argument(
        "--tune-thresholds",
        action="store_true",
        help="Tune one decision threshold per label on the validation set before test evaluation.",
    )

    return parser.parse_args()


def check_or_download_ptbxl(data_dir, download_data=False):
    """
    Check if the local PTB-XL folder is usable.

    The automatic download is optional because PTB-XL is large. This avoids
    starting a long download without the user noticing it.
    """
    data_dir = Path(data_dir)
    database_file = data_dir / "ptbxl_database.csv"
    statements_file = data_dir / "scp_statements.csv"

    if database_file.exists() and statements_file.exists():
        print(f"PTB-XL files found in: {data_dir}", flush=True)
        return

    if download_data:
        import wfdb

        print("PTB-XL was not found locally.", flush=True)
        print("Downloading PTB-XL from PhysioNet. This may take a while.", flush=True)
        data_dir.mkdir(parents=True, exist_ok=True)
        wfdb.dl_database("ptb-xl", dl_dir=str(data_dir))
        print("Download completed.", flush=True)
        return

    raise FileNotFoundError(
        f"Could not find PTB-XL in {data_dir}.\n"
        f"Expected files:\n"
        f"  - {database_file}\n"
        f"  - {statements_file}\n\n"
        "Either move the dataset to this folder or run with --download-data.\n"
        "Manual download command:\n"
        "python -c \"import wfdb; wfdb.dl_database('ptb-xl', dl_dir='data/ptb-xl/1.0.3')\""
    )


def build_model(model_name, metadata_dim, num_labels, use_metadata):
    # All variants output one logit per diagnostic superclass.
    if model_name == "fusion":
        return FusionModel(
            in_channels=12,
            metadata_dim=metadata_dim,
            num_labels=num_labels,
            use_metadata=use_metadata,
        )

    if model_name == "ecg_only":
        return ECGOnlyModel(
            in_channels=12,
            num_labels=num_labels,
        )

    if model_name == "metadata_only":
        return MetadataOnlyModel(
            metadata_dim=metadata_dim,
            num_labels=num_labels,
        )

    raise ValueError(f"Unknown model: {model_name}")


def forward_model(model, ecg, metadata, model_name):
    """Use the correct forward call depending on the selected model."""
    if model_name == "ecg_only":
        return model(ecg)
    if model_name == "metadata_only":
        return model(metadata)
    return model(ecg, metadata)


def train_one_epoch(model, loader, criterion, optimizer, device, model_name, debug_batches=False):
    model.train()
    running_loss = 0.0

    for batch_index, (ecg, metadata, labels) in enumerate(loader):
        if debug_batches and batch_index % 10 == 0:
            print(f"  Training batch {batch_index + 1}/{len(loader)}", flush=True)

        ecg = ecg.to(device, non_blocking=True)
        metadata = metadata.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = forward_model(model, ecg, metadata, model_name)

        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)

    return running_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device, model_name, threshold=0.5, debug_batches=False):
    model.eval()
    running_loss = 0.0

    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch_index, (ecg, metadata, labels) in enumerate(loader):
            if debug_batches and batch_index % 10 == 0:
                print(f"  Evaluation batch {batch_index + 1}/{len(loader)}", flush=True)

            ecg = ecg.to(device, non_blocking=True)
            metadata = metadata.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = forward_model(model, ecg, metadata, model_name)

            loss = criterion(logits, labels)
            running_loss += loss.item() * labels.size(0)

            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    metrics = compute_multilabel_metrics(all_labels, all_logits, threshold=threshold)

    return running_loss / len(loader.dataset), metrics, all_labels, all_logits


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def main():
    args = parse_args()

    use_metadata = not args.no_metadata
    if args.model == "metadata_only":
        use_metadata = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    print("\nStarting PTB-XL pipeline", flush=True)
    print(f"Data directory: {args.data_dir}", flush=True)
    print(f"Model: {args.model}", flush=True)
    print(f"Sampling rate: {args.sampling_rate} Hz", flush=True)
    print(f"Epochs: {args.epochs}", flush=True)
    print(f"Batch size: {args.batch_size}", flush=True)
    print(f"Learning rate: {args.lr}", flush=True)
    print(f"Weight decay: {args.weight_decay}", flush=True)
    print(f"Max samples: {args.max_samples}", flush=True)
    print(f"Use metadata: {use_metadata}", flush=True)
    print(f"Use cache: {args.use_cache}", flush=True)
    print(f"Rebuild cache: {args.rebuild_cache}", flush=True)
    print(f"Cache directory: {args.cache_dir}", flush=True)
    print(f"Early stopping: {not args.disable_early_stopping}", flush=True)
    print(f"Patience: {args.patience}", flush=True)
    print(f"Tune thresholds: {args.tune_thresholds}", flush=True)

    check_or_download_ptbxl(args.data_dir, download_data=args.download_data)

    run_name = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{args.model}"
    run_dir = Path("runs") / run_name
    models_dir = run_dir / "models"
    scores_dir = run_dir / "scores"
    outputs_dir = run_dir / "outputs"
    params_dir = run_dir / "parameters"

    for folder in [models_dir, scores_dir, outputs_dir, params_dir]:
        folder.mkdir(parents=True, exist_ok=True)

    print(f"Run folder: {run_dir}", flush=True)
    save_json(vars(args), params_dir / "parameters.json")

    print("\nLoading PTB-XL metadata and creating dataloaders...", flush=True)
    train_loader, val_loader, test_loader, info = get_ptbxl_loaders(
        data_dir=args.data_dir,
        sampling_rate=args.sampling_rate,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_samples=args.max_samples,
        use_metadata=use_metadata,
        use_cache=args.use_cache,
        rebuild_cache=args.rebuild_cache,
        cache_dir=args.cache_dir,
    )

    print("Dataloaders created successfully.", flush=True)
    print(f"Train samples: {len(train_loader.dataset)} | batches: {len(train_loader)}", flush=True)
    print(f"Validation samples: {len(val_loader.dataset)} | batches: {len(val_loader)}", flush=True)
    print(f"Test samples: {len(test_loader.dataset)} | batches: {len(test_loader)}", flush=True)

    label_names = info["label_columns"]
    metadata_dim = info["metadata_dim"]
    num_labels = len(label_names)

    print(f"Labels: {label_names}", flush=True)
    print(f"Metadata dimension: {metadata_dim}", flush=True)
    print(f"Positive class weights: {info['pos_weight'].tolist()}", flush=True)

    print("\nBuilding model...", flush=True)
    model = build_model(
        model_name=args.model,
        metadata_dim=metadata_dim,
        num_labels=num_labels,
        use_metadata=use_metadata,
    ).to(device)
    print(model, flush=True)

    # pos_weight helps with class imbalance in multi-label classification.
    pos_weight = info["pos_weight"].to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = None
    if args.use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=3,
        )

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_macro_f1": [],
        "val_micro_f1": [],
    }

    best_val_macro_f1 = -1.0
    epochs_without_improvement = 0

    print("\nStarting training...", flush=True)
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}", flush=True)

        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            model_name=args.model,
            debug_batches=args.debug_batches,
        )
        print(f"Train loss: {train_loss:.4f}", flush=True)

        val_loss, val_metrics, _, _ = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            model_name=args.model,
            threshold=0.5,
            debug_batches=args.debug_batches,
        )
        print(f"Validation loss: {val_loss:.4f}", flush=True)
        print(f"Validation macro-F1: {val_metrics['macro_f1']:.4f}", flush=True)
        print(f"Validation micro-F1: {val_metrics['micro_f1']:.4f}", flush=True)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_macro_f1"].append(val_metrics["macro_f1"])
        history["val_micro_f1"].append(val_metrics["micro_f1"])

        if scheduler is not None:
            scheduler.step(val_metrics["macro_f1"])
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"Current learning rate: {current_lr:.6f}", flush=True)

        improved = val_metrics["macro_f1"] > best_val_macro_f1 + args.min_delta
        if improved:
            best_val_macro_f1 = val_metrics["macro_f1"]
            epochs_without_improvement = 0
            print("New best model found. Saving checkpoint.", flush=True)
            save_checkpoint(
                path=models_dir / "best_model.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=val_metrics,
                args=vars(args),
            )
        else:
            epochs_without_improvement += 1
            print(f"No macro-F1 improvement for {epochs_without_improvement} epoch(s).", flush=True)

        if (not args.disable_early_stopping) and epochs_without_improvement >= args.patience:
            print("Early stopping triggered.", flush=True)
            break

    print("\nSaving training curves...", flush=True)
    save_loss_curves(history, outputs_dir / "training_curves.png")
    save_metric_curves(history, outputs_dir / "validation_f1_curves.png")

    print("Loading best model for validation and test evaluation...", flush=True)
    checkpoint = torch.load(models_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    val_loss, val_metrics, val_true, val_logits = evaluate(
        model=model,
        loader=val_loader,
        criterion=criterion,
        device=device,
        model_name=args.model,
        threshold=0.5,
        debug_batches=args.debug_batches,
    )
    val_metrics["validation_loss"] = val_loss

    thresholds = [0.5 for _ in label_names]
    threshold_report = {
        "method": "fixed_0.5",
        "label_names": label_names,
        "thresholds": thresholds,
    }

    if args.tune_thresholds:
        print("Tuning thresholds on validation set...", flush=True)
        thresholds, validation_f1_by_label = find_best_thresholds(val_true, val_logits)
        threshold_report = {
            "method": "per_class_validation_f1_grid_search",
            "label_names": label_names,
            "thresholds": thresholds,
            "validation_best_f1_by_label": validation_f1_by_label,
        }
        val_metrics_tuned = compute_multilabel_metrics(val_true, val_logits, threshold=thresholds)
        threshold_report["validation_macro_f1_after_tuning"] = val_metrics_tuned["macro_f1"]
        print(f"Best thresholds: {thresholds}", flush=True)
        print(f"Validation macro-F1 after threshold tuning: {val_metrics_tuned['macro_f1']:.4f}", flush=True)

    save_json(threshold_report, scores_dir / "thresholds.json")

    print("Running final test evaluation...", flush=True)
    test_loss, test_metrics, y_true, logits = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        model_name=args.model,
        threshold=thresholds,
        debug_batches=args.debug_batches,
    )
    test_metrics["test_loss"] = test_loss
    test_metrics["thresholds"] = thresholds

    save_json(val_metrics, scores_dir / "validation_metrics_fixed_threshold.json")
    save_json(test_metrics, scores_dir / "test_metrics.json")

    torch.save(
        {
            "y_true": val_true,
            "logits": val_logits,
            "label_names": label_names,
        },
        scores_dir / "validation_predictions.pt",
    )

    torch.save(
        {
            "y_true": y_true,
            "logits": logits,
            "label_names": label_names,
        },
        scores_dir / "test_predictions.pt",
    )

    print("Saving Precision-Recall curves...", flush=True)
    save_pr_curves(
        y_true=y_true,
        logits=logits,
        label_names=label_names,
        output_path=outputs_dir / "precision_recall_curves.png",
    )

    print("\nFinal test metrics", flush=True)
    print(json.dumps(test_metrics, indent=4), flush=True)
    print(f"\nOutputs saved in: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
