import argparse
import json
from datetime import datetime
from pathlib import Path

import torch

from utility.metrics import compute_multilabel_metrics, find_best_thresholds
from utility.plots import save_pr_curves


def parse_args():
    parser = argparse.ArgumentParser(description="Soft voting benchmark for PTB-XL models")
    parser.add_argument("--ecg-run", type=str, required=True, help="Run folder of the ECG-only model.")
    parser.add_argument("--metadata-run", type=str, required=True, help="Run folder of the metadata-only model.")
    parser.add_argument("--output-dir", type=str, default=None, help="Folder where the soft voting results are saved.")
    parser.add_argument("--tune-thresholds", action="store_true", help="Tune class thresholds on validation predictions.")
    parser.add_argument("--tune-alpha", action="store_true", help="Tune the ECG/model weight alpha on validation predictions.")
    return parser.parse_args()


def load_predictions(run_dir, split_name):
    path = Path(run_dir) / "scores" / f"{split_name}_predictions.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path}. The model run must save {split_name}_predictions.pt."
        )
    return torch.load(path, map_location="cpu", weights_only=False)


def combine_probabilities(ecg_logits, metadata_logits, alpha):
    """
    Soft voting combines probabilities from two already trained models.

    alpha = 1.0 uses only ECG predictions.
    alpha = 0.0 uses only metadata predictions.
    """
    ecg_probs = torch.sigmoid(ecg_logits)
    metadata_probs = torch.sigmoid(metadata_logits)
    return alpha * ecg_probs + (1.0 - alpha) * metadata_probs


def logits_from_probabilities(probabilities):
    """Convert probabilities back to logits for reusing the same metric functions."""
    probabilities = torch.clamp(probabilities, min=1e-6, max=1 - 1e-6)
    return torch.log(probabilities / (1.0 - probabilities))


def main():
    args = parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else Path("runs") / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_soft_voting")
    scores_dir = output_dir / "scores"
    outputs_dir = output_dir / "outputs"
    scores_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    print(f"Soft voting output folder: {output_dir}", flush=True)

    ecg_val = load_predictions(args.ecg_run, "validation")
    meta_val = load_predictions(args.metadata_run, "validation")
    ecg_test = load_predictions(args.ecg_run, "test")
    meta_test = load_predictions(args.metadata_run, "test")

    label_names = ecg_test["label_names"]
    y_val = ecg_val["y_true"]
    y_test = ecg_test["y_true"]

    if label_names != meta_test["label_names"]:
        raise ValueError("The two runs do not use the same labels.")

    best_alpha = 0.5
    best_val_macro_f1 = -1.0

    if args.tune_alpha:
        print("Tuning soft-voting alpha on validation set...", flush=True)
        for alpha in [i / 10 for i in range(0, 11)]:
            val_probs = combine_probabilities(ecg_val["logits"], meta_val["logits"], alpha)
            val_logits = logits_from_probabilities(val_probs)
            val_metrics = compute_multilabel_metrics(y_val, val_logits, threshold=0.5)
            print(f"  alpha={alpha:.1f} | val macro-F1={val_metrics['macro_f1']:.4f}", flush=True)
            if val_metrics["macro_f1"] > best_val_macro_f1:
                best_val_macro_f1 = val_metrics["macro_f1"]
                best_alpha = alpha
    else:
        print("Using default alpha=0.5 for soft voting.", flush=True)

    val_probs = combine_probabilities(ecg_val["logits"], meta_val["logits"], best_alpha)
    test_probs = combine_probabilities(ecg_test["logits"], meta_test["logits"], best_alpha)
    val_logits = logits_from_probabilities(val_probs)
    test_logits = logits_from_probabilities(test_probs)

    thresholds = [0.5 for _ in label_names]
    if args.tune_thresholds:
        print("Tuning thresholds on validation soft-voting predictions...", flush=True)
        thresholds, validation_f1_by_label = find_best_thresholds(y_val, val_logits)
        print(f"Best thresholds: {thresholds}", flush=True)
    else:
        validation_f1_by_label = None

    val_metrics = compute_multilabel_metrics(y_val, val_logits, threshold=thresholds)
    test_metrics = compute_multilabel_metrics(y_test, test_logits, threshold=thresholds)

    results = {
        "ecg_run": str(args.ecg_run),
        "metadata_run": str(args.metadata_run),
        "alpha": best_alpha,
        "thresholds": thresholds,
        "label_names": label_names,
        "validation_best_f1_by_label": validation_f1_by_label,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
    }

    with open(scores_dir / "soft_voting_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    torch.save(
        {
            "y_true": y_test,
            "logits": test_logits,
            "label_names": label_names,
            "alpha": best_alpha,
            "thresholds": thresholds,
        },
        scores_dir / "soft_voting_test_predictions.pt",
    )

    save_pr_curves(
        y_true=y_test,
        logits=test_logits,
        label_names=label_names,
        output_path=outputs_dir / "soft_voting_precision_recall_curves.png",
    )

    print("\nSoft voting test metrics", flush=True)
    print(json.dumps(test_metrics, indent=4), flush=True)
    print(f"\nOutputs saved in: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
