import matplotlib.pyplot as plt
import torch
from utility.metrics import precision_recall_curve_binary


def save_loss_curves(history, output_path):
    """Save the loss curves for a training run."""
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="Train loss")
    plt.plot(epochs, history["val_loss"], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and validation loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_metric_curves(history, output_path):
    """Save validation macro-F1 and micro-F1 curves."""
    epochs = range(1, len(history["val_macro_f1"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["val_macro_f1"], label="Validation macro-F1")
    plt.plot(epochs, history["val_micro_f1"], label="Validation micro-F1")
    plt.xlabel("Epoch")
    plt.ylabel("F1-score")
    plt.title("Validation F1-score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_pr_curves(y_true, logits, label_names, output_path):
    """Save Precision-Recall curves for all labels."""
    probabilities = torch.sigmoid(logits).numpy()
    y_true = y_true.numpy()

    plt.figure(figsize=(8, 6))

    for class_index, label_name in enumerate(label_names):
        precision, recall, auprc = precision_recall_curve_binary(
            y_true=y_true[:, class_index],
            scores=probabilities[:, class_index],
        )
        plt.plot(recall, precision, label=f"{label_name} (AUPRC={auprc:.3f})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
