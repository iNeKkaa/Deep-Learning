import numpy as np
import torch


def sigmoid_predictions(logits, threshold=0.5):
    """Convert logits into probabilities and binary predictions."""
    probabilities = torch.sigmoid(logits)

    if isinstance(threshold, (list, tuple, np.ndarray)):
        threshold = torch.tensor(threshold, dtype=probabilities.dtype)

    if isinstance(threshold, torch.Tensor):
        threshold = threshold.to(probabilities.device).view(1, -1)

    predictions = (probabilities >= threshold).float()
    return probabilities, predictions


def compute_multilabel_metrics(y_true, logits, threshold=0.5):
    """Compute simple multi-label metrics without relying on sklearn."""
    y_true = y_true.float()
    _, y_pred = sigmoid_predictions(logits, threshold=threshold)

    eps = 1e-8
    tp = (y_true * y_pred).sum(dim=0)
    fp = ((1 - y_true) * y_pred).sum(dim=0)
    fn = (y_true * (1 - y_pred)).sum(dim=0)

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)

    total_tp = tp.sum()
    total_fp = fp.sum()
    total_fn = fn.sum()

    micro_precision = (total_tp / (total_tp + total_fp + eps)).item()
    micro_recall = (total_tp / (total_tp + total_fn + eps)).item()
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall + eps)

    exact_match = (y_true == y_pred).all(dim=1).float().mean().item()

    return {
        "macro_precision": precision.mean().item(),
        "macro_recall": recall.mean().item(),
        "macro_f1": f1.mean().item(),
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "exact_match_accuracy": exact_match,
        "per_class_precision": precision.tolist(),
        "per_class_recall": recall.tolist(),
        "per_class_f1": f1.tolist(),
    }


def find_best_thresholds(y_true, logits, threshold_values=None):
    """
    Choose one threshold per label using the validation set.

    The default grid is simple on purpose. It is enough to check whether the
    fixed 0.5 threshold is limiting the macro-F1 score.
    """
    if threshold_values is None:
        threshold_values = np.arange(0.10, 0.91, 0.05)

    y_true = y_true.float()
    probabilities = torch.sigmoid(logits).float()

    best_thresholds = []
    best_f1_scores = []
    eps = 1e-8

    for class_index in range(y_true.shape[1]):
        true_class = y_true[:, class_index]
        prob_class = probabilities[:, class_index]

        best_threshold = 0.5
        best_f1 = -1.0

        for threshold in threshold_values:
            pred_class = (prob_class >= float(threshold)).float()
            tp = (true_class * pred_class).sum()
            fp = ((1 - true_class) * pred_class).sum()
            fn = (true_class * (1 - pred_class)).sum()

            precision = tp / (tp + fp + eps)
            recall = tp / (tp + fn + eps)
            f1 = 2 * precision * recall / (precision + recall + eps)

            if f1.item() > best_f1:
                best_f1 = f1.item()
                best_threshold = float(threshold)

        best_thresholds.append(best_threshold)
        best_f1_scores.append(best_f1)

    return best_thresholds, best_f1_scores


def precision_recall_curve_binary(y_true, scores):
    """Build a Precision-Recall curve for one binary label."""
    order = np.argsort(-scores)
    y_sorted = y_true[order]
    positives = y_sorted.sum()

    if positives == 0:
        return np.array([1.0]), np.array([0.0]), 0.0

    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)

    precision = tp / np.maximum(tp + fp, 1e-8)
    recall = tp / positives

    precision = np.concatenate([[1.0], precision])
    recall = np.concatenate([[0.0], recall])

    auprc = abs(np.trapezoid(precision, recall))
    return precision, recall, float(auprc)
