"""
metrics/detection.py — AUC-ROC, AUC-PR, F1 for binary deepfake detection.
Handles the edge case where only one class is present in y_true (returns nan/0).
"""

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
import warnings


def compute_detection_metrics(probs, labels) -> dict:
    """Standalone function: compute AUC-ROC, AUC-PR, F1. Guards single-class case."""
    labels = np.array(labels, dtype=int)
    probs  = np.array(probs,  dtype=float)

    unique = np.unique(labels)
    if len(unique) < 2:
        warnings.warn(
            f"Only class(es) {unique.tolist()} present in labels. "
            "Fix the dataset loader — both classes required. "
            "AUC-ROC and AUC-PR are undefined; returning NaN."
        )
        return {"auc_roc": float("nan"), "auc_pr": float("nan"), "f1": 0.0}

    preds = (probs >= 0.5).astype(int)
    return {
        "auc_roc": float(roc_auc_score(labels, probs)),
        "auc_pr":  float(average_precision_score(labels, probs)),
        "f1":      float(f1_score(labels, preds, zero_division=0)),
    }


class DetectionMetrics:
    @staticmethod
    def compute(probs, labels) -> dict:
        return compute_detection_metrics(probs, labels)
