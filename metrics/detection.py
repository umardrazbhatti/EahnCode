"""
metrics/detection.py — AUC-ROC, AUC-PR, F1 for binary deepfake detection.
Handles the edge case where only one class is present in y_true (returns nan/0).
"""

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
import warnings


class DetectionMetrics:
    @staticmethod
    def compute(probs, labels) -> dict:
        labels = np.array(labels, dtype=int)
        probs  = np.array(probs,  dtype=float)

        unique = np.unique(labels)
        if len(unique) < 2:
            warnings.warn(
                f"Only one class ({unique}) present in labels. "
                "AUC-ROC is undefined; returning NaN."
            )
            auc_roc = float("nan")
            auc_pr  = float(labels.mean() if labels[0] == 1 else 1 - labels.mean())
        else:
            auc_roc = float(roc_auc_score(labels, probs))
            auc_pr  = float(average_precision_score(labels, probs))

        preds = (probs >= 0.5).astype(int)
        f1    = float(f1_score(labels, preds, zero_division=0))

        return {"auc_roc": auc_roc, "auc_pr": auc_pr, "f1": f1}
