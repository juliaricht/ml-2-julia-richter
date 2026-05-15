"""Evaluation utilities für binäre Anomalie-Erkennung (imbalanced dataset).

Model gibt kontinuierlich Anomaly-scores aus;  threshold wird ausgewähltauf dem
validation split (by maximizing F1); threshold angewendet auf  test
split. Threshold-independent Metriken (ROC-AUC, PR-AUC) werden ausgegeben.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Tuple

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class Metrics:
    roc_auc: float
    pr_auc: float
    threshold: float
    precision: float
    recall: float
    f1: float
    tn: int
    fp: int
    fn: int
    tp: int

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def best_f1_threshold(y_true: np.ndarray, scores: np.ndarray) -> Tuple[float, float]:
    """Ausgabe von (threshold, f1) doe F1 im geg. Split maximieren.

    precision/recall-curve to enumerate candidate thresholds; this is
    O(n log n) and considers every score that actually appears.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    # precision/recall arrays have len(thresholds)+1 elements; the last
    # entry corresponds to recall=0 with no threshold — drop it.
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    if f1.size == 0:
        return 0.5, 0.0
    best = int(np.argmax(f1))
    return float(thresholds[best]), float(f1[best])


def evaluate(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> Metrics:
    y_pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return Metrics(
        roc_auc=float(roc_auc_score(y_true, scores)),
        pr_auc=float(average_precision_score(y_true, scores)),
        threshold=float(threshold),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        tn=int(tn),
        fp=int(fp),
        fn=int(fn),
        tp=int(tp),
    )
