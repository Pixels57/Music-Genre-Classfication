from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


def costly_misclassification_rate(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    costly_pairs: set[tuple[str, str]] | None = None,
) -> float:
    if costly_pairs is None:
        costly_pairs = {
            ("jazz_blues_classical", "electronic"),
            ("jazz_blues_classical", "hip_hop_rnb"),
            ("acoustic_folk_country", "electronic"),
            ("rock", "latin_world"),
            ("pop", "jazz_blues_classical"),
        }
    true_values = pd.Series(y_true).astype(str).reset_index(drop=True)
    pred_values = pd.Series(y_pred).astype(str).reset_index(drop=True)
    if len(true_values) == 0:
        return 0.0
    costly_count = sum(
        (truth, pred) in costly_pairs or (pred, truth) in costly_pairs
        for truth, pred in zip(true_values, pred_values, strict=False)
        if truth != pred
    )
    return float(costly_count / len(true_values))


def high_confidence_accuracy(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    probabilities: np.ndarray | None,
    threshold: float = 0.65,
) -> float:
    if probabilities is None or len(probabilities) == 0:
        return float("nan")
    confidence = np.max(probabilities, axis=1)
    mask = confidence >= threshold
    if not np.any(mask):
        return float("nan")
    return float(accuracy_score(np.asarray(y_true)[mask], np.asarray(y_pred)[mask]))


def top_k_accuracy(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray | None,
    probability_classes: list[str] | np.ndarray | None,
    k: int = 3,
) -> float:
    if probabilities is None or probability_classes is None or probabilities.shape[1] < 2:
        return float("nan")
    classes = np.asarray(probability_classes).astype(str)
    effective_k = min(k, probabilities.shape[1])
    top_indices = np.argsort(probabilities, axis=1)[:, -effective_k:]
    true_values = np.asarray(y_true).astype(str)
    hits = [truth in classes[row_indices] for truth, row_indices in zip(true_values, top_indices, strict=False)]
    return float(np.mean(hits))


def classification_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    probabilities: np.ndarray | None = None,
    probability_classes: list[str] | np.ndarray | None = None,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "costly_misclassification_rate": costly_misclassification_rate(y_true, y_pred),
        "high_confidence_accuracy": high_confidence_accuracy(y_true, y_pred, probabilities),
        "top_3_accuracy": top_k_accuracy(y_true, probabilities, probability_classes, k=3),
    }
    return metrics
