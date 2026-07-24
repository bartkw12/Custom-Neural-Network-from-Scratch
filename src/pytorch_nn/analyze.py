"""Analysis helpers for visualization workflows."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


IntArray = NDArray[np.int64]
FloatArray = NDArray[np.float64]


def compute_confusion_matrix(
    y_true: IntArray,
    y_pred: IntArray,
    num_classes: int = 10,
) -> IntArray:
    """Compute a confusion matrix for integer class labels."""
    true_labels = np.asarray(y_true, dtype=np.int64).reshape(-1)
    predicted_labels = np.asarray(y_pred, dtype=np.int64).reshape(-1)

    if true_labels.shape != predicted_labels.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    if num_classes < 1:
        raise ValueError("num_classes must be at least 1")

    if true_labels.size == 0:
        return np.zeros((num_classes, num_classes), dtype=np.int64)

    if np.any(true_labels < 0) or np.any(true_labels >= num_classes):
        raise ValueError("y_true contains labels outside the valid class range")

    if np.any(predicted_labels < 0) or np.any(predicted_labels >= num_classes):
        raise ValueError("y_pred contains labels outside the valid class range")

    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(confusion_matrix, (true_labels, predicted_labels), 1)
    return confusion_matrix


def compute_per_class_accuracy(
    y_true: IntArray,
    y_pred: IntArray,
    num_classes: int = 10,
) -> FloatArray:
    """Compute per-class accuracy, returning 0.0 for classes with no true samples."""
    confusion_matrix = compute_confusion_matrix(y_true, y_pred, num_classes=num_classes)
    class_totals = confusion_matrix.sum(axis=1)
    correct_predictions = np.diag(confusion_matrix).astype(np.float64)

    per_class_accuracy = np.zeros(num_classes, dtype=np.float64)
    non_empty_classes = class_totals > 0
    per_class_accuracy[non_empty_classes] = (
        correct_predictions[non_empty_classes] / class_totals[non_empty_classes]
    )
    return per_class_accuracy