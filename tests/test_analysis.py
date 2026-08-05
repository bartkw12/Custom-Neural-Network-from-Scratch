from __future__ import annotations

import numpy as np
import pytest

from pytorch_nn.analyze import compute_confusion_matrix, compute_per_class_accuracy


def test_compute_confusion_matrix_counts() -> None:
    y_true = np.array([0, 0, 1, 1, 2], dtype=np.int64)
    y_pred = np.array([0, 1, 1, 2, 2], dtype=np.int64)

    actual = compute_confusion_matrix(y_true, y_pred, num_classes=3)
    expected = np.array(
        [
            [1, 1, 0],
            [0, 1, 1],
            [0, 0, 1],
        ],
        dtype=np.int64,
    )

    assert np.array_equal(actual, expected)


def test_compute_confusion_matrix_empty_input_returns_zeros() -> None:
    y_true = np.array([], dtype=np.int64)
    y_pred = np.array([], dtype=np.int64)

    actual = compute_confusion_matrix(y_true, y_pred, num_classes=4)

    assert actual.shape == (4, 4)
    assert actual.dtype == np.int64
    assert np.count_nonzero(actual) == 0


def test_compute_confusion_matrix_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="same shape"):
        compute_confusion_matrix(np.array([0, 1]), np.array([0]), num_classes=2)

    with pytest.raises(ValueError, match="y_true contains labels"):
        compute_confusion_matrix(np.array([0, 2]), np.array([0, 1]), num_classes=2)

    with pytest.raises(ValueError, match="y_pred contains labels"):
        compute_confusion_matrix(np.array([0, 1]), np.array([0, 3]), num_classes=2)


def test_compute_per_class_accuracy_values() -> None:
    y_true = np.array([0, 0, 1, 1, 2], dtype=np.int64)
    y_pred = np.array([0, 1, 1, 2, 2], dtype=np.int64)

    actual = compute_per_class_accuracy(y_true, y_pred, num_classes=3)
    expected = np.array([0.5, 0.5, 1.0], dtype=np.float64)

    assert np.allclose(actual, expected)


def test_compute_per_class_accuracy_missing_class_is_zero() -> None:
    y_true = np.array([0, 0, 0], dtype=np.int64)
    y_pred = np.array([0, 1, 0], dtype=np.int64)

    actual = compute_per_class_accuracy(y_true, y_pred, num_classes=3)

    assert np.allclose(actual, np.array([2.0 / 3.0, 0.0, 0.0], dtype=np.float64))
