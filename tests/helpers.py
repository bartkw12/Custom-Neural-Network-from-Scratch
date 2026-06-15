from __future__ import annotations

from collections.abc import Callable

import numpy as np


Array = np.ndarray


def as_float64(array: Array) -> Array:
    return np.asarray(array, dtype=np.float64)


def one_hot(labels: Array, num_classes: int) -> Array:
    encoded = np.zeros((labels.shape[0], num_classes), dtype=np.float64)
    encoded[np.arange(labels.shape[0]), labels] = 1.0
    return encoded


def relative_error(actual: Array, expected: Array) -> float:
    numerator = np.linalg.norm(actual - expected)
    denominator = np.linalg.norm(actual) + np.linalg.norm(expected) + 1e-12
    return float(numerator / denominator)


def finite_difference_gradient(
    objective: Callable[[Array], float],
    values: Array,
    epsilon: float = 1e-6,
) -> Array:
    gradient = np.zeros_like(values, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)

    iterator = np.ndindex(values.shape)
    for index in iterator:
        shifted_plus = values.copy()
        shifted_minus = values.copy()
        shifted_plus[index] += epsilon
        shifted_minus[index] -= epsilon
        gradient[index] = (objective(shifted_plus) - objective(shifted_minus)) / (2.0 * epsilon)

    return gradient