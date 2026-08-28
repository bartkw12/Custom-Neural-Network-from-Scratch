from __future__ import annotations

from collections.abc import Callable

import numpy as np

from custom_nn import NetworkConfig

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


def tiny_classification_batch() -> tuple[Array, Array]:
    inputs = as_float64(
        np.array(
            [
                [1.25, -0.75, 0.5],
                [-0.5, 0.25, 1.0],
                [0.75, 1.5, -1.25],
                [-1.0, -0.5, 0.75],
            ]
        )
    )
    labels = one_hot(np.array([0, 1, 1, 0]), num_classes=2)
    return inputs, labels


def tiny_probe_batch() -> Array:
    return as_float64(
        np.array(
            [
                [0.1, -0.2, 0.3],
                [-0.4, 0.8, -0.6],
            ]
        )
    )


def build_reduced_network_config(**overrides: object) -> NetworkConfig:
    config = NetworkConfig(
        input_size=3,
        output_size=2,
        hidden_units=4,
        hidden_layers=1,
        batch_size=2,
        epochs=2,
        learning_rate=0.01,
        l2_lambda=0.0,
        dropout_rate_input=0.0,
        dropout_rate_hidden=0.0,
        patience=10,
        min_delta=0.0,
        seed=1234,
    )

    for key, value in overrides.items():
        setattr(config, key, value)

    return config


def build_linear_test_config(**overrides: object) -> NetworkConfig:
    layer_specs = [
        {"type": "dense", "n_inputs": 3, "n_neurons": 2, "l2_lambda": 0.0},
        {"type": "softmax"},
    ]
    return build_reduced_network_config(layer_specs=layer_specs, hidden_layers=1, **overrides)


def build_dense_relu_test_config(**overrides: object) -> NetworkConfig:
    layer_specs = [
        {"type": "dense", "n_inputs": 3, "n_neurons": 4, "l2_lambda": 0.0},
        {"type": "relu"},
        {"type": "dense", "n_inputs": 4, "n_neurons": 2, "l2_lambda": 0.0},
        {"type": "softmax"},
    ]
    return build_reduced_network_config(layer_specs=layer_specs, hidden_layers=1, **overrides)


def build_batch_norm_test_config(**overrides: object) -> NetworkConfig:
    layer_specs = [
        {"type": "dense", "n_inputs": 3, "n_neurons": 4, "l2_lambda": 0.0},
        {"type": "batch_norm", "n_neurons": 4, "momentum": 0.185, "epsilon": 1e-5},
        {"type": "relu"},
        {"type": "dense", "n_inputs": 4, "n_neurons": 2, "l2_lambda": 0.0},
        {"type": "softmax"},
    ]
    return build_reduced_network_config(layer_specs=layer_specs, hidden_layers=1, **overrides)