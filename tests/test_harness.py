from __future__ import annotations

import numpy as np

from custom_nn import NeuralNetwork, NetworkConfig

from tests.helpers import (
    as_float64,
    build_batch_norm_test_config,
    build_dense_relu_test_config,
    build_linear_test_config,
    finite_difference_gradient,
    one_hot,
)


def test_custom_nn_imports_are_available() -> None:
    config = NetworkConfig(hidden_layers=1, hidden_units=4, input_size=3, output_size=2)
    model = NeuralNetwork(config)

    inputs = as_float64(np.ones((2, 3)))
    outputs = model.forward(inputs, training=False)

    assert outputs.shape == (2, 2)


def test_helper_utilities_return_expected_shapes() -> None:
    labels = np.array([0, 2, 1])
    encoded = one_hot(labels, num_classes=3)

    gradient = finite_difference_gradient(lambda values: float(np.sum(values ** 2)), as_float64(np.array([[1.0, 2.0]])))

    assert encoded.shape == (3, 3)
    assert gradient.shape == (1, 2)


def test_tiny_float64_batches_are_ready_for_math_checks(tiny_batch, probe_batch) -> None:
    inputs, labels = tiny_batch

    assert inputs.dtype == np.float64
    assert labels.dtype == np.float64
    assert inputs.shape == (4, 3)
    assert labels.shape == (4, 2)
    assert probe_batch.dtype == np.float64
    assert probe_batch.shape == (2, 3)


def test_reduced_config_factories_create_narrow_architectures() -> None:
    linear_config = build_linear_test_config()
    dense_relu_config = build_dense_relu_test_config()
    batch_norm_config = build_batch_norm_test_config()

    assert linear_config.get_layer_specs() == [
        {"type": "dense", "n_inputs": 3, "n_neurons": 2, "l2_lambda": 0.0},
        {"type": "softmax"},
    ]
    assert [spec["type"] for spec in dense_relu_config.get_layer_specs()] == ["dense", "relu", "dense", "softmax"]
    assert [spec["type"] for spec in batch_norm_config.get_layer_specs()] == ["dense", "batch_norm", "relu", "dense", "softmax"]


def test_reduced_fixture_config_can_drive_small_network(reduced_network_config, tiny_batch) -> None:
    model = NeuralNetwork(reduced_network_config)
    inputs, _ = tiny_batch

    outputs = model.forward(inputs, training=False)

    assert outputs.shape == (4, 2)