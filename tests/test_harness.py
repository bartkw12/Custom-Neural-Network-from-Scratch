from __future__ import annotations

import numpy as np

from custom_nn import NeuralNetwork, NetworkConfig

from tests.helpers import as_float64, finite_difference_gradient, one_hot


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