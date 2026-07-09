from __future__ import annotations

import pytest

from custom_nn import NeuralNetwork

from tests.helpers import build_linear_test_config


def test_network_forward_returns_expected_output_shape(reduced_network_config, tiny_batch) -> None:
    model = NeuralNetwork(reduced_network_config)
    x_train, _ = tiny_batch

    outputs = model.forward(x_train, training=False)

    assert outputs.shape == (x_train.shape[0], reduced_network_config.output_size)


def test_train_raises_when_batch_size_exceeds_training_samples(tiny_batch) -> None:
    x_train, y_train = tiny_batch
    x_val = x_train[:2]
    y_val = y_train[:2]

    config = build_linear_test_config(batch_size=x_train.shape[0] + 1, epochs=1)
    model = NeuralNetwork(config)

    with pytest.raises(ValueError, match="batch_size is larger than the number of training samples"):
        model.train(x_train, y_train, x_val, y_val)
