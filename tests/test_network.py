from __future__ import annotations

import numpy as np
import pytest

from custom_nn import NeuralNetwork

from tests.helpers import as_float64, build_linear_test_config, one_hot


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


def test_two_epoch_training_reduces_loss_on_tiny_synthetic_data() -> None:
    x_train = as_float64(
        np.array(
            [
                [2.0, 1.0, 0.5],
                [1.5, 0.8, 0.2],
                [1.7, 1.2, 0.4],
                [1.2, 0.6, 0.3],
                [-2.0, -1.1, -0.4],
                [-1.6, -0.9, -0.2],
                [-1.8, -1.3, -0.5],
                [-1.3, -0.7, -0.1],
            ]
        )
    )
    y_indices = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_train = one_hot(y_indices, num_classes=2)

    x_val = x_train.copy()
    y_val = y_train.copy()

    config = build_linear_test_config(epochs=2, batch_size=4, learning_rate=0.05)
    model = NeuralNetwork(config)

    pre_train_loss = model.evaluate(x_train, y_train)["loss"]
    history = model.train(x_train, y_train, x_val, y_val)
    post_train_loss = model.evaluate(x_train, y_train)["loss"]

    assert len(history["train_loss"]) == 2
    assert post_train_loss < pre_train_loss
