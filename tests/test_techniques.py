from __future__ import annotations

import numpy as np

from custom_nn.model import Layer_Dense
from custom_nn.techniques import (
    ADAM_Optimizer,
    Batch_Normalization,
    Dropout,
    Early_Stopping,
)
from tests.helpers import as_float64


def test_dropout_training_uses_inverted_scaling_and_inference_is_identity() -> None:
    inputs = as_float64(np.ones((2000, 3)))
    dropout = Dropout(dropout_rate_input=0.25, dropout_rate_hidden=0.5)

    train_outputs = dropout.forward(inputs, training=True, input_layer=True)
    keep_prob = 1.0 - 0.25
    scaled_value = 1.0 / keep_prob

    assert np.isclose(train_outputs.mean(), 1.0, atol=0.03)
    assert np.all(np.logical_or(np.isclose(dropout.mask, 0.0), np.isclose(dropout.mask, scaled_value)))

    inference_outputs = dropout.forward(inputs, training=False, input_layer=True)
    np.testing.assert_allclose(inference_outputs, inputs, rtol=0.0, atol=0.0)


def test_early_stopping_triggers_and_restores_best_layer_state() -> None:
    dense = Layer_Dense(n_inputs=2, n_neurons=2, l2_lambda=0.0)
    dense.weights = as_float64(np.array([[0.2, -0.1], [0.5, 0.3]]))
    dense.biases = as_float64(np.array([[0.0, 0.1]]))

    batch_norm = Batch_Normalization(n_neurons=2, momentum=0.185, epsilon=1e-5)
    batch_norm.gamma = as_float64(np.array([1.2, 0.7]))
    batch_norm.beta = as_float64(np.array([-0.2, 0.4]))
    batch_norm.running_mean = as_float64(np.array([0.3, -0.1]))
    batch_norm.running_variance = as_float64(np.array([1.1, 0.8]))

    layers = [dense, batch_norm]
    stopper = Early_Stopping(patience=2, min_delta=0.01)

    assert stopper.forward(1.0, layers) is False

    best_dense_weights = dense.weights.copy()
    best_dense_biases = dense.biases.copy()
    best_gamma = batch_norm.gamma.copy()
    best_beta = batch_norm.beta.copy()
    best_running_mean = batch_norm.running_mean.copy()
    best_running_variance = batch_norm.running_variance.copy()

    dense.weights += 5.0
    dense.biases -= 3.0
    batch_norm.gamma += 2.0
    batch_norm.beta -= 2.0
    batch_norm.running_mean += 4.0
    batch_norm.running_variance *= 2.0

    assert stopper.forward(0.995, layers) is False
    assert stopper.wait == 1

    dense.weights -= 1.0
    batch_norm.gamma += 1.0

    assert stopper.forward(0.997, layers) is True
    assert stopper.wait == 2

    stopper.restore_best_weights(layers)

    np.testing.assert_allclose(dense.weights, best_dense_weights, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(dense.biases, best_dense_biases, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(batch_norm.gamma, best_gamma, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(batch_norm.beta, best_beta, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(batch_norm.running_mean, best_running_mean, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(batch_norm.running_variance, best_running_variance, rtol=0.0, atol=0.0)


def test_adam_first_dense_update_matches_bias_corrected_formula() -> None:
    layer = Layer_Dense(n_inputs=2, n_neurons=2, l2_lambda=0.0)
    layer.weights = as_float64(np.array([[0.25, -0.15], [0.4, 0.05]]))
    layer.biases = as_float64(np.array([[0.1, -0.2]]))
    layer.dweights = as_float64(np.array([[0.03, -0.08], [0.12, 0.05]]))
    layer.dbiases = as_float64(np.array([[0.06, -0.04]]))

    optimizer = ADAM_Optimizer(learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-7, decay=0.0)

    expected_weights = layer.weights - 0.01 * layer.dweights / (np.sqrt(layer.dweights ** 2) + optimizer.epsilon)
    expected_biases = layer.biases - 0.01 * layer.dbiases / (np.sqrt(layer.dbiases ** 2) + optimizer.epsilon)

    optimizer.update_params(layer)

    np.testing.assert_allclose(layer.weights, expected_weights, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(layer.biases, expected_biases, rtol=1e-10, atol=1e-12)


def test_adam_batch_norm_update_matches_bias_corrected_formula() -> None:
    layer = Batch_Normalization(n_neurons=3, momentum=0.185, epsilon=1e-5)
    layer.gamma = as_float64(np.array([1.2, 0.9, 1.5]))
    layer.beta = as_float64(np.array([0.1, -0.2, 0.3]))
    layer.dgamma = as_float64(np.array([[0.05, -0.1, 0.2]]))
    layer.dbeta = as_float64(np.array([[0.02, 0.04, -0.06]]))

    optimizer = ADAM_Optimizer(learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-7, decay=0.0)

    dgamma = layer.dgamma.squeeze()
    dbeta = layer.dbeta.squeeze()
    expected_gamma = layer.gamma - 0.01 * dgamma / (np.sqrt(dgamma ** 2) + optimizer.epsilon)
    expected_beta = layer.beta - 0.01 * dbeta / (np.sqrt(dbeta ** 2) + optimizer.epsilon)

    optimizer.update_params_bn(layer)

    np.testing.assert_allclose(layer.gamma, expected_gamma, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(layer.beta, expected_beta, rtol=1e-10, atol=1e-12)


def test_adam_learning_rate_decay_progression() -> None:
    optimizer = ADAM_Optimizer(learning_rate=0.01, decay=0.1)

    optimizer.pre_update_params()
    assert np.isclose(optimizer.current_learning_rate, 0.01)

    optimizer.post_update_params()
    optimizer.pre_update_params()
    assert np.isclose(optimizer.current_learning_rate, 0.01 / 1.1)

    optimizer.post_update_params()
    optimizer.pre_update_params()
    assert np.isclose(optimizer.current_learning_rate, 0.01 / 1.2)