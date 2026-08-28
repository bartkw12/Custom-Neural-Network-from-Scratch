from __future__ import annotations

import numpy as np

from custom_nn.model import (
    Activation_ReLU,
    Activation_Softmax,
    Categorical_Cross_entropy_loss,
    Layer_Dense,
)
from custom_nn.techniques import Batch_Normalization
from tests.helpers import as_float64, finite_difference_gradient, relative_error


def _dense_objective(layer: Layer_Dense, inputs: np.ndarray, upstream_gradient: np.ndarray) -> float:
    outputs = layer.forward(inputs)
    return float(np.sum(outputs * upstream_gradient))


def _relu_objective(inputs: np.ndarray, upstream_gradient: np.ndarray) -> float:
    activation = Activation_ReLU()
    outputs = activation.forward(inputs)
    return float(np.sum(outputs * upstream_gradient))


def _softmax_cross_entropy_objective(logits: np.ndarray, y_true: np.ndarray) -> float:
    activation = Activation_Softmax()
    loss = Categorical_Cross_entropy_loss(l2_lambda=0.0)
    probabilities = activation.forward(logits)
    return float(loss.forward(probabilities, y_true))


def _batch_norm_objective(layer: Batch_Normalization, inputs: np.ndarray, upstream_gradient: np.ndarray) -> float:
    outputs = layer.forward(inputs, training=True)
    return float(np.sum(outputs * upstream_gradient))


def test_dense_backward_matches_numeric_gradients_without_regularization(float_tolerance) -> None:
    inputs = as_float64(
        np.array(
            [
                [0.2, -0.4, 0.7],
                [1.1, 0.3, -0.6],
                [-0.8, 0.5, 0.9],
            ]
        )
    )
    upstream_gradient = as_float64(
        np.array(
            [
                [0.5, -0.2],
                [-0.3, 0.4],
                [0.1, 0.6],
            ]
        )
    )

    layer = Layer_Dense(n_inputs=3, n_neurons=2, l2_lambda=0.0)
    layer.weights = as_float64(
        np.array(
            [
                [0.15, -0.25],
                [0.35, 0.05],
                [-0.45, 0.2],
            ]
        )
    )
    layer.biases = as_float64(np.array([[0.1, -0.2]]))

    layer.forward(inputs)
    layer.backward(upstream_gradient)

    numeric_dweights = finite_difference_gradient(
        lambda values: _dense_objective(_layer_with(layer, weights=values), inputs, upstream_gradient),
        layer.weights,
    )
    numeric_dbiases = finite_difference_gradient(
        lambda values: _dense_objective(_layer_with(layer, biases=values), inputs, upstream_gradient),
        layer.biases,
    )
    numeric_dinputs = finite_difference_gradient(
        lambda values: _dense_objective(layer, values, upstream_gradient),
        inputs,
    )

    assert relative_error(layer.dweights, numeric_dweights) < float_tolerance["rtol"]
    assert relative_error(layer.dbiases, numeric_dbiases) < float_tolerance["rtol"]
    assert relative_error(layer.dinputs, numeric_dinputs) < float_tolerance["rtol"]


def test_dense_backward_adds_expected_l2_weight_term() -> None:
    inputs = as_float64(
        np.array(
            [
                [0.5, -1.0, 0.25],
                [1.5, 0.75, -0.5],
            ]
        )
    )
    upstream_gradient = as_float64(
        np.array(
            [
                [0.2, -0.3],
                [0.4, 0.1],
            ]
        )
    )
    l2_lambda = 0.3

    layer = Layer_Dense(n_inputs=3, n_neurons=2, l2_lambda=l2_lambda)
    layer.weights = as_float64(
        np.array(
            [
                [0.12, -0.18],
                [0.07, 0.22],
                [-0.31, 0.09],
            ]
        )
    )
    layer.biases = as_float64(np.array([[0.0, 0.05]]))

    layer.forward(inputs)
    layer.backward(upstream_gradient)

    expected_without_regularization = inputs.T @ upstream_gradient
    expected_regularization_term = l2_lambda * layer.weights

    np.testing.assert_allclose(
        layer.dweights,
        expected_without_regularization + expected_regularization_term,
        rtol=1e-10,
        atol=1e-12,
    )


def test_relu_backward_matches_numeric_gradient(float_tolerance) -> None:
    inputs = as_float64(
        np.array(
            [
                [0.8, -1.2, 0.35],
                [-0.45, 1.1, -0.9],
                [1.5, 0.6, -0.25],
            ]
        )
    )
    upstream_gradient = as_float64(
        np.array(
            [
                [0.4, -0.7, 0.2],
                [0.9, 0.3, -0.5],
                [-0.6, 0.8, 0.1],
            ]
        )
    )

    activation = Activation_ReLU()
    activation.forward(inputs)
    activation.backward(upstream_gradient)

    numeric_dinputs = finite_difference_gradient(
        lambda values: _relu_objective(values, upstream_gradient),
        inputs,
    )

    assert relative_error(activation.dinputs, numeric_dinputs) < float_tolerance["rtol"]


def test_softmax_cross_entropy_backward_matches_numeric_gradient(float_tolerance) -> None:
    logits = as_float64(
        np.array(
            [
                [1.2, -0.4, 0.3],
                [-0.7, 0.9, 0.1],
                [0.25, -1.1, 1.4],
            ]
        )
    )
    y_true = as_float64(
        np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
    )

    activation = Activation_Softmax()
    probabilities = activation.forward(logits)
    loss = Categorical_Cross_entropy_loss(l2_lambda=0.0)
    loss.forward(probabilities, y_true)
    activation.backward(y_true)

    numeric_dlogits = finite_difference_gradient(
        lambda values: _softmax_cross_entropy_objective(values, y_true),
        logits,
    )

    assert relative_error(activation.dinputs, numeric_dlogits) < float_tolerance["rtol"]


def test_batch_norm_backward_matches_numeric_gradients(float_tolerance) -> None:
    inputs = as_float64(
        np.array(
            [
                [0.4, -1.2, 0.7],
                [1.1, 0.3, -0.5],
                [-0.8, 1.5, 0.2],
                [0.9, -0.4, 1.3],
            ]
        )
    )
    upstream_gradient = as_float64(
        np.array(
            [
                [0.6, -0.1, 0.3],
                [-0.2, 0.5, -0.4],
                [0.7, 0.2, -0.6],
                [-0.3, 0.4, 0.1],
            ]
        )
    )

    layer = Batch_Normalization(n_neurons=3, momentum=0.185, epsilon=1e-5)
    layer.gamma = as_float64(np.array([1.2, 0.7, 1.5]))
    layer.beta = as_float64(np.array([-0.3, 0.2, 0.05]))

    layer.forward(inputs, training=True)
    layer.backward(upstream_gradient)

    numeric_dgamma = finite_difference_gradient(
        lambda values: _batch_norm_objective(_batch_norm_with(layer, gamma=values), inputs, upstream_gradient),
        layer.gamma,
    )
    numeric_dbeta = finite_difference_gradient(
        lambda values: _batch_norm_objective(_batch_norm_with(layer, beta=values), inputs, upstream_gradient),
        layer.beta,
    )
    numeric_dinputs = finite_difference_gradient(
        lambda values: _batch_norm_objective(_batch_norm_with(layer), values, upstream_gradient),
        inputs,
    )

    assert relative_error(layer.dgamma.squeeze(), numeric_dgamma) < float_tolerance["rtol"]
    assert relative_error(layer.dbeta.squeeze(), numeric_dbeta) < float_tolerance["rtol"]
    assert relative_error(layer.dinputs, numeric_dinputs) < float_tolerance["rtol"]


def test_batch_norm_forward_updates_running_stats_and_inference_uses_them() -> None:
    inputs = as_float64(
        np.array(
            [
                [0.4, -1.2, 0.7],
                [1.1, 0.3, -0.5],
                [-0.8, 1.5, 0.2],
                [0.9, -0.4, 1.3],
            ]
        )
    )
    probe_inputs = as_float64(
        np.array(
            [
                [2.0, -3.0, 1.0],
                [-1.0, 2.5, -0.75],
            ]
        )
    )

    layer = Batch_Normalization(n_neurons=3, momentum=0.185, epsilon=1e-5)
    layer.gamma = as_float64(np.array([1.2, 0.7, 1.5]))
    layer.beta = as_float64(np.array([-0.3, 0.2, 0.05]))

    running_mean_before = layer.running_mean.copy()
    running_variance_before = layer.running_variance.copy()

    layer.forward(inputs, training=True)

    batch_mean = np.mean(inputs, axis=0, keepdims=True)
    batch_var = np.var(inputs, axis=0, keepdims=True)
    expected_running_mean = layer.momentum * running_mean_before + (1 - layer.momentum) * batch_mean
    expected_running_variance = layer.momentum * running_variance_before + (1 - layer.momentum) * batch_var

    np.testing.assert_allclose(layer.running_mean, expected_running_mean, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(layer.running_variance, expected_running_variance, rtol=1e-10, atol=1e-12)

    inference_outputs = layer.forward(probe_inputs, training=False)
    expected_inference_outputs = layer.gamma * (
        (probe_inputs - layer.running_mean) / np.sqrt(layer.running_variance + layer.epsilon)
    ) + layer.beta

    np.testing.assert_allclose(inference_outputs, expected_inference_outputs, rtol=1e-10, atol=1e-12)


def _layer_with(layer: Layer_Dense, *, weights: np.ndarray | None = None, biases: np.ndarray | None = None) -> Layer_Dense:
    probe = Layer_Dense(
        n_inputs=layer.weights.shape[0],
        n_neurons=layer.weights.shape[1],
        l2_lambda=layer.l2_lambda,
    )
    probe.weights = layer.weights.copy() if weights is None else as_float64(weights)
    probe.biases = layer.biases.copy() if biases is None else as_float64(biases)
    return probe


def _batch_norm_with(
    layer: Batch_Normalization,
    *,
    gamma: np.ndarray | None = None,
    beta: np.ndarray | None = None,
) -> Batch_Normalization:
    probe = Batch_Normalization(
        n_neurons=layer.gamma.shape[0],
        momentum=layer.momentum,
        epsilon=layer.epsilon,
    )
    probe.gamma = layer.gamma.copy() if gamma is None else as_float64(gamma)
    probe.beta = layer.beta.copy() if beta is None else as_float64(beta)
    probe.running_mean = np.asarray(layer.running_mean, dtype=np.float64).copy()
    probe.running_variance = np.asarray(layer.running_variance, dtype=np.float64).copy()
    return probe