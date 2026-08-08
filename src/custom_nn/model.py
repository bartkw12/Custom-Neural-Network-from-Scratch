from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .config import LAMBDA


class Layer_Dense:
    """Fully connected linear layer: Z = XW + b, with optional L2 weight regularization."""

    # NN Layer initialization
    def __init__(self, n_inputs: int, n_neurons: int, l2_lambda: float = LAMBDA) -> None:
        """Initialize near-zero random weights and zero biases."""

        # Initialize weights and biases
        self.weights: NDArray = 0.01 * np.random.randn(n_inputs, n_neurons)  # set weights to random (mean 0 and variance 1)
        self.biases: NDArray = np.zeros((1, n_neurons))                      # set biases to be zero

        # L2 regularization
        self.l2_lambda = l2_lambda

        # if vanishing gradient try:
        # self.weights = np.random.randn(n_inputs, n_neurons) * np.sqrt(2 / n_inputs)

    # Forward pass
    def forward(self, inputs: NDArray) -> NDArray:
        """Compute Z = XW + b and cache inputs for the backward pass."""

        # Store inputs for backward pass
        self.inputs = inputs

        # compute linear transformation y = inputs * weights + biases
        self.output = np.dot(inputs, self.weights) + self.biases

        return self.output

    # Backpropagation
    def backward(self, dvalues: NDArray) -> NDArray:
        """Compute gradients w.r.t. weights (including L2 penalty), biases, and inputs."""

        # Weight Decay - L2 Regularization
        # Gradient of loss with respect to weights (including L2 penalty)
        self.dweights = np.dot(self.inputs.T, dvalues) + self.l2_lambda * self.weights

        # Gradient of loss with respect to biases
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        # Gradient of loss with respect to inputs
        self.dinputs = np.dot(dvalues, self.weights.T)

        return self.dinputs


# ReLU Activation
class Activation_ReLU:
    """Element-wise Rectified Linear Unit: output = max(0, input)."""

    # Forward pass
    def forward(self, inputs: NDArray) -> NDArray:
        """Apply ReLU and cache inputs for the backward pass."""

        # Remember input values
        self.inputs = inputs

        # Calculate output values from input
        self.output = np.maximum(0, inputs)

        return self.output

    # Backpropagation
    def backward(self, dvalues: NDArray) -> NDArray:
        """Pass upstream gradients through only where the pre-activation input was positive."""

        # copy to modify original variable
        # gradient of the loss w respect to the input of the ReLU function
        self.dinputs = dvalues.copy()

        # 0 gradient for input if values were neg.
        self.dinputs[self.inputs <= 0] = 0

        return self.dinputs


# Softmax Activation
class Activation_Softmax:
    """Softmax activation that converts logits into a normalized class-probability distribution.

    The ``backward`` pass is fused with cross-entropy loss, yielding the simplified
    gradient ``(y_hat - y_true) / n_samples``.
    """

    # Forward pass
    def forward(self, inputs: NDArray) -> NDArray:
        """Convert logits to class probabilities using the numerically stable softmax formula."""

        # Remember input values
        self.inputs = inputs

        # Determine probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

        # Normalize probabilities per sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.output = probabilities

        return self.output

    # Backpropagation
    def backward(self, y_true: NDArray) -> NDArray:
        """Compute the fused softmax + cross-entropy gradient: (y_hat - y_true) / n_samples."""

        # this backwards pass simplifies with the categorical cross entropy loss backwards pass
        # Gradient of the loss with respect to the inputs
        samples = y_true.shape[0]
        self.dinputs = (self.output - y_true) / samples

        return self.dinputs


# Categorical Cross-entropy Loss Function
class Categorical_Cross_entropy_loss:
    """Mean categorical cross-entropy loss with optional L2 regularization support."""

    def __init__(self, l2_lambda: float = LAMBDA) -> None:
        """Store the L2 regularization coefficient (used externally by the optimizer)."""
        self.l2_lambda = l2_lambda

    # Forward Pass
    def forward(self, y_pred: NDArray, y_true: NDArray) -> float:
        """Compute mean cross-entropy loss given softmax probabilities and one-hot labels."""

        # clip data to prevent 0 div error and not skew mean
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # one hot encoded labels from data processing
        correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        # Loss calc
        negative_log_likelihoods = -np.log(correct_confidences)

        return np.mean(negative_log_likelihoods)
