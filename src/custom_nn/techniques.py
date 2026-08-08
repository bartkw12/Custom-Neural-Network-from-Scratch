from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from .config import BETA1, BETA2, DECAY, DROPOUT_RATE_HIDDEN, DROPOUT_RATE_INPUT, EPSILON, EPSILON_A, LEARNING_RATE, MIN_DELTA, MOMENTUM, PATIENCE


# ADAM Optimizer
class ADAM_Optimizer:
    """Adam optimizer with optional per-batch learning-rate decay.

    Maintains first- and second-moment estimates (momentum and RMS cache) per
    parameter and applies bias correction before each weight update.
    """

    def __init__(self, learning_rate: float = LEARNING_RATE, beta1: float = BETA1, beta2: float = BETA2, epsilon: float = EPSILON_A, decay: float = DECAY) -> None:
        """Initialize Adam hyperparameters and set the iteration counter to zero."""
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.decay = decay
        self.iterations = 0

    def pre_update_params(self) -> None:
        """Decay the learning rate before each batch update, if a decay factor is set."""
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

            # print(f"The learning rate currently is: {self.current_learning_rate}")

    def update_params(self, layer: Any) -> None:
        """Apply one Adam update step to the ``weights`` and ``biases`` of a dense layer."""

        # If layer does not contain cache arrays, create them filled with zeros
        # arrays store the exponentially decaying averages of gradients (momentum) and squared gradients (cache)
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)

        # first momentum estimate for weights and biases
        layer.weight_momentums = self.beta1 * layer.weight_momentums + (1 - self.beta1) * layer.dweights
        layer.bias_momentums = self.beta1 * layer.bias_momentums + (1 - self.beta1) * layer.dbiases

        # apply bias correction to first moment estimates
        weight_momentums_corrected = layer.weight_momentums / (1 - self.beta1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / (1 - self.beta1 ** (self.iterations + 1))

        # second moment estimate - cache update w squared gradients
        layer.weight_cache = self.beta2 * layer.weight_cache + (1 - self.beta2) * layer.dweights ** 2
        layer.bias_cache = self.beta2 * layer.bias_cache + (1 - self.beta2) * layer.dbiases**2

        # apply bias correction to second moment estimates
        weight_cache_corrected = layer.weight_cache / (1 - self.beta2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / (1 - self.beta2 ** (self.iterations + 1))

        # update weights and biases w corrected momentum and cache
        layer.weights += -self.current_learning_rate * weight_momentums_corrected / (np.sqrt(weight_cache_corrected)
                                                                                     + self.epsilon)
        layer.biases += -self.current_learning_rate * bias_momentums_corrected / (np.sqrt(bias_cache_corrected)
                                                                                  + self.epsilon)

    def update_params_bn(self, layer: Any) -> None:
        """Apply one Adam update step to the ``gamma`` and ``beta`` of a batch normalization layer."""

        # If no momentums/caches exist, create them filled w zeroes
        if not hasattr(layer, 'gamma_cache'):
            layer.gamma_momentums = np.zeros_like(layer.gamma)
            layer.gamma_cache = np.zeros_like(layer.gamma)
            layer.beta_momentums = np.zeros_like(layer.beta)
            layer.beta_cache = np.zeros_like(layer.beta)

        # Convert shape from (1, n_neurons) to (n_neurons,) - potential shape error
        dgamma = layer.dgamma.squeeze()
        dbeta = layer.dbeta.squeeze()

        # updates for gamma
        layer.gamma_momentums = (self.beta1 * layer.gamma_momentums + (1 - self.beta1) * dgamma)
        gamma_momentums_corrected = (layer.gamma_momentums / (1 - self.beta1 ** (self.iterations + 1)))

        layer.gamma_cache = (self.beta2 * layer.gamma_cache + (1 - self.beta2) * (dgamma ** 2))
        gamma_cache_corrected = (layer.gamma_cache / (1 - self.beta2 ** (self.iterations + 1)))

        layer.gamma -= self.current_learning_rate * gamma_momentums_corrected / (np.sqrt(gamma_cache_corrected) + self.epsilon)

        # updates for beta
        layer.beta_momentums = (self.beta1 * layer.beta_momentums + (1 - self.beta1) * dbeta)
        beta_momentums_corrected = (layer.beta_momentums / (1 - self.beta1 ** (self.iterations + 1)))

        layer.beta_cache = (self.beta2 * layer.beta_cache + (1 - self.beta2) * (dbeta ** 2))
        beta_cache_corrected = (layer.beta_cache / (1 - self.beta2 ** (self.iterations + 1)))

        layer.beta -= self.current_learning_rate * beta_momentums_corrected / (np.sqrt(beta_cache_corrected) + self.epsilon)

    def post_update_params(self) -> None:
        """Increment the internal iteration counter after all parameter updates for a batch."""
        self.iterations += 1


# Early Stopping
class Early_Stopping:
    """Halt training when validation loss stops improving and restore the best weights."""

    def __init__(self, patience: int = PATIENCE, min_delta: float = MIN_DELTA) -> None:
        """Initialize patience counter and best-loss tracker."""
        self.patience = patience    # Number of epochs to wait for improvement
        self.min_delta = min_delta  # Minimum change in validation loss to qualify as improvement
        self.best_loss = np.inf     # Stores the best validation loss encountered
        self.best_layer_states = None  # Stores the best trainable layer state
        self.wait = 0               # Counter for epochs without improvement

    def _capture_layer_state(self, layer: Any) -> dict[str, NDArray]:
        """Snapshot the trainable parameters of a single layer into a plain dict."""
        state: dict[str, NDArray] = {}

        if hasattr(layer, 'weights'):
            state['weights'] = layer.weights.copy()
            state['biases'] = layer.biases.copy()

        if hasattr(layer, 'gamma'):
            state['gamma'] = layer.gamma.copy()
            state['beta'] = layer.beta.copy()
            state['running_mean'] = layer.running_mean.copy()
            state['running_variance'] = layer.running_variance.copy()

        return state

    def forward(self, validation_loss: float, layers: list[Any]) -> bool:
        """Check for improvement; return True if training should stop."""
        if validation_loss < self.best_loss - self.min_delta:
            # Improvement has been detected
            self.best_loss = validation_loss
            self.wait = 0
            # Save the best state from all trainable layers.
            self.best_layer_states = [self._capture_layer_state(layer) for layer in layers]
        else:
            # if no improvement is detected
            self.wait += 1
            if self.wait >= self.patience:
                return True  # Stop training
        return False

    def restore_best_weights(self, layers: list[Any]) -> None:
        """Restore all layers to the parameter snapshot captured at the best validation loss."""
        # Restore the best state to the layers.
        if self.best_layer_states is not None:
            for layer, state in zip(layers, self.best_layer_states):
                if 'weights' in state:
                    layer.weights = state['weights'].copy()
                    layer.biases = state['biases'].copy()

                if 'gamma' in state:
                    layer.gamma = state['gamma'].copy()
                    layer.beta = state['beta'].copy()
                    layer.running_mean = state['running_mean'].copy()
                    layer.running_variance = state['running_variance'].copy()


# Dropout
class Dropout:
    """Inverted dropout regularization: scales active neurons by ``1/keep_prob`` during training."""

    def __init__(self, dropout_rate_input: float = DROPOUT_RATE_INPUT, dropout_rate_hidden: float = DROPOUT_RATE_HIDDEN) -> None:
        """Convert dropout rates to keep-probabilities and initialise the mask to None."""
        # percentage of neurons to keep active
        self.dropout_rate_input = 1 - dropout_rate_input
        self.dropout_rate_hidden = 1 - dropout_rate_hidden
        self.mask = None

    def forward(self, inputs: NDArray, training: bool = True, input_layer: bool = False) -> NDArray:
        """Apply an inverted dropout mask during training; pass inputs unchanged during inference."""

        # Save input values
        self.inputs = inputs

        if training:
            # determine what layer we are on and generate the dropout mask
            # divide by dropout to ensure the expected value remains the same across training and inference
            dropout_rate = self.dropout_rate_input if input_layer else self.dropout_rate_hidden
            self.mask = np.random.binomial(1, dropout_rate, size=inputs.shape) / dropout_rate

            # apply mask
            self.output = inputs * self.mask

        else:
            # no dropout applied during inference
            self.output = inputs

        return self.output

    def backward(self, dvalues: NDArray) -> NDArray:
        """Apply the stored dropout mask to upstream gradients."""

        # calc. gradient for active neuron inputs
        self.dinputs = dvalues * self.mask

        return self.dinputs


# Batch Normalization
class Batch_Normalization:
    """Per-feature batch normalization with learnable scale (gamma) and shift (beta).

    During training normalizes using batch statistics; at inference uses running statistics
    accumulated via exponential moving average.
    """

    def __init__(self, n_neurons: int, momentum: float = MOMENTUM, epsilon: float = EPSILON) -> None:
        """Initialize gamma=1, beta=0, and running-statistics buffers."""

        # Initialize the trainable scale (gamma) and shift (beta) parameters
        self.gamma = np.ones(n_neurons)
        self.beta = np.zeros(n_neurons)

        # momentum - how much of the old moving average to retain
        # epsilon - prevents 0 div error
        self.momentum = momentum
        self.epsilon = epsilon

        # store moving average of batch means and variances
        self.running_mean = np.zeros(n_neurons)
        self.running_variance = np.ones(n_neurons)

    def forward(self, inputs: NDArray, training: bool = True) -> NDArray:
        """Normalize using batch stats (training) or running stats (inference), then scale and shift."""

        # Save input values
        self.inputs = inputs

        # During training (training=True), while during inference/prediction (training=False)
        if training:
            # 1) Calculate the Mean and Variance
            batch_mean = np.mean(inputs, axis=0, keepdims=True)
            batch_var = np.var(inputs, axis=0, keepdims=True)

            # 2) Normalize the Batch
            x_hat = (inputs - batch_mean) / np.sqrt(batch_var + self.epsilon)

            # 3) Scale and Shift - update running stats
            y_out = self.gamma * x_hat + self.beta

            # Save variables for backward pass
            self.batch_mean = batch_mean
            self.batch_var = batch_var
            self.x_hat = x_hat

            # 4) Update running stats
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
            self.running_variance = self.momentum * self.running_variance + (1 - self.momentum) * batch_var

        # if running on test data
        else:
            x_hat = (inputs - self.running_mean) / np.sqrt(self.running_variance + self.epsilon)
            y_out = self.gamma * x_hat + self.beta

        self.output = y_out

        return self.output

    def backward(self, dvalues: NDArray) -> NDArray:
        """Compute gradients w.r.t. gamma, beta, and inputs via the full batch-norm backward formula."""

        # batch sample shape
        m = dvalues.shape[0]

        # NOTE: gamma and beta values are calc. here but need to be actually utilized in ADAM optimizer
        # compute gradient for gamma
        self.dgamma = np.sum(dvalues * self.x_hat, axis=0, keepdims=True) # Shape (1, n_neurons)

        # compute gradient for beta
        self.dbeta = np.sum(dvalues, axis=0, keepdims=True)

        # gradient w respect to x_hat
        dx_hat = dvalues * self.gamma  # Shape (batch_size, n_neurons)

        # gradient w respect to batch variance - Shape (1, n_neurons)
        dvar = np.sum(dx_hat * (self.inputs - self.batch_mean) * -0.5 * (self.batch_var + self.epsilon) ** (-1.5),
                      axis=0, keepdims=True)

        # gradient w respect to batch mean
        dmean = np.sum(dx_hat * (-1 / np.sqrt(self.batch_var + self.epsilon)), axis=0, keepdims=True) + dvar * np.sum(
            -2 * (self.inputs - self.batch_mean), axis=0, keepdims=True) / m

        # finally calc. gradient for inputs
        self.dinputs = dx_hat / np.sqrt(self.batch_var + self.epsilon) + dvar * 2 * (
                    self.inputs - self.batch_mean) / m + dmean / m  # Shape (batch_size, n_neurons)

        return self.dinputs
