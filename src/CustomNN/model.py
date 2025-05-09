from config import LAMBDA
import numpy as np


class Layer_Dense:

    # NN Layer initialization
    def __init__(self, n_inputs, n_neurons, l2_lambda=LAMBDA):

        # Initialize weights and biases
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)  # set weights to random (mean 0 and variance 1)
        self.biases = np.zeros((1, n_neurons))                      # set biases to be zero

        # L2 regularization
        self.l2_lambda = l2_lambda

        # if vanishing gradient try:
        # self.weights = np.random.randn(n_inputs, n_neurons) * np.sqrt(2 / n_inputs)

    # Forward pass
    def forward(self, inputs):

        # Store inputs for backward pass
        self.inputs = inputs

        # compute linear transformation y = inputs * weights + biases
        self.output = np.dot(inputs, self.weights) + self.biases

        return self.output

    # Backpropagation
    def backward(self, dvalues):

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

    # Forward pass
    def forward(self, inputs):

        # Remember input values
        self.inputs = inputs

        # Calculate output values from input
        self.output = np.maximum(0, inputs)

        return self.output

    # Backpropagation
    def backward(self, dvalues):

        # copy to modify original variable
        # gradient of the loss w respect to the input of the ReLU function
        self.dinputs = dvalues.copy()

        # 0 gradient for input if values were neg.
        self.dinputs[self.inputs <= 0] = 0

        return self.dinputs


# Softmax Activation
class Activation_Softmax:

    # Forward pass
    def forward(self, inputs):

        # Remember input values
        self.inputs = inputs

        # Determine probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

        # Normalize probabilities per sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.output = probabilities

        return self.output

    # Backpropagation
    def backward(self, y_true):

        # this backwards pass simplifies with the categorical cross entropy loss backwards pass
        # Gradient of the loss with respect to the inputs
        self.dinputs = self.output - y_true

        return self.dinputs


# Categorical Cross-entropy Loss Function
class Categorical_Cross_entropy_loss:

    def __init__(self, l2_lambda=LAMBDA):
        self.l2_lambda = l2_lambda

    # Forward Pass
    def forward(self, y_pred, y_true):

        # clip data to prevent 0 div error and not skew mean
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # one hot encoded labels from data processing
        correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        # Loss calc
        negative_log_likelihoods = -np.log(correct_confidences)

        return np.mean(negative_log_likelihoods)