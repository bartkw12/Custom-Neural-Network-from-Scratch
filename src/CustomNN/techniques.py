from config import LEARNING_RATE, DROPOUT_RATE_INPUT, DROPOUT_RATE_HIDDEN, PATIENCE, MIN_DELTA, MOMENTUM, EPSILON, BETA1, BETA2, EPSILON_A, DECAY
import numpy as np


# ADAM Optimizer
class ADAM_Optimizer:

    def __init__(self, learning_rate=LEARNING_RATE, beta1=BETA1, beta2=BETA2, epsilon=EPSILON_A, decay=DECAY):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.decay = decay
        self.iterations = 0

        # print(self.beta1, self.beta2, self.decay)

    # adjust the learning rate based on decay factor and # of iterations
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

            # print(f"The learning rate currently is: {self.current_learning_rate}")

    # Update parameters
    def update_params(self, layer):

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

        # Debug prints
        # print("Weight Momentums:\n", layer.weight_momentums)
        # print("Bias Momentums:\n", layer.bias_momentums)
        # print("Weight Cache:\n", layer.weight_cache)
        # print("Bias Cache:\n", layer.bias_cache)

        # apply bias correction to second moment estimates
        weight_cache_corrected = layer.weight_cache / (1 - self.beta2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / (1 - self.beta2 ** (self.iterations + 1))

        # Debug prints
        # print("Corrected Weight Momentums:\n", weight_momentums_corrected)
        # print("Corrected Bias Momentums:\n", bias_momentums_corrected)
        # print("Corrected Weight Cache:\n", weight_cache_corrected)
        # print("Corrected Bias Cache:\n", bias_cache_corrected)

        # update weights and biases w corrected momentum and cache
        layer.weights += -self.current_learning_rate * weight_momentums_corrected / (np.sqrt(weight_cache_corrected)
                                                                                     + self.epsilon)
        layer.biases += -self.current_learning_rate * bias_momentums_corrected / (np.sqrt(bias_cache_corrected)
                                                                                  + self.epsilon)

    # separate method to update gamma and beta from batch norm
    def update_params_bn(self, layer):

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

    # Call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1


# Early Stopping
class Early_Stopping:
    def __init__(self, patience=PATIENCE, min_delta=MIN_DELTA):
        self.patience = patience    # Number of epochs to wait for improvement
        self.min_delta = min_delta  # Minimum change in validation loss to qualify as improvement
        self.best_loss = np.inf     # Stores the best validation loss encountered
        self.best_weights = None    # Stores the best weights
        self.wait = 0               # Counter for epochs without improvement

    def forward(self, validation_loss, layers):
        if validation_loss < self.best_loss - self.min_delta:
            # Improvement has been detected
            self.best_loss = validation_loss
            self.wait = 0
            # Save the best weights from the layers list
            self.best_weights = [layer.weights.copy() for layer in layers if hasattr(layer, 'weights')]
        else:
            # if no improvement is detected
            self.wait += 1
            if self.wait >= self.patience:
                return True  # Stop training
        return False

    def restore_best_weights(self, layers):
        # Restore the best weights to the layers
        if self.best_weights is not None:
            idx = 0
            for layer in layers:
                if hasattr(layer, 'weights'):
                    layer.weights = self.best_weights[idx].copy()
                    idx += 1


# Dropout
class Dropout:

    def __init__(self, dropout_rate_input=DROPOUT_RATE_INPUT, dropout_rate_hidden=DROPOUT_RATE_HIDDEN):
        # percentage of neurons to keep active
        self.dropout_rate_input = 1 - dropout_rate_input
        self.dropout_rate_hidden = 1 - dropout_rate_hidden
        self.mask = None

    def forward(self, inputs, training=True, input_layer=False):

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

    def backward(self, dvalues):

        # calc. gradient for active neuron inputs
        self.dinputs = dvalues * self.mask

        return self.dinputs


# Batch Normalization
class Batch_Normalization:

    def __init__(self, n_neurons, momentum=MOMENTUM, epsilon=EPSILON):

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

    def forward(self, inputs, training=True):

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

    def backward(self, dvalues):

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
