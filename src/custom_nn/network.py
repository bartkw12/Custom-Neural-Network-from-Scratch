from pathlib import Path

import numpy as np
import torch

from .config import NetworkConfig, default_config
from .model import Layer_Dense, Activation_ReLU, Activation_Softmax, Categorical_Cross_entropy_loss
from .techniques import ADAM_Optimizer, Batch_Normalization, Dropout, Early_Stopping


class NeuralNetwork:
    def __init__(self, config: NetworkConfig | None = None):
        self.config = config if config is not None else default_config()

        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)

        self.layers = self._build_layers()
        self.loss_function = Categorical_Cross_entropy_loss(l2_lambda=self.config.l2_lambda)
        self.optimizer = ADAM_Optimizer(
            learning_rate=self.config.learning_rate,
            beta1=self.config.adam_beta1,
            beta2=self.config.adam_beta2,
            epsilon=self.config.adam_epsilon,
            decay=self.config.adam_decay,
        )
        self.early_stopper = Early_Stopping(
            patience=self.config.patience,
            min_delta=self.config.min_delta,
        )

    def _build_layers(self):
        layers = []

        for spec in self.config.get_layer_specs():
            layer_type = spec["type"]

            if layer_type == "dense":
                layers.append(
                    Layer_Dense(
                        n_inputs=spec["n_inputs"],
                        n_neurons=spec["n_neurons"],
                        l2_lambda=spec.get("l2_lambda", self.config.l2_lambda),
                    )
                )
            elif layer_type == "batch_norm":
                layers.append(
                    Batch_Normalization(
                        n_neurons=spec["n_neurons"],
                        momentum=spec.get("momentum", self.config.bn_momentum),
                        epsilon=spec.get("epsilon", self.config.bn_epsilon),
                    )
                )
            elif layer_type == "relu":
                layers.append(Activation_ReLU())
            elif layer_type == "dropout":
                layer = Dropout(
                    dropout_rate_input=spec.get("dropout_rate_input", self.config.dropout_rate_input),
                    dropout_rate_hidden=spec.get("dropout_rate_hidden", self.config.dropout_rate_hidden),
                )
                layer.input_layer = spec.get("input_layer", False)
                layers.append(layer)
            elif layer_type == "softmax":
                layers.append(Activation_Softmax())
            else:
                raise ValueError(f"Unsupported layer type: {layer_type}")

        return layers

    def forward(self, X, training=True):
        outputs = X

        for layer in self.layers:
            if isinstance(layer, Batch_Normalization):
                outputs = layer.forward(outputs, training=training)
            elif isinstance(layer, Dropout):
                outputs = layer.forward(
                    outputs,
                    training=training,
                    input_layer=getattr(layer, "input_layer", False),
                )
            else:
                outputs = layer.forward(outputs)

        return outputs

    def backward(self, y_true):
        dvalues = self.layers[-1].backward(y_true)

        for layer in reversed(self.layers[:-1]):
            if hasattr(layer, "backward") and dvalues is not None:
                dvalues = layer.backward(dvalues)

        return dvalues

    def _update_weights(self):
        self.optimizer.pre_update_params()

        for layer in self.layers:
            if isinstance(layer, Layer_Dense):
                self.optimizer.update_params(layer)
            elif isinstance(layer, Batch_Normalization):
                self.optimizer.update_params_bn(layer)

        self.optimizer.post_update_params()

    def _compute_loss(self, X, Y, training=False):
        outputs = self.forward(X, training=training)
        return self.loss_function.forward(outputs, Y)

    def _compute_accuracy(self, X, Y):
        predictions = self.predict(X)
        true_labels = np.argmax(Y, axis=1)
        return float(np.mean(predictions == true_labels))

    def train(self, X_train, Y_train, X_val, Y_val):
        history = {
            "train_loss": [],
            "val_loss": [],
        }

        for epoch in range(self.config.epochs):
            indices = np.arange(X_train.shape[0])
            np.random.shuffle(indices)
            X_train = X_train[indices]
            Y_train = Y_train[indices]

            total_loss = 0.0
            num_batches = X_train.shape[0] // self.config.batch_size
            if num_batches == 0:
                raise ValueError("batch_size is larger than the number of training samples")

            for batch_index in range(num_batches):
                start = batch_index * self.config.batch_size
                end = start + self.config.batch_size

                X_batch = X_train[start:end]
                Y_batch = Y_train[start:end]

                outputs = self.forward(X_batch, training=True)
                batch_loss = self.loss_function.forward(outputs, Y_batch)
                total_loss += batch_loss

                self.backward(Y_batch)
                self._update_weights()

            avg_train_loss = total_loss / num_batches
            val_loss = self._compute_loss(X_val, Y_val, training=False)

            history["train_loss"].append(avg_train_loss)
            history["val_loss"].append(val_loss)

            stop = self.early_stopper.forward(val_loss, self.layers)
            if stop:
                self.early_stopper.restore_best_weights(self.layers)
                break

        return history

    def predict(self, X):
        outputs = self.forward(X, training=False)
        return np.argmax(outputs, axis=1)

    def evaluate(self, X, Y):
        return {
            "loss": self._compute_loss(X, Y, training=False),
            "accuracy": self._compute_accuracy(X, Y),
        }

    def save(self, path):
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        state = {}

        for index, layer in enumerate(self.layers):
            prefix = f"layer_{index}"

            if hasattr(layer, "weights"):
                state[f"{prefix}_weights"] = layer.weights
                state[f"{prefix}_biases"] = layer.biases

            if hasattr(layer, "gamma"):
                state[f"{prefix}_gamma"] = layer.gamma
                state[f"{prefix}_beta"] = layer.beta
                state[f"{prefix}_running_mean"] = layer.running_mean
                state[f"{prefix}_running_variance"] = layer.running_variance

        np.savez(save_path, **state)

    def load(self, path):
        loaded = np.load(path)

        for index, layer in enumerate(self.layers):
            prefix = f"layer_{index}"

            weights_key = f"{prefix}_weights"
            biases_key = f"{prefix}_biases"
            gamma_key = f"{prefix}_gamma"
            beta_key = f"{prefix}_beta"
            running_mean_key = f"{prefix}_running_mean"
            running_variance_key = f"{prefix}_running_variance"

            if hasattr(layer, "weights") and weights_key in loaded:
                layer.weights = loaded[weights_key]
                layer.biases = loaded[biases_key]

            if hasattr(layer, "gamma") and gamma_key in loaded:
                layer.gamma = loaded[gamma_key]
                layer.beta = loaded[beta_key]
                layer.running_mean = loaded[running_mean_key]
                layer.running_variance = loaded[running_variance_key]