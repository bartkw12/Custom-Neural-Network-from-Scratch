from __future__ import annotations

from dataclasses import dataclass
from typing import Any


# Hyperparameters for NN

# Training Parameters
BATCH_SIZE = 256
LEARNING_RATE = 0.002
EPOCHS = 50
LAMBDA = 0.001

# Model Parameters
INPUT_SIZE = 784
OUTPUT_SIZE = 10
HIDDEN_UNITS = 80
HIDDEN_LAYERS = 4

# Technique Parameters

# For Early Stopping
PATIENCE = 5
MIN_DELTA = 1e-5

# For Dropout
DROPOUT_RATE_INPUT = 0.10
DROPOUT_RATE_HIDDEN = 0.3

# For Batch Normalization
MOMENTUM = 0.185
EPSILON = 1e-5

# For ADAM optimizer
DECAY = 5e-7
EPSILON_A = 1e-7
BETA1 = 0.9
BETA2 = 0.999

# Other Settings
SEED = 9782


@dataclass(slots=True)
class NetworkConfig:
	"""Centralized configuration for architecture, training, and regularization hyperparameters.

	Supports both a simplified interface (``hidden_layers`` / ``hidden_units``) and a fully
	explicit ``layer_specs`` override for custom architectures.
	"""
	# Architecture (simplified)
	input_size: int = INPUT_SIZE
	output_size: int = OUTPUT_SIZE
	hidden_units: int = HIDDEN_UNITS
	hidden_layers: int = HIDDEN_LAYERS

	# Architecture (explicit override)
	layer_specs: list[dict[str, Any]] | None = None

	# Training
	batch_size: int = BATCH_SIZE
	learning_rate: float = LEARNING_RATE
	epochs: int = EPOCHS
	l2_lambda: float = LAMBDA
	seed: int = SEED

	# Dropout
	dropout_rate_input: float = DROPOUT_RATE_INPUT
	dropout_rate_hidden: float = DROPOUT_RATE_HIDDEN

	# Batch Normalization
	bn_momentum: float = MOMENTUM
	bn_epsilon: float = EPSILON

	# ADAM optimizer
	adam_decay: float = DECAY
	adam_epsilon: float = EPSILON_A
	adam_beta1: float = BETA1
	adam_beta2: float = BETA2

	# Early Stopping
	patience: int = PATIENCE
	min_delta: float = MIN_DELTA

	def __post_init__(self) -> None:
		"""Validate that at least one hidden layer is configured when using the simplified interface."""
		if self.hidden_layers < 1 and self.layer_specs is None:
			raise ValueError("hidden_layers must be at least 1 when layer_specs is not provided")

	def get_layer_specs(self) -> list[dict[str, Any]]:
		"""Return the ordered list of layer specification dicts used to build the network."""
		if self.layer_specs is not None:
			return self.layer_specs

		layer_specs = [
			{"type": "dense", "n_inputs": self.input_size, "n_neurons": self.hidden_units, "l2_lambda": self.l2_lambda},
			{"type": "batch_norm", "n_neurons": self.hidden_units, "momentum": self.bn_momentum, "epsilon": self.bn_epsilon},
			{"type": "relu"},
			{
				"type": "dropout",
				"dropout_rate_input": self.dropout_rate_input,
				"dropout_rate_hidden": self.dropout_rate_hidden,
				"input_layer": True,
			},
		]

		for _ in range(self.hidden_layers - 1):
			layer_specs.extend(
				[
					{"type": "dense", "n_inputs": self.hidden_units, "n_neurons": self.hidden_units, "l2_lambda": self.l2_lambda},
					{"type": "batch_norm", "n_neurons": self.hidden_units, "momentum": self.bn_momentum, "epsilon": self.bn_epsilon},
					{"type": "relu"},
					{
						"type": "dropout",
						"dropout_rate_input": self.dropout_rate_input,
						"dropout_rate_hidden": self.dropout_rate_hidden,
						"input_layer": False,
					},
				]
			)

		layer_specs.extend(
			[
				{"type": "dense", "n_inputs": self.hidden_units, "n_neurons": self.output_size, "l2_lambda": self.l2_lambda},
				{"type": "softmax"},
			]
		)

		return layer_specs


def default_config() -> NetworkConfig:
	"""Return a ``NetworkConfig`` instance populated with the default hyperparameter values."""
	return NetworkConfig()