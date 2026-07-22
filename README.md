# Custom Neural Network From Scratch

## Abstract
This repository presents a fully connected neural network for the Fashion-MNIST classification dataset, implemented from first principles in NumPy, with PyTorch used only for dataset loading in the custom pipeline. The project is structured as a reusable Python package and includes a parallel PyTorch implementation for architecture and training pipeline comparison.

The current custom model supports mini-batch training with Adam optimization, L2 regularization, batch normalization, dropout, and early stopping. Hyperparameters and layer wiring are centralized through a typed `NetworkConfig` interface.

## Why Build a Neural Network From Scratch?

High-level ML libraries make it easy to train models, but they hide the numerical and architectural details that make neural networks work. This project was built to demonstrate a deeper understanding of:

- how tensors flow through a network during forward propagation
- how gradients are derived and propagated during backpropagation
- how optimizers such as Adam update parameters
- how dropout, batch normalization, and L2 regularization affect training
- how train/validation/test preprocessing should be structured to avoid data leakage
- how a custom implementation can be compared against a framework implementation

## Features


### Custom NumPy Implementation

- Fully connected dense layers
- ReLU activation
- Softmax classification output
- Cross-entropy loss
- Manual backpropagation
- Mini-batch training
- Adam optimizer
- L2 regularization
- Dropout using inverted dropout scaling
- Batch normalization
- Early stopping
- Model evaluation and prediction API
- Save/load support for trained parameters


### PyTorch Reference Implementation

- Equivalent architecture for comparison
- Same Fashion-MNIST preprocessing pipeline
- Same train/validation/test split
- Matching hyperparameters where possible
- Training and validation history logging
- Test accuracy comparison


### Planned / In Progress

- Confusion matrix visualization
- Per-class accuracy analysis
- Sample prediction grids (correct vs. incorrect)



## Method Summary
- `custom_nn`: the primary from scratch implementation under `src/custom_nn`
- `NeuralNetwork`: reusable training, evaluation, prediction, save, and load API
- `NetworkConfig`: centralized configuration for architecture and training behavior
- `pytorch_nn`: reference implementation built around the same layer specification and preprocessing flow
- shared Fashion-MNIST preprocessing: shuffle, train/validation split, one-hot encoding, and standardization from training statistics only

## Training Pipeline

The training pipeline follows these steps:

1. Load Fashion-MNIST using `torchvision`.
2. Convert images and labels into NumPy arrays.
3. Shuffle the training set using a fixed random seed.
4. Split the original training set into training and validation subsets.
5. One-hot encode labels.
6. Standardize features using statistics computed from the training set only.
7. Train the custom NumPy network using mini-batch gradient descent with Adam.
8. Track training and validation loss across epochs.
9. Apply early stopping based on validation performance.
10. Evaluate the final model on the held-out test set.

## Model Architecture

The default custom network is a fully connected classifier for flattened Fashion-MNIST images.

```text
Input: 28 × 28 grayscale image
Flatten: 784 features
Dense Layer: [PLACEHOLDER: hidden units]
Batch Normalization
ReLU
Dropout
Dense Layer: [PLACEHOLDER: hidden units]
Batch Normalization
ReLU
Dropout
Output Dense Layer: 10 classes
Softmax + Cross-Entropy Loss.
```

## Training Result
![Training Curve](results/trainingcurve.JPG)

The plot above is retained from the current repository results. Training history from the custom implementation is also written to `results/custom_nn_history.json` for later inspection.


## Mathematical Overview

This implementation manually computes all forward and backward operations.

### Dense Layer

Forward pass:

```math
Z = XW + b
```

Backward pass:
```math
\frac{\partial L}{\partial W} = X^T \frac{\partial L}{\partial Z}
```

```math
\frac{\partial L}{\partial b} = \sum_i \frac{\partial L}{\partial Z_i}
```

```math
\frac{\partial L}{\partial X} = \frac{\partial L}{\partial Z} W^T
```

### ReLU Activation
```math
\text{ReLU}(x) = \max(0, x)
```

### Softmax Activation
```math
\hat{y}_i = \frac{e^{z_i}}{\sum_j e^{z_j}}
```

### Cross-Entropy Loss
```math
L = -\sum_i y_i \log \hat{y}_i
```


## Installation
```bash
git clone https://github.com/bartkw12/Custom-Neural-Network-from-Scratch.git
cd Custom-Neural-Network-from-Scratch
pip install -r requirements.txt
pip install -e .
```

## Reproducing a Run
```bash
python train.py
```

Or run the package entry point:

```bash
python -m custom_nn
```

These compatibility paths default to the `custom` subcommand of the unified CLI.

### Unified CLI Commands

Run custom NumPy training:

```bash
python train.py custom
```

Run PyTorch reference training:

```bash
python train.py pytorch
```

Generate explicit comparison artifacts from saved run outputs:

```bash
python train.py compare
```

PyTorch compatibility entry point (also delegates to the unified CLI):

```bash
python run_pytorch.py
```

### Common Training Flags

The `custom` and `pytorch` subcommands share the same experiment flags:

- `--config` (JSON config file)
- `--epochs`
- `--learning-rate`
- `--batch-size`
- `--hidden-layers`
- `--hidden-units`
- `--seed`
- `--no-early-stopping`
- `--save-path`

Example with overrides:

```bash
python train.py custom --epochs 20 --learning-rate 0.001 --batch-size 128
```

### JSON Config File Format

The CLI accepts a JSON object whose keys map directly to `NetworkConfig` fields.
CLI flags override values from the config file.

Example `experiment.json`:

```json
{
	"epochs": 12,
	"learning_rate": 0.001,
	"batch_size": 128,
	"hidden_layers": 3,
	"hidden_units": 96,
	"seed": 9782,
	"l2_lambda": 0.001,
	"dropout_rate_input": 0.1,
	"dropout_rate_hidden": 0.3,
	"patience": 5,
	"min_delta": 1e-05
}
```

Run with config:

```bash
python train.py pytorch --config experiment.json
```

### Artifact Output Layout

Each completed training run writes artifacts to both a stable "latest" location and a timestamped archive location.

```text
results/
├── latest/
│   ├── custom/
│   │   ├── run_summary.json
│   │   ├── metrics.json
│   │   ├── history.json
│   │   ├── history.csv
│   │   └── best_checkpoint.npz
│   └── pytorch/
│       ├── run_summary.json
│       ├── metrics.json
│       ├── history.json
│       ├── history.csv
│       └── best_checkpoint.pt
└── runs/
		└── <run_id>/
				├── custom/
				└── pytorch/
```

The compare workflow is explicit and artifact-driven. By default it reads:

- `results/latest/custom/run_summary.json`
- `results/latest/custom/history.json`
- `results/latest/pytorch/run_summary.json`
- `results/latest/pytorch/history.json`

On first run, `torchvision` downloads Fashion-MNIST into `data/`.

## Programmatic Usage
```python
from custom_nn import NeuralNetwork, NetworkConfig, load_fashion_MNIST, preprocess_data

config = NetworkConfig()
model = NeuralNetwork(config)

train_dataset, test_dataset = load_fashion_MNIST(seed=config.seed)
(X_train, Y_train), (X_val, Y_val), (X_test, Y_test) = preprocess_data(train_dataset, test_dataset)

history = model.train(X_train, Y_train, X_val, Y_val)
metrics = model.evaluate(X_test, Y_test)
```

## Repository Layout
```text
src/
├── custom_nn/   NumPy-based NN, training pipeline, and configuration
└── pytorch_nn/  PyTorch reference model and comparison utilities
train.py         Thin compatibility entry point to unified CLI
run_pytorch.py   Thin compatibility entry point to unified CLI (PyTorch default)
results/         Run artifacts (latest + archived runs), plots, and histories
data/            Fashion-MNIST download cache
```
