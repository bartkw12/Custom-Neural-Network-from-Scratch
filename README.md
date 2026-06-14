# Custom Neural Network From Scratch

## Abstract
This repository presents a fully connected neural network for the Fashion-MNIST classification dataset, implemented from first principles in NumPy, with PyTorch used only for dataset loading in the custom pipeline. The project is structured as a reusable Python package and includes a parallel PyTorch implementation for architecture and training pipeline comparison.

The current custom model supports mini-batch training with Adam optimization, L2 regularization, batch normalization, dropout, and early stopping. Hyperparameters and layer wiring are centralized through a typed `NetworkConfig` interface.

## Method Summary
- `custom_nn`: the primary from scratch implementation under `src/custom_nn`
- `NeuralNetwork`: reusable training, evaluation, prediction, save, and load API
- `NetworkConfig`: centralized configuration for architecture and training behavior
- `pytorch_nn`: reference implementation built around the same layer specification and preprocessing flow
- shared Fashion-MNIST preprocessing: shuffle, train/validation split, one-hot encoding, and standardization from training statistics only

## Training Result
![Training Curve](results/trainingcurve.JPG)

The plot above is retained from the current repository results. Training history from the custom implementation is also written to `results/custom_nn_history.json` for later inspection.

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

On first run, `torchvision` downloads Fashion-MNIST into `data/`. The training script saves loss history to `results/custom_nn_history.json` and displays a Matplotlib plot at the end of execution.

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
train.py         Thin repository entry point for custom model training
results/         Saved plots and training histories
data/            Fashion-MNIST download cache
```

