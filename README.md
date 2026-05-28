# Custom-Neural-Network-from-Scratch
Custom Neural Network (NN) coded in Python without the use of any external AI/ML libraries. Implmented to solve a  multi-class classification problem using the Fashion MNIST dataset, achieving a test misclassification rate of 11.39%.

The dataset can be downloaded from:
https://www.kaggle.com/datasets/zalando-research/fashionmnist?resource=download.
(or via torch datasets)
- fashion-mnist train.csv: 60,000 training examples.
- fashion-mnist test.csv: 10,000 test examples.

Each example is a 28 × 28 grayscale image, resulting in 784 pixels per image. Each
pixel-value is an integer between 0 and 255, indicating the lightness or darkness of
that pixel. The CSV files have 785 columns: the first column is the class label, and
the remaining 784 columns are the pixel-values.

The NN features the following techniques:
- Adam optimizer with weight decay
- Batch normalization
- Dropout regularization
- Early stopping
- L2 regularization

## Training Results
![Training Curve](results/trainingcurve.JPG)

Note: These results were attained with the current hyperparameter values in the config.py file.
There is still potential for hyperparameter optimization.

## Requirements
- Python 3.10+
- NumPy
- Matplotlib
- PyTorch (for dataset loading)

## Installation
```bash
git clone https://github.com/bartkw12/Custom-Neural-Network-from-Scratch.git
cd Custom-Neural-Network-from-Scratch
pip install -r requirements.txt
pip install -e .
```

## Usage
```bash
python train.py
```

Or run the package entry point:

```bash
python -m custom_nn
```

Programmatic usage:

```python
from custom_nn import NeuralNetwork, NetworkConfig, load_fashion_MNIST, preprocess_data

config = NetworkConfig()
model = NeuralNetwork(config)

train_dataset, test_dataset = load_fashion_MNIST(seed=config.seed)
(X_train, Y_train), (X_val, Y_val), (X_test, Y_test) = preprocess_data(train_dataset, test_dataset)

history = model.train(X_train, Y_train, X_val, Y_val)
metrics = model.evaluate(X_test, Y_test)
```

## Project Structure
```bash
Custom-Neural-Network-from-Scratch/
├── results/
│   └── trainingcurve.JPG        # Saved training visualization
├── src/
│   └── custom_nn/
│       ├── __init__.py           # Public package exports
│       ├── __main__.py           # Enables `python -m custom_nn`
│       ├── config.py             # Hyperparameters + NetworkConfig dataclass
│       ├── data_preprocessing.py # Data loading & preprocessing
│       ├── model.py              # Core layer and loss implementations
│       ├── network.py            # NeuralNetwork class API
│       └── techniques.py         # Optimization/regularization
├── train.py                      # Thin training entry point
├── pyproject.toml
├── requirements.txt
├── README.md
└── AGENTS.md
```

