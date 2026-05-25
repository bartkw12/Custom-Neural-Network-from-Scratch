import numpy as np
import torch
from torchvision import datasets
from config import SEED

def load_fashion_MNIST(seed=SEED):
    """
    Load the necessary train and test datasets.
    """
    # Seed the pseudo number generator
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load dataset
    train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True)
    test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True)

    return train_dataset, test_dataset


def preprocess_data(train_dataset, test_dataset, validation_ratio=0.2):
    """
    Preprocess the dataset by reshaping it to a proper numpy array, splitting further to obtain a validation set,
    one-hot encoding, and finally standardizing the data.
    """
    # Preprocess the data
    # Prepare the data as numpy arrays - reshapes each 28×28 image into a 784-element vector
    # also, convert pixel values from integers (0-255) to floats of 0-1.0 for stability.
    X_train = train_dataset.data.numpy().reshape(-1, 28*28).astype('float32') / 255.0
    Y_train = train_dataset.targets.numpy()
    X_test = test_dataset.data.numpy().reshape(-1, 28*28).astype('float32') / 255.0
    Y_test = test_dataset.targets.numpy()

    # Shuffle the training data before splitting
    # Create a random permutation of indices
    indices = np.arange(X_train.shape[0])
    np.random.shuffle(indices)

    # Apply shuffle to X_train and Y_train
    X_train = X_train[indices]
    Y_train = Y_train[indices]

    # Split training set into train and validation set (80/20)
    validation_size = int(validation_ratio * X_train.shape[0])
    X_validation, Y_validation = X_train[:validation_size], Y_train[:validation_size]
    X_train, Y_train = X_train[validation_size:], Y_train[validation_size:]

    # One-Hot Encoding
    # Convert labels to one-hot encoding for multi-class classification
    # required for multi-class classification with softmax output and cross-entropy loss
    def one_hot_encode(labels, num_classes=10):
        return np.eye(num_classes)[labels]

    Y_train = one_hot_encode(Y_train)
    Y_validation = one_hot_encode(Y_validation)
    Y_test = one_hot_encode(Y_test)

    # Standardization
    # Calculate the mean and standard deviation of the training features
    X_train_mean = X_train.mean(axis=0)
    X_train_std = X_train.std(axis=0)
    X_train_std[X_train_std == 0] = 1.0  # To avoid division by zero

    # Standardize all three subsets of data to stabilize training
    X_train = (X_train - X_train_mean) / X_train_std
    X_validation = (X_validation - X_train_mean) / X_train_std
    X_test = (X_test - X_train_mean) / X_train_std

    return (X_train, Y_train), (X_validation, Y_validation), (X_test, Y_test)


if __name__ == "__main__":
    train_dataset, test_dataset = load_fashion_MNIST()
    (X_train, Y_train), (X_validation, Y_validation), (X_test, Y_test) = preprocess_data(train_dataset, test_dataset)

