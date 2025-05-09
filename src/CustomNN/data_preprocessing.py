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

    # print(train_dataset.data.shape, test_dataset.data.shape)

    return train_dataset, test_dataset