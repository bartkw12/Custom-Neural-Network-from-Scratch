"""Analysis helpers for visualization workflows."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray

from custom_nn import NetworkConfig, NeuralNetwork, load_fashion_MNIST, preprocess_data
from pytorch_nn.model import FashionMNISTNet


IntArray = NDArray[np.int64]
FloatArray = NDArray[np.float64]


def compute_confusion_matrix(
    y_true: IntArray,
    y_pred: IntArray,
    num_classes: int = 10,
) -> IntArray:
    """Compute a confusion matrix for integer class labels."""
    true_labels = np.asarray(y_true, dtype=np.int64).reshape(-1)
    predicted_labels = np.asarray(y_pred, dtype=np.int64).reshape(-1)

    if true_labels.shape != predicted_labels.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    if num_classes < 1:
        raise ValueError("num_classes must be at least 1")

    if true_labels.size == 0:
        return np.zeros((num_classes, num_classes), dtype=np.int64)

    if np.any(true_labels < 0) or np.any(true_labels >= num_classes):
        raise ValueError("y_true contains labels outside the valid class range")

    if np.any(predicted_labels < 0) or np.any(predicted_labels >= num_classes):
        raise ValueError("y_pred contains labels outside the valid class range")

    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(confusion_matrix, (true_labels, predicted_labels), 1)
    return confusion_matrix


def compute_per_class_accuracy(
    y_true: IntArray,
    y_pred: IntArray,
    num_classes: int = 10,
) -> FloatArray:
    """Compute per-class accuracy, returning 0.0 for classes with no true samples."""
    confusion_matrix = compute_confusion_matrix(y_true, y_pred, num_classes=num_classes)
    class_totals = confusion_matrix.sum(axis=1)
    correct_predictions = np.diag(confusion_matrix).astype(np.float64)

    per_class_accuracy = np.zeros(num_classes, dtype=np.float64)
    non_empty_classes = class_totals > 0
    per_class_accuracy[non_empty_classes] = (
        correct_predictions[non_empty_classes] / class_totals[non_empty_classes]
    )
    return per_class_accuracy


def _load_summary(summary_path: str | Path) -> dict:
    """Load a run summary JSON file."""
    summary_path = Path(summary_path)
    with summary_path.open("r", encoding="utf-8") as summary_file:
        return json.load(summary_file)


def _default_summary_path(backend: str) -> Path:
    """Return the default summary path for a backend (latest run)."""
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "results" / "latest" / backend / "run_summary.json"


def _config_from_summary(summary: dict) -> NetworkConfig:
    """Build a NetworkConfig from a run summary dict."""
    return NetworkConfig(**summary["config"])


def _load_custom_model(summary: dict) -> NeuralNetwork:
    """Load a saved custom NeuralNetwork model from a run summary."""
    config = _config_from_summary(summary)
    model = NeuralNetwork(config)
    checkpoint_path = Path(summary["checkpoints"]["latest"])
    model.load(checkpoint_path)
    return model


def _load_pytorch_model(summary: dict) -> FashionMNISTNet:
    """Load a saved PyTorch model from a run summary."""
    config = _config_from_summary(summary)
    model = FashionMNISTNet(config)
    checkpoint_path = Path(summary["checkpoints"]["latest"])
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()
    return model


def _rebuild_test_set(config: NetworkConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Rebuild the test set (X_test, Y_test_onehot, raw_test_images).

    Returns:
        X_test: Standardized test features (784,) per sample.
        Y_test: One-hot encoded test labels.
        raw_test_images: Unstandardized raw 28x28 images, aligned with X_test/Y_test by index.
    """
    train_dataset, test_dataset = load_fashion_MNIST(seed=config.seed)
    (_, _), (_, _), (X_test, Y_test) = preprocess_data(train_dataset, test_dataset)

    # Extract raw unstandardized images (same index order as X_test/Y_test).
    raw_test_images = (test_dataset.data.numpy() / 255.0).astype(np.float32)

    return X_test, Y_test, raw_test_images


def _predict_custom(model: NeuralNetwork, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Predict on custom model, returning both class indices and softmax probabilities.

    Returns:
        y_pred: Class indices (shape: (n_samples,)).
        probs: Softmax probabilities (shape: (n_samples, num_classes)).
    """
    y_pred = model.predict(X_test)
    probs = model.forward(X_test, training=False)
    return y_pred, probs


def _predict_pytorch(model: FashionMNISTNet, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Predict on PyTorch model, returning both class indices and softmax probabilities.

    Returns:
        y_pred: Class indices (shape: (n_samples,)).
        probs: Softmax probabilities (shape: (n_samples, num_classes)).
    """
    X_test_tensor = torch.from_numpy(X_test).to(dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        logits = model(X_test_tensor)
        probs = torch.softmax(logits, dim=1).numpy()
        y_pred = logits.argmax(dim=1).numpy()

    return y_pred, probs