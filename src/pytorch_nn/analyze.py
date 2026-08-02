"""Analysis helpers for visualization workflows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy.typing import NDArray

from custom_nn import FASHION_MNIST_CLASSES, NetworkConfig, NeuralNetwork, load_fashion_MNIST, preprocess_data
from pytorch_nn.compare import load_comparison_histories
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


def plot_confusion_matrix(
    confusion_matrix: IntArray,
    class_names: Sequence[str],
    model_label: str,
    output_path: str | Path,
) -> Path:
    """Render and save an annotated confusion-matrix heatmap."""
    matrix = np.asarray(confusion_matrix, dtype=np.int64)

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("confusion_matrix must be a square 2D array")

    num_classes = matrix.shape[0]
    if len(class_names) != num_classes:
        raise ValueError("class_names length must match confusion_matrix dimensions")

    total_samples = int(matrix.sum())
    accuracy = float(np.trace(matrix) / total_samples) if total_samples > 0 else 0.0

    figure, axis = plt.subplots(figsize=(10, 8))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    axis.set_xticks(np.arange(num_classes))
    axis.set_yticks(np.arange(num_classes))
    axis.set_xticklabels(class_names)
    axis.set_yticklabels(class_names)
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.set_title(f"{model_label} Confusion Matrix (Accuracy: {accuracy * 100:.2f}%)")

    max_value = int(matrix.max()) if matrix.size > 0 else 0
    threshold = max_value / 2.0
    for row in range(num_classes):
        for col in range(num_classes):
            value = int(matrix[row, col])
            text_color = "white" if value > threshold else "black"
            axis.text(col, row, f"{value}", ha="center", va="center", color=text_color, fontsize=9)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output_path


def plot_per_class_accuracy_comparison(
    custom_accuracy: FloatArray,
    pytorch_accuracy: FloatArray,
    class_names: Sequence[str],
    output_path: str | Path,
) -> Path:
    """
    Render a grouped bar chart comparing per-class accuracy between custom and PyTorch models.
    
    Classes are sorted by average accuracy (hardest first) to highlight the most difficult items.
    """
    custom_acc = np.asarray(custom_accuracy, dtype=np.float64).reshape(-1)
    pytorch_acc = np.asarray(pytorch_accuracy, dtype=np.float64).reshape(-1)

    num_classes = len(custom_acc)
    if len(pytorch_acc) != num_classes or len(class_names) != num_classes:
        raise ValueError("custom_accuracy, pytorch_accuracy, and class_names must all have the same length")

    # Compute average accuracy per class (lower = harder).
    avg_accuracy = (custom_acc + pytorch_acc) / 2.0

    # Sort by average accuracy (ascending, so hardest first).
    sorted_indices = np.argsort(avg_accuracy)

    sorted_class_names = [str(class_names[i]) for i in sorted_indices]
    sorted_custom_acc = custom_acc[sorted_indices] * 100.0
    sorted_pytorch_acc = pytorch_acc[sorted_indices] * 100.0

    figure, axis = plt.subplots(figsize=(12, 6))

    x_positions = np.arange(num_classes)
    bar_width = 0.35

    axis.bar(x_positions - bar_width / 2, sorted_custom_acc, bar_width, label="Custom NN", alpha=0.8)
    axis.bar(x_positions + bar_width / 2, sorted_pytorch_acc, bar_width, label="PyTorch NN", alpha=0.8)

    axis.set_xlabel("Clothing Item (sorted by difficulty)")
    axis.set_ylabel("Accuracy (%)")
    axis.set_title("Per-Class Accuracy Comparison (Hardest First)")
    axis.set_xticks(x_positions)
    axis.set_xticklabels(sorted_class_names, rotation=45, ha="right")
    axis.set_ylim(0, 105)
    axis.legend()
    axis.grid(axis="y", alpha=0.3)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output_path


def plot_sample_predictions(
    raw_images: np.ndarray,
    y_true: IntArray,
    y_pred: IntArray,
    class_names: Sequence[str],
    model_label: str,
    output_path: str | Path,
    sample_count: int = 25,
    seed: int | None = None,
) -> Path:
    """
    Render a grid of sample predictions with color-coded borders and titles.

    Each cell shows a grayscale image with title 'Pred: {label} / True: {label}'.
    Green title and border indicates a correct prediction; red indicates incorrect.
    The selection deliberately includes a mix of correct and incorrect predictions.
    """
    images = np.asarray(raw_images)
    true_labels = np.asarray(y_true, dtype=np.int64).reshape(-1)
    pred_labels = np.asarray(y_pred, dtype=np.int64).reshape(-1)

    if images.shape[0] != true_labels.shape[0] or images.shape[0] != pred_labels.shape[0]:
        raise ValueError("raw_images, y_true, and y_pred must all have the same number of samples")

    rng = np.random.default_rng(seed)

    correct_indices = np.where(true_labels == pred_labels)[0]
    incorrect_indices = np.where(true_labels != pred_labels)[0]

    # Aim for at least min(5, total_incorrect) incorrect samples in the grid.
    n_incorrect_target = min(5, len(incorrect_indices))
    n_correct_target = sample_count - n_incorrect_target

    # Sample without replacement, capped at what is available.
    n_incorrect_actual = min(n_incorrect_target, len(incorrect_indices))
    n_correct_actual = min(n_correct_target, len(correct_indices))
    actual_count = n_incorrect_actual + n_correct_actual

    chosen_incorrect = rng.choice(incorrect_indices, size=n_incorrect_actual, replace=False)
    chosen_correct = rng.choice(correct_indices, size=n_correct_actual, replace=False)

    selected_indices = np.concatenate([chosen_incorrect, chosen_correct])
    rng.shuffle(selected_indices)

    n_cols = int(np.ceil(np.sqrt(actual_count)))
    n_rows = int(np.ceil(actual_count / n_cols))

    figure, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.0, n_rows * 2.4))
    axes_flat = np.asarray(axes).reshape(-1)

    figure.suptitle(f"{model_label} — Sample Predictions", fontsize=13, y=1.01)

    for plot_index, ax in enumerate(axes_flat):
        if plot_index >= actual_count:
            ax.axis("off")
            continue

        sample_index = int(selected_indices[plot_index])
        image = images[sample_index]
        true_class = str(class_names[true_labels[sample_index]])
        pred_class = str(class_names[pred_labels[sample_index]])
        is_correct = bool(true_labels[sample_index] == pred_labels[sample_index])
        color = "green" if is_correct else "red"

        # Display grayscale image (handles both (H, W) and (H, W, C) shapes).
        display_image = image.squeeze()
        ax.imshow(display_image, cmap="gray", vmin=0.0, vmax=1.0)

        ax.set_title(f"Pred: {pred_class}\nTrue: {true_class}", fontsize=7, color=color)
        ax.set_xticks([])
        ax.set_yticks([])

        # Color-code all four spines.
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output_path


def plot_training_curves_combined(
    histories: dict[str, dict[str, list[float]]],
    output_path: str | Path,
) -> Path:
    """
    Render a combined training-curves figure with two side-by-side subplots.

    Left subplot: training loss for both backends.
    Right subplot: validation loss for both backends.

    The ``histories`` dict must follow the shape returned by
    ``pytorch_nn.compare.load_comparison_histories``:
    ``{"custom_nn": {"train_loss": [...], "val_loss": [...]}, "pytorch_nn": {...}}``.
    """
    custom_history = histories["custom_nn"]
    pytorch_history = histories["pytorch_nn"]

    custom_train = custom_history["train_loss"]
    custom_val = custom_history["val_loss"]
    pytorch_train = pytorch_history["train_loss"]
    pytorch_val = pytorch_history["val_loss"]

    figure, (ax_train, ax_val) = plt.subplots(1, 2, figsize=(14, 5))

    # --- Left: training loss ---
    ax_train.plot(range(1, len(custom_train) + 1), custom_train, label="Custom NN", linewidth=2)
    ax_train.plot(range(1, len(pytorch_train) + 1), pytorch_train, label="PyTorch NN", linewidth=2)
    ax_train.set_xlabel("Epoch")
    ax_train.set_ylabel("Cross-entropy Loss")
    ax_train.set_title("Training Loss")
    ax_train.legend()
    ax_train.grid(alpha=0.3)

    # --- Right: validation loss ---
    ax_val.plot(range(1, len(custom_val) + 1), custom_val, label="Custom NN", linewidth=2)
    ax_val.plot(range(1, len(pytorch_val) + 1), pytorch_val, label="PyTorch NN", linewidth=2)
    ax_val.set_xlabel("Epoch")
    ax_val.set_ylabel("Cross-entropy Loss")
    ax_val.set_title("Validation Loss")
    ax_val.legend()
    ax_val.grid(alpha=0.3)

    figure.suptitle("Custom NN vs PyTorch NN — Training Curves", fontsize=13)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output_path


def generate_analysis_artifacts(
    custom_summary_path: str | Path | None = None,
    pytorch_summary_path: str | Path | None = None,
    sample_count: int = 25,
    sample_seed: int | None = None,
) -> dict[str, Path]:
    """Generate all Phase 6 analysis artifacts into the top-level results directory."""
    if custom_summary_path is None:
        custom_summary_path = _default_summary_path("custom")
    if pytorch_summary_path is None:
        pytorch_summary_path = _default_summary_path("pytorch")

    custom_summary = _load_summary(custom_summary_path)
    pytorch_summary = _load_summary(pytorch_summary_path)

    custom_model = _load_custom_model(custom_summary)
    pytorch_model = _load_pytorch_model(pytorch_summary)

    custom_config = _config_from_summary(custom_summary)
    pytorch_config = _config_from_summary(pytorch_summary)

    custom_x_test, custom_y_test_onehot, custom_raw_images = _rebuild_test_set(custom_config)
    pytorch_x_test, pytorch_y_test_onehot, pytorch_raw_images = _rebuild_test_set(pytorch_config)

    custom_y_true = np.argmax(custom_y_test_onehot, axis=1).astype(np.int64)
    pytorch_y_true = np.argmax(pytorch_y_test_onehot, axis=1).astype(np.int64)

    custom_y_pred, _ = _predict_custom(custom_model, custom_x_test)
    pytorch_y_pred, _ = _predict_pytorch(pytorch_model, pytorch_x_test)

    num_classes = len(FASHION_MNIST_CLASSES)
    custom_cm = compute_confusion_matrix(custom_y_true, custom_y_pred, num_classes=num_classes)
    pytorch_cm = compute_confusion_matrix(pytorch_y_true, pytorch_y_pred, num_classes=num_classes)

    custom_per_class_accuracy = compute_per_class_accuracy(custom_y_true, custom_y_pred, num_classes=num_classes)
    pytorch_per_class_accuracy = compute_per_class_accuracy(pytorch_y_true, pytorch_y_pred, num_classes=num_classes)

    results_dir = Path(__file__).resolve().parents[2] / "results"
    histories = load_comparison_histories()

    artifacts = {
        "confusion_matrix_custom": plot_confusion_matrix(
            confusion_matrix=custom_cm,
            class_names=FASHION_MNIST_CLASSES,
            model_label="Custom NN",
            output_path=results_dir / "confusion_matrix_custom.png",
        ),
        "confusion_matrix_pytorch": plot_confusion_matrix(
            confusion_matrix=pytorch_cm,
            class_names=FASHION_MNIST_CLASSES,
            model_label="PyTorch NN",
            output_path=results_dir / "confusion_matrix_pytorch.png",
        ),
        "per_class_accuracy_comparison": plot_per_class_accuracy_comparison(
            custom_accuracy=custom_per_class_accuracy,
            pytorch_accuracy=pytorch_per_class_accuracy,
            class_names=FASHION_MNIST_CLASSES,
            output_path=results_dir / "per_class_accuracy_comparison.png",
        ),
        "sample_predictions_custom": plot_sample_predictions(
            raw_images=custom_raw_images,
            y_true=custom_y_true,
            y_pred=custom_y_pred,
            class_names=FASHION_MNIST_CLASSES,
            model_label="Custom NN",
            output_path=results_dir / "sample_predictions_custom.png",
            sample_count=sample_count,
            seed=sample_seed,
        ),
        "sample_predictions_pytorch": plot_sample_predictions(
            raw_images=pytorch_raw_images,
            y_true=pytorch_y_true,
            y_pred=pytorch_y_pred,
            class_names=FASHION_MNIST_CLASSES,
            model_label="PyTorch NN",
            output_path=results_dir / "sample_predictions_pytorch.png",
            sample_count=sample_count,
            seed=sample_seed,
        ),
        "comparison_training_curves": plot_training_curves_combined(
            histories=histories,
            output_path=results_dir / "comparison_training_curves.png",
        ),
    }

    return artifacts