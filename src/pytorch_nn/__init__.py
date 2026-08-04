"""PyTorch comparison package."""

from .model import FashionMNISTNet
from .train import (
	build_lr_scheduler,
	build_optimizer,
	evaluate_split,
	evaluate_test_set,
	get_device,
	prepare_dataloaders,
	save_history,
	seed_everything,
	train_model,
)


def load_pytorch_history(history_path=None):
	from .compare import load_pytorch_history as _load_pytorch_history

	return _load_pytorch_history(history_path)


def load_comparison_histories(custom_history_path=None, pytorch_history_path=None):
	from .compare import load_comparison_histories as _load_comparison_histories

	return _load_comparison_histories(custom_history_path, pytorch_history_path)


def compute_confusion_matrix(y_true, y_pred, num_classes=10):
	from .analyze import compute_confusion_matrix as _compute_confusion_matrix

	return _compute_confusion_matrix(y_true, y_pred, num_classes=num_classes)


def compute_per_class_accuracy(y_true, y_pred, num_classes=10):
	from .analyze import compute_per_class_accuracy as _compute_per_class_accuracy

	return _compute_per_class_accuracy(y_true, y_pred, num_classes=num_classes)


def generate_analysis_artifacts(
	custom_summary_path=None,
	pytorch_summary_path=None,
	sample_count=25,
	sample_seed=None,
):
	from .analyze import generate_analysis_artifacts as _generate_analysis_artifacts

	return _generate_analysis_artifacts(
		custom_summary_path=custom_summary_path,
		pytorch_summary_path=pytorch_summary_path,
		sample_count=sample_count,
		sample_seed=sample_seed,
	)

__all__ = [
	"FashionMNISTNet",
	"get_device",
	"seed_everything",
	"build_lr_scheduler",
	"build_optimizer",
	"prepare_dataloaders",
	"train_model",
	"evaluate_split",
	"evaluate_test_set",
	"save_history",
	"load_pytorch_history",
	"load_comparison_histories",
	"compute_confusion_matrix",
	"compute_per_class_accuracy",
	"generate_analysis_artifacts",
]