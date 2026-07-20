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
]