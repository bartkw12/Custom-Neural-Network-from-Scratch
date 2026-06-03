"""Training entry point for the PyTorch comparison implementation."""

import random

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from custom_nn.config import NetworkConfig, default_config
from custom_nn.data_preprocessing import load_fashion_MNIST, preprocess_data


def get_device() -> torch.device:
	if torch.cuda.is_available():
		return torch.device("cuda")

	mps_backend = getattr(torch.backends, "mps", None)
	if mps_backend is not None and mps_backend.is_available():
		return torch.device("mps")

	return torch.device("cpu")


def seed_everything(seed: int) -> None:
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)

	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)

	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False


def _learning_rate_decay_factor(iteration: int, decay: float) -> float:
	return 1.0 / (1.0 + decay * iteration)


def build_lr_scheduler(
	optimizer: torch.optim.Optimizer,
	config: NetworkConfig | None = None,
) -> torch.optim.lr_scheduler.LambdaLR:
	config = config if config is not None else default_config()
	return torch.optim.lr_scheduler.LambdaLR(
		optimizer,
		lr_lambda=lambda iteration: _learning_rate_decay_factor(iteration, config.adam_decay),
	)


def _build_dataloader(
	features: torch.Tensor,
	labels: torch.Tensor,
	*,
	batch_size: int,
	shuffle: bool,
	drop_last: bool,
) -> DataLoader:
	dataset = TensorDataset(features, labels)
	return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)


def _to_class_indices(labels: torch.Tensor) -> torch.Tensor:
	return labels.argmax(dim=1).to(dtype=torch.int64)


def prepare_dataloaders(config: NetworkConfig | None = None) -> tuple[DataLoader, DataLoader, DataLoader]:
	config = config if config is not None else default_config()
	seed_everything(config.seed)

	train_dataset, test_dataset = load_fashion_MNIST(seed=config.seed)
	(x_train, y_train), (x_val, y_val), (x_test, y_test) = preprocess_data(train_dataset, test_dataset)

	x_train_tensor = torch.from_numpy(x_train).to(dtype=torch.float32)
	y_train_tensor = _to_class_indices(torch.from_numpy(y_train))
	x_val_tensor = torch.from_numpy(x_val).to(dtype=torch.float32)
	y_val_tensor = _to_class_indices(torch.from_numpy(y_val))
	x_test_tensor = torch.from_numpy(x_test).to(dtype=torch.float32)
	y_test_tensor = _to_class_indices(torch.from_numpy(y_test))

	train_loader = _build_dataloader(
		x_train_tensor,
		y_train_tensor,
		batch_size=config.batch_size,
		shuffle=True,
		drop_last=True,
	)
	val_loader = _build_dataloader(
		x_val_tensor,
		y_val_tensor,
		batch_size=config.batch_size,
		shuffle=False,
		drop_last=False,
	)
	test_loader = _build_dataloader(
		x_test_tensor,
		y_test_tensor,
		batch_size=config.batch_size,
		shuffle=False,
		drop_last=False,
	)

	return train_loader, val_loader, test_loader