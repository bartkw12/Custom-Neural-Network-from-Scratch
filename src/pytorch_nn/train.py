"""Training entry point for the PyTorch comparison implementation."""

import copy
import json
import random
from pathlib import Path

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


def build_optimizer(
	parameters,
	config: NetworkConfig | None = None,
) -> torch.optim.Adam:
	config = config if config is not None else default_config()
	return torch.optim.Adam(
		parameters,
		lr=config.learning_rate,
		betas=(config.adam_beta1, config.adam_beta2),
		eps=config.adam_epsilon,
		weight_decay=config.l2_lambda,
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


def train_model(
	model: torch.nn.Module,
	train_loader: DataLoader,
	val_loader: DataLoader,
	config: NetworkConfig | None = None,
	device: torch.device | None = None,
) -> dict[str, list[float]]:
	config = config if config is not None else default_config()
	device = device if device is not None else get_device()
	model = model.to(device)

	criterion = torch.nn.CrossEntropyLoss()
	optimizer = build_optimizer(model.parameters(), config)
	scheduler = build_lr_scheduler(optimizer, config)
	history = {
		"train_loss": [],
		"val_loss": [],
	}
	best_val_loss = float("inf")
	best_state = None
	wait = 0

	for _ in range(config.epochs):
		model.train()
		train_loss_total = 0.0
		train_sample_count = 0

		for features, labels in train_loader:
			features = features.to(device)
			labels = labels.to(device)

			optimizer.zero_grad(set_to_none=True)
			logits = model(features)
			loss = criterion(logits, labels)
			loss.backward()
			optimizer.step()
			scheduler.step()

			batch_size = labels.shape[0]
			train_loss_total += loss.item() * batch_size
			train_sample_count += batch_size

		avg_train_loss = train_loss_total / train_sample_count

		model.eval()
		val_loss_total = 0.0
		val_sample_count = 0

		with torch.no_grad():
			for features, labels in val_loader:
				features = features.to(device)
				labels = labels.to(device)

				logits = model(features)
				loss = criterion(logits, labels)

				batch_size = labels.shape[0]
				val_loss_total += loss.item() * batch_size
				val_sample_count += batch_size

		avg_val_loss = val_loss_total / val_sample_count
		history["train_loss"].append(avg_train_loss)
		history["val_loss"].append(avg_val_loss)

		if avg_val_loss < best_val_loss - config.min_delta:
			best_val_loss = avg_val_loss
			best_state = copy.deepcopy(model.state_dict())
			wait = 0
		else:
			wait += 1
			if wait >= config.patience:
				if best_state is not None:
					model.load_state_dict(best_state)
				break

	return history


def evaluate_split(
	model: torch.nn.Module,
	data_loader: DataLoader,
	device: torch.device | None = None,
) -> dict[str, float]:
	device = device if device is not None else get_device()
	model = model.to(device)
	criterion = torch.nn.CrossEntropyLoss()

	model.eval()
	total_loss = 0.0
	total_samples = 0
	correct_predictions = 0

	with torch.no_grad():
		for features, labels in data_loader:
			features = features.to(device)
			labels = labels.to(device)

			logits = model(features)
			loss = criterion(logits, labels)
			predictions = logits.argmax(dim=1)

			batch_size = labels.shape[0]
			total_loss += loss.item() * batch_size
			total_samples += batch_size
			correct_predictions += (predictions == labels).sum().item()

	accuracy = correct_predictions / total_samples
	return {
		"loss": total_loss / total_samples,
		"accuracy": accuracy,
		"misclassification_error": 1.0 - accuracy,
	}


def evaluate_test_set(
	model: torch.nn.Module,
	test_loader: DataLoader,
	device: torch.device | None = None,
) -> dict[str, float]:
	metrics = evaluate_split(model, test_loader, device=device)

	print(f"Final Test Misclassification Error: {100 * metrics['misclassification_error']:.2f} %")
	return metrics


def save_history(
	history: dict[str, list[float]],
	output_path: str | Path | None = None,
) -> Path:
	if output_path is None:
		output_path = Path(__file__).resolve().parents[2] / "results" / "pytorch_history.json"
	else:
		output_path = Path(output_path)

	output_path.parent.mkdir(parents=True, exist_ok=True)
	with output_path.open("w", encoding="utf-8") as history_file:
		json.dump(history, history_file, indent=2)

	return output_path


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