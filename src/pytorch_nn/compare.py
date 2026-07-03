"""Comparison helpers for loading saved training histories and generating plots."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def _results_dir() -> Path:
	return Path(__file__).resolve().parents[2] / "results"


def _load_history(history_path: str | Path) -> dict[str, list[float]]:
	history_path = Path(history_path)
	with history_path.open("r", encoding="utf-8") as history_file:
		return json.load(history_file)


def _save_plot(figure: plt.Figure, output_path: str | Path) -> Path:
	output_path = Path(output_path)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	figure.savefig(output_path, dpi=200, bbox_inches="tight")
	plt.close(figure)
	return output_path


def load_pytorch_history(history_path: str | Path | None = None) -> dict[str, list[float]]:
	if history_path is None:
		history_path = _results_dir() / "pytorch_history.json"

	return _load_history(history_path)


def load_comparison_histories(
	custom_history_path: str | Path | None = None,
	pytorch_history_path: str | Path | None = None,
) -> dict[str, dict[str, list[float]]]:
	if custom_history_path is None:
		custom_history_path = _results_dir() / "custom_nn_history.json"

	return {
		"custom_nn": _load_history(custom_history_path),
		"pytorch_nn": load_pytorch_history(pytorch_history_path),
	}


def plot_training_loss_comparison(
	histories: dict[str, dict[str, list[float]]],
	output_path: str | Path | None = None,
) -> Path:
	if output_path is None:
		output_path = _results_dir() / "comparison_train_loss.png"

	figure, axis = plt.subplots(figsize=(10, 6))
	custom_train = histories["custom_nn"]["train_loss"]
	pytorch_train = histories["pytorch_nn"]["train_loss"]
	axis.plot(range(1, len(custom_train) + 1), custom_train, label="Custom NN", linewidth=2)
	axis.plot(range(1, len(pytorch_train) + 1), pytorch_train, label="PyTorch NN", linewidth=2)
	axis.set_xlabel("Epoch")
	axis.set_ylabel("Cross-entropy Loss")
	axis.set_title("Training Loss Comparison")
	axis.legend()
	axis.grid(alpha=0.3)
	return _save_plot(figure, output_path)


def plot_validation_loss_comparison(
	histories: dict[str, dict[str, list[float]]],
	output_path: str | Path | None = None,
) -> Path:
	if output_path is None:
		output_path = _results_dir() / "comparison_val_loss.png"

	figure, axis = plt.subplots(figsize=(10, 6))
	custom_val = histories["custom_nn"]["val_loss"]
	pytorch_val = histories["pytorch_nn"]["val_loss"]
	axis.plot(range(1, len(custom_val) + 1), custom_val, label="Custom NN", linewidth=2)
	axis.plot(range(1, len(pytorch_val) + 1), pytorch_val, label="PyTorch NN", linewidth=2)
	axis.set_xlabel("Epoch")
	axis.set_ylabel("Cross-entropy Loss")
	axis.set_title("Validation Loss Comparison")
	axis.legend()
	axis.grid(alpha=0.3)
	return _save_plot(figure, output_path)


def create_accuracy_table(
	custom_test_error: float,
	pytorch_test_error: float,
	output_path: str | Path | None = None,
) -> Path:
	if output_path is None:
		output_path = _results_dir() / "comparison_accuracy_table.png"

	custom_accuracy = 100.0 - custom_test_error
	pytorch_accuracy = 100.0 - pytorch_test_error

	figure, axis = plt.subplots(figsize=(8, 2.8))
	axis.axis("off")
	table = axis.table(
		cellText=[
			["Custom NN", f"{custom_accuracy:.2f}%", f"{custom_test_error:.2f}%"],
			["PyTorch NN", f"{pytorch_accuracy:.2f}%", f"{pytorch_test_error:.2f}%"],
		],
		colLabels=["Model", "Test Accuracy", "Test Error"],
		loc="center",
		cellLoc="center",
	)
	table.auto_set_font_size(False)
	table.set_fontsize(11)
	table.scale(1.2, 1.6)
	axis.set_title("Final Test Accuracy Comparison", pad=12)
	return _save_plot(figure, output_path)


def generate_comparison_artifacts(
	custom_test_error: float,
	pytorch_test_error: float,
	custom_history_path: str | Path | None = None,
	pytorch_history_path: str | Path | None = None,
) -> dict[str, Path]:
	histories = load_comparison_histories(custom_history_path, pytorch_history_path)
	return {
		"train_loss_plot": plot_training_loss_comparison(histories),
		"val_loss_plot": plot_validation_loss_comparison(histories),
		"accuracy_table": create_accuracy_table(custom_test_error, pytorch_test_error),
	}


def _build_argument_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Generate comparison plots for custom_nn vs pytorch_nn.")
	parser.add_argument("--custom-test-error", type=float, required=True, help="Custom NN test error percentage.")
	parser.add_argument("--pytorch-test-error", type=float, required=True, help="PyTorch NN test error percentage.")
	parser.add_argument("--custom-history", type=Path, default=None, help="Path to custom_nn_history.json.")
	parser.add_argument("--pytorch-history", type=Path, default=None, help="Path to pytorch_history.json.")
	return parser


def main() -> None:
	parser = _build_argument_parser()
	args = parser.parse_args()
	artifacts = generate_comparison_artifacts(
		custom_test_error=args.custom_test_error,
		pytorch_test_error=args.pytorch_test_error,
		custom_history_path=args.custom_history,
		pytorch_history_path=args.pytorch_history,
	)

	for name, path in artifacts.items():
		print(f"{name}: {path}")


if __name__ == "__main__":
	main()