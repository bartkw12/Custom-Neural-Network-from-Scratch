"""Unified CLI for running custom, PyTorch, and comparison workflows."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from custom_nn.config import NetworkConfig, default_config
from custom_nn.data_preprocessing import load_fashion_MNIST, preprocess_data
from custom_nn.network import NeuralNetwork
from pytorch_nn.compare import generate_comparison_artifacts
from pytorch_nn.model import FashionMNISTNet
from pytorch_nn.train import evaluate_test_set, prepare_dataloaders, save_history, train_model


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _results_dir() -> Path:
    return _repo_root() / "results"


def _load_json_config(config_path: str | Path | None) -> dict[str, Any]:
    if config_path is None:
        return {}

    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as config_file:
        loaded = json.load(config_file)

    if not isinstance(loaded, dict):
        raise ValueError("Config file must contain a JSON object.")

    return loaded


def _build_config(args: argparse.Namespace) -> NetworkConfig:
    base = asdict(default_config())
    file_values = _load_json_config(getattr(args, "config", None))

    unknown_keys = sorted(set(file_values) - set(base))
    if unknown_keys:
        raise ValueError(f"Unsupported config keys: {', '.join(unknown_keys)}")

    merged: dict[str, Any] = {**base, **file_values}

    # Explicit CLI flags always override config-file values.
    cli_overrides = {
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "hidden_layers": args.hidden_layers,
        "hidden_units": args.hidden_units,
        "seed": args.seed,
    }
    for key, value in cli_overrides.items():
        if value is not None:
            merged[key] = value

    if getattr(args, "no_early_stopping", False):
        merged["patience"] = max(int(merged["epochs"]) + 1, 1)

    return NetworkConfig(**merged)


def _add_training_subcommand_arguments(parser: argparse.ArgumentParser, *, default_save_file: str) -> None:
    parser.add_argument("--config", type=Path, default=None, help="Path to a JSON config file.")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs.")
    parser.add_argument("--learning-rate", type=float, default=None, help="Learning rate override.")
    parser.add_argument("--batch-size", type=int, default=None, help="Mini-batch size override.")
    parser.add_argument("--hidden-layers", type=int, default=None, help="Number of hidden layers.")
    parser.add_argument("--hidden-units", type=int, default=None, help="Number of units per hidden layer.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed override.")
    parser.add_argument("--no-early-stopping", action="store_true", help="Disable early stopping for this run.")
    parser.add_argument(
        "--save-path",
        type=Path,
        default=_results_dir() / default_save_file,
        help="Path where run history JSON will be written.",
    )


def _run_custom(args: argparse.Namespace) -> int:
    config = _build_config(args)
    model = NeuralNetwork(config)

    train_dataset, test_dataset = load_fashion_MNIST(seed=config.seed)
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = preprocess_data(train_dataset, test_dataset)

    history = model.train(x_train, y_train, x_val, y_val)
    args.save_path.parent.mkdir(parents=True, exist_ok=True)
    with args.save_path.open("w", encoding="utf-8") as history_file:
        json.dump(history, history_file, indent=2)

    train_metrics = model.evaluate(x_train, y_train)
    test_metrics = model.evaluate(x_test, y_test)

    print(f"history_path: {args.save_path}")
    print(f"Final Training Misclassification Error: {100 * (1.0 - train_metrics['accuracy']):.2f} %")
    print(f"Final Test Misclassification Error: {100 * (1.0 - test_metrics['accuracy']):.2f} %")
    return 0


def _run_pytorch(args: argparse.Namespace) -> int:
    config = _build_config(args)
    train_loader, val_loader, test_loader = prepare_dataloaders(config)

    model = FashionMNISTNet(config)
    history = train_model(model, train_loader, val_loader, config)
    save_history(history, args.save_path)
    metrics = evaluate_test_set(model, test_loader)

    print(f"history_path: {args.save_path}")
    print(f"Final Test Misclassification Error: {100 * metrics['misclassification_error']:.2f} %")
    return 0


def _run_compare(args: argparse.Namespace) -> int:
    artifacts = generate_comparison_artifacts(
        custom_test_error=args.custom_test_error,
        pytorch_test_error=args.pytorch_test_error,
        custom_history_path=args.custom_history,
        pytorch_history_path=args.pytorch_history,
    )

    for name, path in artifacts.items():
        print(f"{name}: {path}")

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Custom NN experiment CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    custom_parser = subparsers.add_parser("custom", help="Train and evaluate the NumPy custom model.")
    _add_training_subcommand_arguments(custom_parser, default_save_file="custom_nn_history.json")
    custom_parser.set_defaults(handler=_run_custom)

    pytorch_parser = subparsers.add_parser("pytorch", help="Train and evaluate the PyTorch reference model.")
    _add_training_subcommand_arguments(pytorch_parser, default_save_file="pytorch_history.json")
    pytorch_parser.set_defaults(handler=_run_pytorch)

    compare_parser = subparsers.add_parser("compare", help="Generate comparison artifacts from saved run data.")
    compare_parser.add_argument(
        "--custom-test-error",
        type=float,
        required=True,
        help="Custom model test error percentage.",
    )
    compare_parser.add_argument(
        "--pytorch-test-error",
        type=float,
        required=True,
        help="PyTorch model test error percentage.",
    )
    compare_parser.add_argument("--custom-history", type=Path, default=None, help="Path to custom history JSON.")
    compare_parser.add_argument("--pytorch-history", type=Path, default=None, help="Path to PyTorch history JSON.")
    compare_parser.set_defaults(handler=_run_compare)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
