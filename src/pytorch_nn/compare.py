"""Comparison helpers for loading saved training histories."""

import json
from pathlib import Path


def _results_dir() -> Path:
	return Path(__file__).resolve().parents[2] / "results"


def _load_history(history_path: str | Path) -> dict[str, list[float]]:
	history_path = Path(history_path)
	with history_path.open("r", encoding="utf-8") as history_file:
		return json.load(history_file)


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