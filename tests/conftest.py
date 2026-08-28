from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from custom_nn import NetworkConfig
from tests.helpers import (
    build_batch_norm_test_config,
    build_dense_relu_test_config,
    build_linear_test_config,
    build_reduced_network_config,
    tiny_classification_batch,
    tiny_probe_batch,
)


@pytest.fixture(autouse=True)
def deterministic_numpy_seed() -> None:
    np.random.seed(1234)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(1234)


@pytest.fixture
def float_tolerance() -> dict[str, float]:
    return {
        "rtol": 1e-7,
        "atol": 1e-9,
    }


@pytest.fixture
def tiny_batch() -> tuple[np.ndarray, np.ndarray]:
    return tiny_classification_batch()


@pytest.fixture
def probe_batch() -> np.ndarray:
    return tiny_probe_batch()


@pytest.fixture
def reduced_network_config() -> NetworkConfig:
    return build_reduced_network_config()


@pytest.fixture
def linear_test_config() -> NetworkConfig:
    return build_linear_test_config()


@pytest.fixture
def dense_relu_test_config() -> NetworkConfig:
    return build_dense_relu_test_config()


@pytest.fixture
def batch_norm_test_config() -> NetworkConfig:
    return build_batch_norm_test_config()