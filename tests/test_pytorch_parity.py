from __future__ import annotations

import numpy as np
import torch
from torch import nn

from custom_nn import NeuralNetwork
from custom_nn.model import Layer_Dense
from custom_nn.techniques import Batch_Normalization
from pytorch_nn import FashionMNISTNet

from tests.helpers import as_float64, build_batch_norm_test_config, build_linear_test_config, one_hot, tiny_probe_batch


def _sync_custom_parameters_to_pytorch(custom_model: NeuralNetwork, pytorch_model: FashionMNISTNet) -> None:
    custom_param_layers = [
        layer for layer in custom_model.layers if isinstance(layer, (Layer_Dense, Batch_Normalization))
    ]
    pytorch_param_layers = [
        layer for layer in pytorch_model.layers if isinstance(layer, (nn.Linear, nn.BatchNorm1d))
    ]

    assert len(custom_param_layers) == len(pytorch_param_layers)

    for custom_layer, torch_layer in zip(custom_param_layers, pytorch_param_layers, strict=True):
        if isinstance(custom_layer, Layer_Dense):
            assert isinstance(torch_layer, nn.Linear)

            torch_layer.weight.data = torch.from_numpy(custom_layer.weights.T.copy()).to(dtype=torch.float64)
            torch_layer.bias.data = torch.from_numpy(custom_layer.biases.reshape(-1).copy()).to(dtype=torch.float64)
        else:
            assert isinstance(custom_layer, Batch_Normalization)
            assert isinstance(torch_layer, nn.BatchNorm1d)
            assert torch_layer.running_mean is not None
            assert torch_layer.running_var is not None

            torch_layer.weight.data = torch.from_numpy(custom_layer.gamma.copy()).to(dtype=torch.float64)
            torch_layer.bias.data = torch.from_numpy(custom_layer.beta.copy()).to(dtype=torch.float64)
            torch_layer.running_mean.data = torch.from_numpy(np.asarray(custom_layer.running_mean).reshape(-1).copy()).to(dtype=torch.float64)
            torch_layer.running_var.data = torch.from_numpy(np.asarray(custom_layer.running_variance).reshape(-1).copy()).to(dtype=torch.float64)


def test_strict_custom_pytorch_inference_parity_with_synchronized_parameters() -> None:
    config = build_batch_norm_test_config(seed=1234)
    custom_model = NeuralNetwork(config)
    pytorch_model = FashionMNISTNet(config).double()

    custom_model.layers[0].weights = as_float64(
        np.array(
            [
                [0.21, -0.14, 0.05, 0.09],
                [-0.18, 0.31, 0.22, -0.27],
                [0.07, 0.11, -0.35, 0.16],
            ]
        )
    )
    custom_model.layers[0].biases = as_float64(np.array([[0.02, -0.03, 0.01, 0.04]]))

    custom_model.layers[1].gamma = as_float64(np.array([1.1, 0.9, 1.3, 0.8]))
    custom_model.layers[1].beta = as_float64(np.array([0.05, -0.02, 0.03, 0.01]))
    custom_model.layers[1].running_mean = as_float64(np.array([0.2, -0.1, 0.4, -0.3]))
    custom_model.layers[1].running_variance = as_float64(np.array([1.2, 0.7, 1.5, 0.9]))

    custom_model.layers[3].weights = as_float64(
        np.array(
            [
                [0.12, -0.28],
                [0.33, 0.07],
                [-0.21, 0.19],
                [0.05, -0.16],
            ]
        )
    )
    custom_model.layers[3].biases = as_float64(np.array([[-0.01, 0.02]]))

    _sync_custom_parameters_to_pytorch(custom_model, pytorch_model)

    inputs = tiny_probe_batch()
    custom_probs = custom_model.forward(inputs, training=False)

    pytorch_model.eval()
    torch_inputs = torch.from_numpy(inputs).to(dtype=torch.float64)
    pytorch_logits = pytorch_model(torch_inputs)
    pytorch_probs = torch.softmax(pytorch_logits, dim=1).detach().cpu().numpy()

    np.testing.assert_allclose(custom_probs, pytorch_probs, rtol=1e-10, atol=1e-12)


def test_looser_trained_behavior_parity_both_improve_and_agree_on_predictions() -> None:
    x_train = as_float64(
        np.array(
            [
                [2.0, 1.0, 0.5],
                [1.5, 0.8, 0.2],
                [1.7, 1.2, 0.4],
                [1.2, 0.6, 0.3],
                [-2.0, -1.1, -0.4],
                [-1.6, -0.9, -0.2],
                [-1.8, -1.3, -0.5],
                [-1.3, -0.7, -0.1],
            ]
        )
    )
    y_indices = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_train = one_hot(y_indices, num_classes=2)

    # Use linear config (dense -> softmax) to avoid batch-norm training divergence
    config = build_linear_test_config(epochs=2, batch_size=4, learning_rate=0.05, seed=1234)

    # Build both models from the same seed-determined initialization
    custom_model = NeuralNetwork(config)
    pytorch_model = FashionMNISTNet(config).double()

    # Synchronize initial weights so both start from identical parameters
    _sync_custom_parameters_to_pytorch(custom_model, pytorch_model)

    # Record pre-training losses
    custom_pre_loss = custom_model.evaluate(x_train, y_train)["loss"]

    y_train_indices = torch.from_numpy(y_indices).to(dtype=torch.int64)
    x_train_tensor = torch.from_numpy(x_train).to(dtype=torch.float64)
    dataset = torch.utils.data.TensorDataset(x_train_tensor, y_train_indices)
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False, drop_last=True)

    criterion = torch.nn.CrossEntropyLoss()
    pt_optimizer = torch.optim.Adam(
        pytorch_model.parameters(),
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_epsilon,
        weight_decay=config.l2_lambda,
    )

    pytorch_pre_loss = 0.0
    pytorch_model.eval()
    with torch.no_grad():
        for features, labels in loader:
            logits = pytorch_model(features)
            pytorch_pre_loss += criterion(logits, labels).item()
    pytorch_pre_loss /= len(loader)

    # Train both for exactly 2 epochs
    custom_model.train(x_train, y_train, x_train, y_train)

    pytorch_model.train()
    for _ in range(config.epochs):
        for features, labels in loader:
            pt_optimizer.zero_grad(set_to_none=True)
            logits = pytorch_model(features)
            loss = criterion(logits, labels)
            loss.backward()
            pt_optimizer.step()

    # Assert both losses decreased after training
    custom_post_loss = custom_model.evaluate(x_train, y_train)["loss"]

    pytorch_post_loss = 0.0
    pytorch_model.eval()
    with torch.no_grad():
        for features, labels in loader:
            logits = pytorch_model(features)
            pytorch_post_loss += criterion(logits, labels).item()
    pytorch_post_loss /= len(loader)

    assert custom_post_loss < custom_pre_loss, (
        f"Custom model loss did not decrease: {custom_pre_loss:.6f} -> {custom_post_loss:.6f}"
    )
    assert pytorch_post_loss < pytorch_pre_loss, (
        f"PyTorch model loss did not decrease: {pytorch_pre_loss:.6f} -> {pytorch_post_loss:.6f}"
    )

    # Assert both models agree on class predictions for unambiguous extreme inputs
    # (near-origin points are marginal after 2 epochs; use examples far from the boundary)
    probe = as_float64(
        np.array(
            [
                [2.0, 1.0, 0.5],
                [-2.0, -1.1, -0.4],
            ]
        )
    )
    custom_preds = custom_model.predict(probe)

    pytorch_model.eval()
    with torch.no_grad():
        pt_logits = pytorch_model(torch.from_numpy(probe).to(dtype=torch.float64))
    pytorch_preds = pt_logits.argmax(dim=1).numpy()

    assert np.array_equal(custom_preds, pytorch_preds), (
        f"Prediction disagreement: custom={custom_preds}, pytorch={pytorch_preds}"
    )
