from __future__ import annotations

from torch import Tensor, nn

from custom_nn.config import NetworkConfig, default_config


class FashionMNISTNet(nn.Module):
    def __init__(self, config: NetworkConfig | None = None):
        super().__init__()
        self.config = config if config is not None else default_config()
        self.layers = nn.ModuleList(self._build_layers())
        self.reset_parameters()

    def _build_layers(self) -> list[nn.Module]:
        layers: list[nn.Module] = []

        for spec in self.config.get_layer_specs():
            layer_type = spec["type"]

            if layer_type == "dense":
                layers.append(nn.Linear(spec["n_inputs"], spec["n_neurons"]))
            elif layer_type == "batch_norm":
                layers.append(nn.BatchNorm1d(spec["n_neurons"], eps=spec.get("epsilon", self.config.bn_epsilon)))
            elif layer_type == "relu":
                layers.append(nn.ReLU())
            elif layer_type == "dropout":
                dropout_rate = (
                    spec.get("dropout_rate_input", self.config.dropout_rate_input)
                    if spec.get("input_layer", False)
                    else spec.get("dropout_rate_hidden", self.config.dropout_rate_hidden)
                )
                layers.append(nn.Dropout(p=dropout_rate))
            elif layer_type == "softmax":
                continue
            else:
                raise ValueError(f"Unsupported layer type: {layer_type}")

        return layers

    def reset_parameters(self) -> None:
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0.0, std=0.01)
                nn.init.zeros_(layer.bias)

    def forward(self, inputs: Tensor) -> Tensor:
        outputs = inputs

        if outputs.dim() > 2:
            outputs = outputs.flatten(start_dim=1)

        for layer in self.layers:
            outputs = layer(outputs)

        return outputs