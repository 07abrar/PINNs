"""Neural network architectures for PINNs."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional

from src.utils.net2net import widen_linear


class NeuralNet(nn.Module):
    """Fully connected neural network used in PINNs.

    Parameters:
        input_dim: Number of input features.
        hidden_dim: Width of each hidden layer.
        output_dim: Number of output features.
        num_hidden_layers: Number of hidden layers.
        activation: Activation function name (`tanh`, `relu`, ...).
        device: Device on which tensors should be allocated.
        dtype: Default tensor data type.
        state_dict_path: Optional path to a checkpoint for initialization.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_hidden_layers: int,
        activation: str = "tanh",
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        state_dict_path: Optional[str] = None,
    ) -> None:
        super().__init__()

        self.device = device
        self.dtype = dtype
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList(
            [
                nn.Linear(hidden_dim, hidden_dim)
                for _ in range(num_hidden_layers)
            ]
        )
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        activations = {
            "relu": F.relu,
            "tanh": torch.tanh,
            "sigmoid": torch.sigmoid,
            "softplus": F.softplus,
        }
        if activation not in activations:
            supported_activations = list(activations.keys())
            raise ValueError(
                f"Unsupported activation function: '{activation}'. "
                f"Supported activations are: {supported_activations}"
            )
        self.activation = activations[activation]

        # Default loss function used by training strategies
        self.mse_loss = nn.MSELoss()

        self.to(device=device, dtype=dtype)

        if state_dict_path is not None:
            self._initialize_from_checkpoint(state_dict_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute network outputs for the given input coordinates."""
        x = self.activation(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = self.activation(hidden_layer(x))
        return self.output_layer(x)

    # ------------------------------------------------------------------
    def _initialize_from_checkpoint(self, path: str) -> None:
        """Load weights from a checkpoint, widening layers if needed."""
        checkpoint = torch.load(path, map_location=self.device)
        state_dict = checkpoint.get("model_state", checkpoint)

        # Load matching parameters
        self.load_state_dict(state_dict, strict=False)

        layers = [self.input_layer] + list(self.hidden_layers) + [self.output_layer]
        names = ["input_layer"] + [f"hidden_layers.{i}" for i in range(len(self.hidden_layers))] + ["output_layer"]

        for i in range(len(layers) - 1):
            name, next_name = names[i], names[i + 1]
            w_old = state_dict.get(f"{name}.weight")
            if w_old is None:
                continue
            h_old = w_old.shape[0]
            h_new = layers[i].weight.shape[0]
            add_k = h_new - h_old
            if add_k <= 0:
                continue

            b_old = state_dict.get(f"{name}.bias")
            layers[i].weight.data[:h_old, :w_old.shape[1]] = w_old
            if b_old is not None:
                layers[i].bias.data[:h_old] = b_old

            wnext_old = state_dict.get(f"{next_name}.weight")
            if wnext_old is None:
                continue
            layers[i + 1].weight.data[:, :wnext_old.shape[1]] = wnext_old

            widen_linear(layers[i], layers[i + 1], add_k)
