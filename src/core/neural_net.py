"""Neural network architectures for PINNs."""

import torch
import torch.nn as nn
import torch.nn.functional as F


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute network outputs for the given input coordinates."""
        x = self.activation(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = self.activation(hidden_layer(x))
        return self.output_layer(x)
