import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralNet(nn.Module):
    """
    A fully connected neural network for PINNs.
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
    ):
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
            raise ValueError(f"Unsupported activation function: {activation}")
        self.activation = activations[activation]

        # Loss function
        self.mse_loss = nn.MSELoss()

    def forward(self, x):
        """Forward pass through the network"""
        x = self.activation(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = self.activation(hidden_layer(x))
        x = self.output_layer(x)
        return x
