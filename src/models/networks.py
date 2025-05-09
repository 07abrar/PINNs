import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralNet(nn.Module):
    """
    A fully connected neural network for PINNs.

    input_dim: Number of input features (e.g., spatial and temporal coordinates).
    hidden_dim: Number of neurons in each hidden layer.
    output_dim: Number of output features (e.g., solution of the PDE).
    num_hidden_layers: Number of hidden layers in the network.
    activation: Activation function to use (default is tanh).
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_hidden_layers: int,
        activation: str = "tanh",
    ) -> None:
        super(NeuralNet, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_hidden_layers)]
        )
        self.output_layer = nn.Linear(hidden_dim, output_dim)

        # Map activation function name to actual function
        activations = {
            "relu": F.relu,
            "tanh": torch.tanh,
            "sigmoid": torch.sigmoid,
            "softplus": F.softplus,
        }
        if activation not in activations:
            raise ValueError(f"Unsupported activation function: {activation}")
        self.activation = activations[activation]

    def forward(self, x):
        """
        Forward pass through the network.
        """
        x = self.activation(self.input_layer(x))  # Apply activation to the input layer
        for hidden_layer in self.hidden_layers:
            x = self.activation(hidden_layer(x))  # Apply activation to each hidden layer
        # No activation function is applied to the output layer
        # as it will restrict the output range
        x = self.output_layer(x)
        return x