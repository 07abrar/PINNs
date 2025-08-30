from typing import Dict

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
        problem,  # PDEProblem instance
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_hidden_layers: int,
        activation: str = "tanh",
        device: str = "cpu",
        dtype: torch.dtype = torch.float32
    ):
        super().__init__()

        self.problem = problem
        self.device = device
        self.dtype = dtype

        # Create neural network
        # self.network = NeuralNet(
        #     input_dim, hidden_dim, output_dim,
        #     num_hidden_layers, activation
        # ).to(device=device, dtype=dtype)
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim)
             for _ in range(num_hidden_layers)]
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

        # Loss function
        self.mse_loss = nn.MSELoss()

    def forward(self, x):
        """Forward pass through the network"""
        x = self.activation(self.input_layer(
            x))  # Apply activation to the input layer
        for hidden_layer in self.hidden_layers:
            # Apply activation to each hidden layer
            x = self.activation(hidden_layer(x))
        # No activation function is applied to the output layer
        # as it will restrict the output range
        x = self.output_layer(x)
        return x

    def compute_pde_loss(self, collocation_points: torch.Tensor) -> torch.Tensor:
        """Compute PDE residual loss"""
        coords = collocation_points.clone().detach().requires_grad_(True)
        u_pred = self.forward(coords)

        # Use the problem's residual function
        residual = self.problem.compute_residual(coords, u_pred)
        target = torch.zeros_like(residual)

        return self.mse_loss(residual, target)

    def compute_boundary_loss(
        self,
        boundary_points: torch.Tensor,
        boundary_values: torch.Tensor
    ) -> torch.Tensor:
        """Compute boundary condition loss"""
        u_pred = self.forward(boundary_points)
        return self.mse_loss(u_pred, boundary_values)

    def compute_total_loss(
        self,
        collocation_points: torch.Tensor,
        boundary_points: torch.Tensor,
        boundary_values: torch.Tensor,
        bc_weight: float = 10.0
    ) -> Dict[str, torch.Tensor]:
        """Compute total weighted loss"""
        pde_loss = self.compute_pde_loss(collocation_points)
        boundary_loss = self.compute_boundary_loss(
            boundary_points, boundary_values)
        total_loss = bc_weight * boundary_loss + pde_loss

        return {
            'total_loss': total_loss,
            'pde_loss': pde_loss,
            'boundary_loss': boundary_loss
        }
