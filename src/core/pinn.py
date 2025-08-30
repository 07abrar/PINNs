import torch
import torch.nn as nn
from typing import Dict, Any

from ..models.networks import NeuralNet


class PINN(nn.Module):
    """
    Physics-Informed Neural Network - Core Implementation

    Responsibilities:
    - Neural network management
    - Loss computation (PDE + boundary)
    - Forward pass

    NOT responsible for:
    - Data generation (handled by Domain)
    - Training loops (handled by Trainer)
    - Optimization (handled by Trainer)
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
        self.network = NeuralNet(
            input_dim, hidden_dim, output_dim,
            num_hidden_layers, activation
        ).to(device=device, dtype=dtype)

        # Loss function
        self.mse_loss = nn.MSELoss()

    def forward(self, x):
        """Forward pass through the network"""
        return self.network(x)

    def compute_pde_loss(self, collocation_points: torch.Tensor) -> torch.Tensor:
        """Compute PDE residual loss"""
        coords = collocation_points.clone().detach().requires_grad_(True)
        u_pred = self.network(coords)

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
        u_pred = self.network(boundary_points)
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
