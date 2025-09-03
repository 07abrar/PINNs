from abc import ABC, abstractmethod
import time
from typing import Dict, Any

import torch
from torch import optim

from src.core.neural_net import NeuralNet
from src.core.problem import PDEProblem


class TrainingStrategy(ABC):
    """Base class for training strategies"""

    @abstractmethod
    def train(
        self,
        networks: NeuralNet,
        problem: PDEProblem,
        optimizer: optim.Optimizer,
        boundary_points: torch.Tensor,
        boundary_values: torch.Tensor,
        collocation_points: torch.Tensor,
        **kwargs,
    ) -> Dict[str, Any]:
        """Execute the training strategy"""


class StandardStrategy(TrainingStrategy):
    """Standard PINN training strategy"""

    def train(
        self,
        networks: NeuralNet,
        problem: PDEProblem,
        optimizer: optim.Optimizer,
        boundary_points: torch.Tensor,
        boundary_values: torch.Tensor,
        collocation_points: torch.Tensor,
        epochs: int = 1000,
        loss_threshold: float = 1e-4,
        bc_weight: float = 10.0,
        print_every: int = 100,
        **kwargs,
    ) -> Dict[str, Any]:
        """Standard training loop"""

        networks.train()
        loss_history = []
        start_time = time.time()

        print("Starting training...")
        print("Epoch - Total Loss - PDE Loss - Boundary Loss")
        print("-" * 50)

        for epoch in range(epochs):
            optimizer.zero_grad()

            # Compute losses
            losses = self.__compute_total_loss(
                networks,
                problem,
                collocation_points,
                boundary_points,
                boundary_values,
                bc_weight,
            )

            # Backward pass
            losses["total_loss"].backward()
            optimizer.step()

            # Record loss
            current_loss = losses["total_loss"].item()
            loss_history.append(current_loss)

            # Print progress
            if epoch % print_every == 0:
                print(
                    f"{epoch:5d} - {losses['total_loss'].item():.6f} - "
                    f"{losses['pde_loss'].item():.6f} - {losses['boundary_loss'].item():.6f}"
                )

            # Check convergence
            if current_loss < loss_threshold:
                print(
                    f"Converged at epoch {epoch} with loss {current_loss:.6f}"
                )
                break

        elapsed = time.time() - start_time
        networks.eval()

        return {
            "loss_history": loss_history,
            "final_loss": loss_history[-1] if loss_history else float("inf"),
            "epochs_completed": len(loss_history),
            "training_time": elapsed,
            "converged": (
                loss_history[-1] < loss_threshold if loss_history else False
            ),
        }

    @staticmethod
    def __compute_pde_loss(
        networks: NeuralNet,
        problem: PDEProblem,
        collocation_points: torch.Tensor,
    ) -> torch.Tensor:
        """Compute PDE residual loss"""
        coords = collocation_points.clone().detach().requires_grad_(True)
        u_pred = networks.forward(coords)

        # Use the problem's residual function
        residual = problem.compute_residual(coords, u_pred)
        target = torch.zeros_like(residual)

        return networks.mse_loss(residual, target)

    @staticmethod
    def __compute_boundary_loss(
        networks: NeuralNet,
        boundary_points: torch.Tensor,
        boundary_values: torch.Tensor,
    ) -> torch.Tensor:
        """Compute boundary condition loss"""
        u_pred = networks.forward(boundary_points)
        return networks.mse_loss(u_pred, boundary_values)

    @staticmethod
    def __compute_total_loss(
        networks: NeuralNet,
        problem: PDEProblem,
        collocation_points: torch.Tensor,
        boundary_points: torch.Tensor,
        boundary_values: torch.Tensor,
        bc_weight: float = 10.0,
    ) -> Dict[str, torch.Tensor]:
        """Compute total weighted loss"""
        pde_loss = StandardStrategy.__compute_pde_loss(
            networks, problem, collocation_points
        )
        boundary_loss = StandardStrategy.__compute_boundary_loss(
            networks, boundary_points, boundary_values
        )
        total_loss = bc_weight * boundary_loss + pde_loss

        return {
            "total_loss": total_loss,
            "pde_loss": pde_loss,
            "boundary_loss": boundary_loss,
        }


class AdaptiveSamplingStrategy(TrainingStrategy):
    """Adaptive sampling training strategy - placeholder"""

    def train(self, **kwargs) -> Dict[str, Any]:
        """Placeholder for adaptive sampling"""
        print("Adaptive sampling strategy not yet implemented")
        # Fall back to standard strategy for now
        return StandardStrategy().train(**kwargs)
