from abc import ABC, abstractmethod
from typing import Dict, Any, List
import torch
import time


class TrainingStrategy(ABC):
    """Base class for training strategies"""

    @abstractmethod
    def train(self, pinn, optimizer, boundary_points, boundary_values,
              collocation_points, **kwargs) -> Dict[str, Any]:
        """Execute the training strategy"""
        pass


class StandardStrategy(TrainingStrategy):
    """Standard PINN training strategy"""

    def train(
        self,
        pinn,
        optimizer,
        boundary_points,
        boundary_values,
        collocation_points,
        epochs: int = 1000,
        loss_threshold: float = 1e-4,
        bc_weight: float = 10.0,
        print_every: int = 100,
        **kwargs
    ) -> Dict[str, Any]:
        """Standard training loop"""

        pinn.train()
        loss_history = []
        start_time = time.time()

        print("Starting training...")
        print("Epoch - Total Loss - PDE Loss - Boundary Loss")
        print("-" * 50)

        for epoch in range(epochs):
            optimizer.zero_grad()

            # Compute losses
            losses = pinn.compute_total_loss(
                collocation_points, boundary_points, boundary_values, bc_weight
            )

            # Backward pass
            losses['total_loss'].backward()
            optimizer.step()

            # Record loss
            current_loss = losses['total_loss'].item()
            loss_history.append(current_loss)

            # Print progress
            if epoch % print_every == 0:
                print(f"{epoch:5d} - {losses['total_loss'].item():.6f} - "
                      f"{losses['pde_loss'].item():.6f} - {losses['boundary_loss'].item():.6f}")

            # Check convergence
            if current_loss < loss_threshold:
                print(
                    f"Converged at epoch {epoch} with loss {current_loss:.6f}")
                break

        elapsed = time.time() - start_time
        pinn.eval()

        return {
            'loss_history': loss_history,
            'final_loss': loss_history[-1] if loss_history else float('inf'),
            'epochs_completed': len(loss_history),
            'training_time': elapsed,
            'converged': loss_history[-1] < loss_threshold if loss_history else False
        }


class AdaptiveSamplingStrategy(TrainingStrategy):
    """Adaptive sampling training strategy - placeholder"""

    def train(self, **kwargs) -> Dict[str, Any]:
        """Placeholder for adaptive sampling"""
        print("Adaptive sampling strategy not yet implemented")
        # Fall back to standard strategy for now
        return StandardStrategy().train(**kwargs)
