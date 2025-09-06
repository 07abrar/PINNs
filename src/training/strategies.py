"""Training strategy implementations."""

from abc import ABC, abstractmethod
import math
import time
from typing import Any, Dict, List


import torch
from torch import optim
from tqdm.auto import tqdm

from src.core.neural_net import NeuralNet
from src.core.problem import PDEProblem


class TrainingStrategy(ABC):
    """Base class for training strategies"""

    @abstractmethod
    def train(
        self,
        model: NeuralNet,
        problem: PDEProblem,
        optimizer: optim.Optimizer,
        boundary_points: torch.Tensor,
        boundary_values: torch.Tensor,
        collocation_points: torch.Tensor,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Execute the training strategy"""


class StandardStrategy(TrainingStrategy):
    """Standard PINN training strategy"""

    def train(
        self,
        model: NeuralNet,
        problem: PDEProblem,
        optimizer: optim.Optimizer,
        boundary_points: torch.Tensor,
        boundary_values: torch.Tensor,
        collocation_points: torch.Tensor,
        epochs: int = 1000,
        loss_threshold: float = 1e-4,
        bc_weight: float = 10.0,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Standard training loop"""

        model.train()
        loss_history: List[float] = []
        start_time = time.time()

        # Display smoothing + cadence
        ema_alpha = 0.1
        show_every = 1
        pbar = tqdm(
            total=1.0,
            desc=f"Loss → {loss_threshold:.1e}",
            unit="frac",
            dynamic_ncols=True,
            mininterval=0.3,
            leave=False,
        )
        ema = None
        baseline = None  # first EMA, anchors the progress scale
        denom = None

        for epoch in range(epochs):
            optimizer.zero_grad()

            # Compute losses
            losses = self.__compute_total_loss(
                model,
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
            val = losses["total_loss"].item()
            loss_history.append(val)

            # EMA for stable display + stopping
            ema = val if ema is None else (
                1 - ema_alpha) * ema + ema_alpha * val

            # Initialize progress scale on first step
            if baseline is None:
                # Ensure baseline is above threshold to avoid zero denom
                baseline = max(ema, loss_threshold * 1.01)
                denom = max(1e-12, math.log(baseline) -
                            math.log(loss_threshold))

            # Fraction toward threshold on log scale, clipped to [0,1]
            frac = (math.log(baseline) - math.log(max(ema, 1e-20))) / denom
            frac = 0.0 if math.isnan(frac) else max(0.0, min(1.0, frac))

            if epoch % show_every == 0:
                pbar.set_postfix(
                    # ema=f"{ema:.2e}",
                    total=f"{val:.2e}",
                    pde=f"{losses['pde_loss'].item():.2e}",
                    bc=f"{losses['boundary_loss'].item():.2e}",
                )
            # set bar to current fraction (update expects a delta)
            pbar.update(frac - pbar.n)

            # Convergence on EMA to avoid flicker
            if ema < loss_threshold:
                msg = f"Converged at epoch {epoch} with EMA {ema:.3e} < {loss_threshold:.3e}"
                pbar.set_postfix_str("early_stop")
                tqdm.write(msg)
                break

        pbar.close()

        elapsed = time.time() - start_time
        model.eval()

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
        model: NeuralNet,
        problem: PDEProblem,
        collocation_points: torch.Tensor,
    ) -> torch.Tensor:
        """Compute PDE residual loss"""
        coords = collocation_points.clone().detach().requires_grad_(True)
        u_pred = model.forward(coords)

        # Use the problem's residual function
        residual = problem.compute_residual(coords, u_pred)
        target = torch.zeros_like(residual)

        return model.mse_loss(residual, target)

    @staticmethod
    def __compute_boundary_loss(
        model: NeuralNet,
        boundary_points: torch.Tensor,
        boundary_values: torch.Tensor,
    ) -> torch.Tensor:
        """Compute boundary condition loss"""
        u_pred = model.forward(boundary_points)
        return model.mse_loss(u_pred, boundary_values)

    @staticmethod
    def __compute_total_loss(
        model: NeuralNet,
        problem: PDEProblem,
        collocation_points: torch.Tensor,
        boundary_points: torch.Tensor,
        boundary_values: torch.Tensor,
        bc_weight: float = 10.0,
    ) -> Dict[str, torch.Tensor]:
        """Compute total weighted loss"""
        pde_loss = StandardStrategy.__compute_pde_loss(
            model, problem, collocation_points
        )
        boundary_loss = StandardStrategy.__compute_boundary_loss(
            model, boundary_points, boundary_values
        )
        total_loss = bc_weight * boundary_loss + pde_loss

        return {
            "total_loss": total_loss,
            "pde_loss": pde_loss,
            "boundary_loss": boundary_loss,
        }


class AdaptiveSamplingStrategy(TrainingStrategy):
    """Adaptive sampling training strategy - placeholder"""

    def train(self, **kwargs: Any) -> Dict[str, Any]:
        """Placeholder for adaptive sampling"""
        print("Adaptive sampling strategy not yet implemented")
        # Fall back to standard strategy for now
        return StandardStrategy().train(**kwargs)
