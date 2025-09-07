"""Training utilities."""

from typing import Any, Dict, Optional, Union

import torch

from src.config.experiment import ExperimentConfig
from src.core.neural_net import NeuralNet
from src.core.problem import PDEProblem
from src.geometry.domains import Domain
from src.training.strategies import StandardStrategy, TrainingStrategy


class Trainer:
    """High level training orchestrator for PINN models."""

    def __init__(
        self,
        model: NeuralNet,
        problem: PDEProblem,
        domain: Domain,
        optimizer_config: Dict[str, Any],
        strategy: Union[str, TrainingStrategy] = "standard",
        experiment_config: Optional[ExperimentConfig] = None,
        checkpoint_path: Optional[str] = None,
        checkpoint_interval: int = 1,
    ) -> None:
        self.model = model
        self.problem = problem
        self.domain = domain
        self.experiment_config = experiment_config or ExperimentConfig()
        self.checkpoint_path = checkpoint_path
        self.checkpoint_interval = checkpoint_interval
        self._best_loss = float("inf")

        self._create_optimizer(optimizer_config)

        if isinstance(strategy, str):
            if strategy == "standard":
                self.strategy = StandardStrategy()
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
        else:
            self.strategy = strategy

    def _create_optimizer(self, config: Dict[str, Any]) -> None:
        """Create optimizer from config"""
        optimizer_type = config.get("type", "adam").lower()
        lr = config.get("lr", 1e-3)

        if optimizer_type == "adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=config.get("weight_decay", 1e-6),
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")

    def train(
        self,
        epochs: int = 1000,
        loss_threshold: float = 1e-4,
        bc_weight: float = 10.0,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Execute training process"""

        # Generate training data
        boundary_points = self.domain.boundary_points
        collocation_points = self.domain.collocation_points

        # Convert to tensors
        boundary_points = torch.tensor(
            boundary_points,
            dtype=self.model.dtype,
            device=self.model.device,
        )
        collocation_points = torch.tensor(
            collocation_points,
            dtype=self.model.dtype,
            device=self.model.device,
        )

        # Get boundary values from problem
        boundary_values = torch.full(
            (len(boundary_points), 1),
            self.problem.get_boundary_value(),
            dtype=self.model.dtype,
            device=self.model.device,
        )

        def epoch_callback(epoch: int, loss_val: float) -> None:
            if self.checkpoint_path is None:
                return
            if epoch % self.checkpoint_interval != 0:
                return
            if loss_val < self._best_loss:
                torch.save(self.model.state_dict(), self.checkpoint_path)
                self._best_loss = loss_val

        # Execute training strategy
        return self.strategy.train(
            model=self.model,
            problem=self.problem,
            optimizer=self.optimizer,
            boundary_points=boundary_points,
            boundary_values=boundary_values,
            collocation_points=collocation_points,
            epochs=epochs,
            loss_threshold=loss_threshold,
            bc_weight=bc_weight,
            epoch_callback=epoch_callback,
            **kwargs,
        )
