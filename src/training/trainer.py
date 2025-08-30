import time
from typing import Dict, List, Any, Optional
import torch

from .strategies import StandardStrategy
from ..config.experiment import ExperimentConfig


class Trainer:
    """
    Main training orchestrator

    Responsibilities:
    - Coordinate training process
    - Handle data generation via Domain
    - Execute training strategies
    - Manage logging and saving
    """

    def __init__(
        self,
        pinn,           # PINN instance
        domain,         # Domain instance
        optimizer_config: Dict[str, Any],
        strategy="standard",
        experiment_config: Optional[ExperimentConfig] = None
    ):
        self.pinn = pinn
        self.domain = domain
        self.experiment_config = experiment_config or ExperimentConfig()

        # Create optimizer
        self._create_optimizer(optimizer_config)

        # Set training strategy
        if isinstance(strategy, str):
            if strategy == "standard":
                self.strategy = StandardStrategy()
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
        else:
            self.strategy = strategy

    def _create_optimizer(self, config: Dict[str, Any]):
        """Create optimizer from config"""
        optimizer_type = config.get("type", "adam").lower()
        lr = config.get("lr", 1e-3)

        if optimizer_type == "adam":
            self.optimizer = torch.optim.Adam(
                self.pinn.parameters(),
                lr=lr,
                weight_decay=config.get("weight_decay", 1e-6)
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")

    def train(
        self,
        epochs: int = 1000,
        loss_threshold: float = 1e-4,
        bc_weight: float = 10.0,
        **kwargs
    ) -> Dict[str, Any]:
        """Execute training process"""

        # Generate training data
        boundary_points = self.domain.boundary_points
        collocation_points = self.domain.collocation_points

        # Convert to tensors
        boundary_points = torch.tensor(
            boundary_points, dtype=self.pinn.dtype, device=self.pinn.device)
        collocation_points = torch.tensor(
            collocation_points, dtype=self.pinn.dtype, device=self.pinn.device)

        # Get boundary values from problem
        boundary_values = torch.full(
            (len(boundary_points), 1),
            self.pinn.problem.get_boundary_value(),
            dtype=self.pinn.dtype,
            device=self.pinn.device
        )

        # Execute training strategy
        results = self.strategy.train(
            pinn=self.pinn,
            optimizer=self.optimizer,
            boundary_points=boundary_points,
            boundary_values=boundary_values,
            collocation_points=collocation_points,
            epochs=epochs,
            loss_threshold=loss_threshold,
            bc_weight=bc_weight,
            **kwargs
        )

        return results
