"""PDE problem definitions."""

from typing import Any, Callable, Dict

import torch


class PDEProblem:
    """Container for PDE residual and boundary conditions.

    Parameters:
        residual_fn: Callable computing the PDE residual given coordinates and
            network predictions.
        boundary_conditions: Mapping of boundary condition types to values.
    """

    def __init__(
        self,
        residual_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        boundary_conditions: Dict[str, float] | None = None,
    ) -> None:
        self.residual_fn = residual_fn
        self.boundary_conditions = boundary_conditions or {"dirichlet": 0.0}

    def compute_residual(
        self, coords: torch.Tensor, u_pred: torch.Tensor
    ) -> torch.Tensor:
        """Compute PDE residual at ``coords`` using ``u_pred``."""
        return self.residual_fn(coords, u_pred)

    def get_boundary_value(self, boundary_type: str = "dirichlet") -> float:
        """Return boundary condition value for ``boundary_type``."""
        return self.boundary_conditions.get(boundary_type, 0.0)
