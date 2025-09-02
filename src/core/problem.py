from typing import Any, Callable, Dict
import torch


class PDEProblem:
    """
    Encapsulates a PDE problem definition
    """

    def __init__(
        self,
        residual_fn: Callable,
        boundary_conditions: Dict[str, Any] = None,
    ):
        self.residual_fn = residual_fn
        self.boundary_conditions = boundary_conditions or {"dirichlet": 0.0}

    def compute_residual(
        self, coords: torch.Tensor, u_pred: torch.Tensor
    ) -> torch.Tensor:
        """Compute PDE residual at given coordinates"""
        return self.residual_fn(coords, u_pred)

    def get_boundary_value(self, boundary_type: str = "dirichlet") -> float:
        """Get boundary condition value"""
        return self.boundary_conditions.get(boundary_type, 0.0)
