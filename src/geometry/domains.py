"""Geometric domain definitions used for training PINNs."""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import numpy as np
import shapely.geometry as sg
from scipy.stats import qmc

from src.utils.visualization import training_data_plot


class Domain(ABC):
    """Base class for all computational domains"""

    @property
    @abstractmethod
    def boundary_points(self) -> np.ndarray:
        """Get points on the domain boundary"""

    @property
    @abstractmethod
    def collocation_points(self) -> np.ndarray:
        """Get points inside the domain"""

    @abstractmethod
    def visualization_mask(
        self, x_grid: np.ndarray, y_grid: np.ndarray
    ) -> np.ndarray:
        """Create mask for visualization (True = inside domain)"""

    @abstractmethod
    def training_data_plot(self, save_path: str | None = None) -> None:
        """Plot training data points"""


class CircularDomain(Domain):
    """Circular domain using Shapely for geometry operations"""

    def __init__(
        self,
        center: Tuple[float, float] = (0, 0),
        radius: float = 1.0,
        training_data: Dict[str, int] | None = None,
    ) -> None:
        if training_data is None:
            training_data = {"boundary": 100, "collocation": 1000}
        self.training_data = training_data
        self.center = center
        self.radius = radius
        self._geometry = sg.Point(self.center).buffer(self.radius)

    @property
    def boundary_points(self) -> np.ndarray:
        """Generate uniformly distributed points on the circle boundary"""
        n_points = self.training_data["boundary"]
        engine = qmc.LatinHypercube(d=1)
        theta = engine.random(n=n_points)[:, 0] * 2 * np.pi

        points = np.zeros((n_points, 2))
        points[:, 0] = self.center[0] + self.radius * np.cos(theta)
        points[:, 1] = self.center[1] + self.radius * np.sin(theta)

        return points

    @property
    def collocation_points(self) -> np.ndarray:
        """Generate uniformly distributed points inside the circle"""
        n_points = self.training_data["collocation"]
        engine = qmc.LatinHypercube(d=2)
        samples = engine.random(n=n_points)

        # Convert to polar coordinates for uniform distribution
        r = np.sqrt(samples[:, 0]) * self.radius
        theta = samples[:, 1] * 2 * np.pi

        points = np.zeros((n_points, 2))
        points[:, 0] = self.center[0] + r * np.cos(theta)
        points[:, 1] = self.center[1] + r * np.sin(theta)

        return points

    def visualization_mask(
        self, x_grid: np.ndarray, y_grid: np.ndarray
    ) -> np.ndarray:
        """Create a mask for points inside the circle"""
        dx = x_grid - self.center[0]
        dy = y_grid - self.center[1]
        return (dx**2 + dy**2) <= self.radius**2

    def training_data_plot(self, save_path: str | None = None) -> None:
        """Plot training data points"""
        training_data_plot(
            self.boundary_points,
            self.collocation_points,
            self.visualization_mask,
            save_path,
        )


class PolygonDomain(Domain):
    """Polygon domain using Shapely for geometry operations"""

    def __init__(
        self,
        vertices: List[Tuple[float, float]],
        training_data: Dict[str, int] | None = None,
    ) -> None:
        """
        Initialize polygon from vertices
        vertices: [(x1,y1), (x2,y2), ..., (xn,yn)]
        """
        if training_data is None:
            training_data = {"boundary": 100, "collocation": 1000}
        if len(vertices) < 3:
            raise ValueError("Polygon requires at least 3 vertices")

        self.vertices = vertices
        try:
            self._geometry = sg.Polygon(vertices)
            if not self._geometry.is_valid:
                raise ValueError("Invalid polygon geometry")
        except Exception as e:
            raise ValueError(f"Failed to create polygon: {e}") from e

        # Get bounding box for sampling
        minx, miny, maxx, maxy = self._geometry.bounds
        self.bbox = (minx, miny, maxx, maxy)
        self.training_data = training_data

    @property
    def boundary_points(self) -> np.ndarray:
        """Generate points on polygon boundary with proportional distribution"""
        n_points = self.training_data["boundary"]
        boundary = self._geometry.exterior
        boundary_coords = np.array(boundary.coords[:])
        if len(boundary_coords) >= n_points:
            indices = np.linspace(
                0, len(boundary_coords) - 1, n_points, dtype=int
            )
            return boundary_coords[indices]

        # Calculate edge lengths for proportional distribution
        edges = np.diff(boundary_coords, axis=0)
        edge_lengths = np.sqrt((edges**2).sum(axis=1))
        total_length = edge_lengths.sum()
        points_per_edge = (edge_lengths / total_length * n_points).astype(int)
        # Ensure at least one point per edge
        points_per_edge = np.maximum(points_per_edge, 1)

        points = []
        for i, n in enumerate(points_per_edge):
            t = np.linspace(0, 1, n, endpoint=False)
            start, end = (
                boundary_coords[i],
                boundary_coords[(i + 1) % len(boundary_coords)],
            )
            points.append(start + t[:, np.newaxis] * (end - start))
        return np.vstack(points)[:n_points]

    @property
    def collocation_points(self) -> np.ndarray:
        """Generate points inside polygon using efficient rejection sampling"""
        n_points = self.training_data["collocation"]
        minx, miny, maxx, maxy = self.bbox

        points = []
        max_attempts = n_points * 10  # Prevent infinite loop
        attempts = 0

        while len(points) < n_points and attempts < max_attempts:
            # Generate random points in bounding box
            candidates = np.random.uniform(
                low=[minx, miny],
                high=[maxx, maxy],
                size=(n_points - len(points), 2),
            )

            # Check which points are inside polygon
            for point in candidates:
                if self._geometry.contains(sg.Point(point)):
                    points.append(point)
                    if len(points) >= n_points:
                        break

            attempts += 1

        if len(points) < n_points:
            print(
                f"Warning: Only generated {len(points)} interior points out of {n_points} requested"
            )

        return np.array(points)

    def visualization_mask(
        self, x_grid: np.ndarray, y_grid: np.ndarray
    ) -> np.ndarray:
        """Create a mask for points inside the polygon"""
        mask = np.zeros_like(x_grid, dtype=bool)
        flat_points = np.column_stack((x_grid.ravel(), y_grid.ravel()))

        # Check each point
        for i, point in enumerate(flat_points):
            mask.flat[i] = self._geometry.contains(sg.Point(point))

        return mask

    def training_data_plot(self, save_path: str | None = None) -> None:
        """Plot training data points"""
        training_data_plot(
            self.boundary_points,
            self.collocation_points,
            self.visualization_mask,
            save_path,
        )


class RectangularDomain(Domain):
    """Rectangular domain - specialized case"""

    def __init__(
        self,
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        training_data: Dict[str, int] | None = None,
    ) -> None:
        if training_data is None:
            training_data = {"boundary": 100, "collocation": 1000}
        self.x_range = x_range
        self.y_range = y_range

        # Create rectangle vertices
        vertices = [
            (x_range[0], y_range[0]),  # bottom-left
            (x_range[1], y_range[0]),  # bottom-right
            (x_range[1], y_range[1]),  # top-right
            (x_range[0], y_range[1]),  # top-left
        ]
        self._geometry = sg.Polygon(vertices)
        self.training_data = training_data

    @property
    def boundary_points(self) -> np.ndarray:
        """Generate points on rectangle boundary"""
        n_points = self.training_data["boundary"]
        # Distribute points along 4 edges
        points_per_edge = n_points // 4
        remainder = n_points % 4

        points = []

        # Bottom edge
        x = np.linspace(
            self.x_range[0],
            self.x_range[1],
            points_per_edge + (1 if remainder > 0 else 0),
        )
        y = np.full_like(x, self.y_range[0])
        points.extend(zip(x, y))
        remainder -= 1

        # Right edge
        y = np.linspace(
            self.y_range[0],
            self.y_range[1],
            points_per_edge + (1 if remainder > 0 else 0),
        )[
            1:
        ]  # Skip corner
        x = np.full_like(y, self.x_range[1])
        points.extend(zip(x, y))
        remainder -= 1

        # Top edge
        x = np.linspace(
            self.x_range[1],
            self.x_range[0],
            points_per_edge + (1 if remainder > 0 else 0),
        )[
            1:
        ]  # Skip corner
        y = np.full_like(x, self.y_range[1])
        points.extend(zip(x, y))
        remainder -= 1

        # Left edge
        y = np.linspace(
            self.y_range[1],
            self.y_range[0],
            points_per_edge + (1 if remainder > 0 else 0),
        )[
            1:-1
        ]  # Skip corners
        x = np.full_like(y, self.x_range[0])
        points.extend(zip(x, y))

        return np.array(points)

    @property
    def collocation_points(self) -> np.ndarray:
        """Generate points inside rectangle"""
        n_points = self.training_data["collocation"]
        engine = qmc.LatinHypercube(d=2)
        samples = engine.random(n=n_points)

        # Scale to rectangle bounds
        points = np.zeros((n_points, 2))
        points[:, 0] = self.x_range[0] + samples[:, 0] * (
            self.x_range[1] - self.x_range[0]
        )
        points[:, 1] = self.y_range[0] + samples[:, 1] * (
            self.y_range[1] - self.y_range[0]
        )

        return points

    def visualization_mask(
        self, x_grid: np.ndarray, y_grid: np.ndarray
    ) -> np.ndarray:
        """Create rectangular mask for plotting"""
        return (
            (x_grid >= self.x_range[0])
            & (x_grid <= self.x_range[1])
            & (y_grid >= self.y_range[0])
            & (y_grid <= self.y_range[1])
        )

    def training_data_plot(self, save_path: str | None = None) -> None:
        """Plot training data points"""
        training_data_plot(
            self.boundary_points,
            self.collocation_points,
            self.visualization_mask,
            save_path,
        )
