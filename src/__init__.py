"""
PINN Library - Physics-Informed Neural Networks

A modular library for solving PDEs using physics-informed neural networks.
"""

# Core components
from .core.pinn import PINN
from .core.problem import PDEProblem

# Geometry
from .geometry.domains import CircularDomain, PolygonDomain, RectangularDomain

# Training
from .training.trainer import Trainer
from .training.strategies import StandardStrategy, AdaptiveSamplingStrategy

# Configuration
from .config.experiment import ExperimentConfig

# Models (existing - keep as is for now)
from .models.networks import NeuralNet

# Utils (existing - keep as is for now)
from .utils.visualization import TrainingDataVisualizer

__version__ = "0.2.0"

__all__ = [
    # Core
    'PINN', 'PDEProblem',
    # Geometry
    'CircularDomain', 'PolygonDomain', 'RectangularDomain',
    # Training
    'Trainer', 'StandardStrategy', 'AdaptiveSamplingStrategy',
    # Config
    'ExperimentConfig',
    # Models
    'NeuralNet',
    # Utils
    'TrainingDataVisualizer'
]
