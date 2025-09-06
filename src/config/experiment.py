from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import torch


def _current_timestamp() -> str:
    """Return the current timestamp in ISO format."""
    return datetime.now().isoformat()


@dataclass
class ExperimentConfig:
    """Simple experiment configuration container.

    Attributes:
        name: Human readable name of the experiment.
        seed: Random seed used for reproducibility.
        timestamp: Creation time of the configuration.
    """

    name: str = "default"
    seed: int = 42
    timestamp: str = field(default_factory=_current_timestamp)

    def apply(self) -> None:
        """Apply configuration settings such as random seeding."""
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
