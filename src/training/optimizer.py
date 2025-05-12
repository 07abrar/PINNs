import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR

class Optimizer:
    """
    Class for configuring and managing optimizers for training PINNs.
    """
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer_type="adam",
        lr=0.001,
        weight_decay=1e-6,
        scheduler_type=None,
        **kwargs
    ) -> None:
        """
        Initialize the optimizer.

        Args:
            model: The neural network model
            optimizer_type (str): Type of optimizer ('adam', 'lbfgs')
            lr (float): Learning rate
            weight_decay (float): Weight decay for regularization
            scheduler_type (str, optional): Type of learning rate scheduler
            **kwargs: Additional arguments for the optimizer
        """
        self.model = model
        self.optimizer_type = optimizer_type.lower()
        self.lr = lr

        # Configure optimizer
        if self.optimizer_type == "adam":
            self.optimizer = optim.Adam(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                decoupled_weight_decay=True,
                amsgrad=True,
                **kwargs
            )
        elif self.optimizer_type == "lbfgs":
            self.optimizer = optim.LBFGS(
                model.parameters(),
                lr=lr,
                **kwargs
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

        # Configure scheduler
        self.scheduler = None
        if scheduler_type:
            self._init_scheduler(scheduler_type, **kwargs)

    def _init_scheduler(self, scheduler_type, **kwargs):
        """
        Initialize learning rate scheduler.

        Args:
            scheduler_type (str): Type of scheduler
            **kwargs: Additional arguments for the scheduler
        """
        if scheduler_type == "plateau":
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=kwargs.get('factor', 0.5),
                patience=kwargs.get('patience', 10),
                verbose=kwargs.get('verbose', True)
            )
        elif scheduler_type == "step":
            self.scheduler = StepLR(
                self.optimizer,
                step_size=kwargs.get('step_size', 100),
                gamma=kwargs.get('gamma', 0.5)
            )
        elif scheduler_type == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=kwargs.get('T_max', 100),
                eta_min=kwargs.get('eta_min', 0)
            )

    def step(self, closure=None):
        """
        Perform a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss. Required for LBFGS.
        """
        if self.optimizer_type == "lbfgs" and closure is None:
            raise ValueError("LBFGS optimizer requires a closure function")

        if self.optimizer_type == "lbfgs":
            return self.optimizer.step(closure)
        else:
            self.optimizer.step()

    def zero_grad(self):
        """
        Clear the gradients of all optimized parameters.
        """
        self.optimizer.zero_grad()

    def scheduler_step(self, metric=None):
        """
        Step the learning rate scheduler.

        Args:
            metric (float, optional): Metric value to use for ReduceLROnPlateau
        """
        if self.scheduler is None:
            return

        if isinstance(self.scheduler, ReduceLROnPlateau):
            if metric is None:
                raise ValueError("ReduceLROnPlateau scheduler requires a metric value")
            self.scheduler.step(metric)
        else:
            self.scheduler.step()

    def get_lr(self):
        """
        Get current learning rate.

        Returns:
            float: Current learning rate
        """
        for param_group in self.optimizer.param_groups:
            return param_group['lr']