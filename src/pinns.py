import time
from typing import List, Tuple, Optional

import torch
import torch.nn as nn

from src import NeuralNet, Losses, TrainingDataGenerator, Optimizer


class PINN(nn.Module):
    """
    Physics-Informed Neural Network (PINN) class.
    This class encapsulates the neural network from NeuralNet
    and the loss functions from Losses.
    """

    def __init__(
        self,
        # args for the PDE
        pde_residual: callable,
        # Args for the neural network
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_hidden_layers: int,
        activation: str = "tanh",
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        # Args for training data distribution
        Nd: int = 50,
        Nc: int = 1000,
        # Args for optimizer
        optimizer_type: str = "adam",
        lr: float = 1e-3,
        weight_decay: float = 1e-6,
        scheduler_type: str = None,
        **opt_kwargs
    ) -> None:
        """
        Initialize the PINN.
        Args:
            pde_residual (callable): The PDE residual function
            input_dim (int): Input dimension of the neural network
            hidden_dim (int): Hidden dimension of the neural network
            output_dim (int): Output dimension of the neural network
            num_hidden_layers (int): Number of hidden layers in the neural network
            activation (str): Activation function for the neural network
            device (str): Device to run the model on ('cpu' or 'cuda')
            Nd (int): Number of boundary points
            Nc (int): Number of collocation points
            optimizer_type (str): Type of optimizer ('adam', 'lbfgs')
            lr (float): Learning rate for the optimizer
            weight_decay (float): Weight decay for regularization
            scheduler_type (str, optional): Type of learning rate scheduler
        """
        super(PINN, self).__init__()
        self.pde_residual = pde_residual
        self.device = device
        self.dtype = dtype

        # Initialize the neural network
        self.model = NeuralNet(
            input_dim, hidden_dim, output_dim, num_hidden_layers, activation
        ).to(device=device, dtype=dtype)

        # Initialize the loss calculator
        self.loss_calculator = Losses(
            pde_residual=self.pde_residual,
            model=self.model,
            dtype=dtype,
            device=device
        )

        # Initialize training data
        self.data_generator = TrainingDataGenerator(Nd=Nd, Nc=Nc)
        Xb, Ub, Xf = self.data_generator.generate_training_data()
        self.x_train_Nu = torch.tensor(Xb, dtype=self.dtype, device=device)
        self.u_train_Nu = torch.tensor(Ub, dtype=self.dtype, device=device)
        self.x_train_Nf = torch.tensor(Xf, dtype=self.dtype, device=device)

        # Initialize optimizer
        self.optimizer = Optimizer(
            model=self.model,
            optimizer_type=optimizer_type,
            lr=lr,
            weight_decay=weight_decay,
            scheduler_type=scheduler_type,
            **opt_kwargs
        )

    def __call__(
        self,
        bc_weight: float = 10.0
    ) -> torch.Tensor:
        """
        Perform one training step and evaluation:
        - switch to train mode
        - zero gradients
        - compute total loss
        - backpropagate
        - optimizer step
        - switch to eval mode
        Returns:
            loss (torch.Tensor): training loss tensor
            test_loss (torch.Tensor): boundary loss tensor
        """
        self.train()
        self.zero_grad()
        loss = self.compute_loss(
            self.x_train_Nu,
            self.u_train_Nu,
            self.x_train_Nf,
            bc_weight
        )
        self.loss_backward(loss)
        self.step()
        self.eval()
        return loss

    def train(self, mode: bool = True):
        return super().train(mode)

    def execute_training_loop(self, loss_threshold: float, **kwargs) -> Tuple[List[float], float]:
        """
        Executes the training loop until the loss falls below loss_threshold.
        """
        self.train(mode=True)  # Set the model to training mode

        epochs = 0
        loss_values1 = 1.0
        loss_values2 = 1.0  # Ensure loop starts
        recorded_loss_values = []
        start_time = time.time()

        while loss_values2 > loss_threshold:
            current_loss = self(**kwargs)  # Calls PINN.__call__
            recorded_loss_values.append(current_loss.item())

            if epochs == 0:
                loss_values1 = 1.0
                loss_values2 = recorded_loss_values[0]
                # "Test Loss" here is overall current step loss
                print("Training Loss ----- Test Loss")
            else:
                loss_values1 = recorded_loss_values[epochs-1]
                loss_values2 = recorded_loss_values[epochs]

            if epochs % 100 == 0:
                print(f"{epochs} - {current_loss}")

            epochs += 1

        elapsed = time.time() - start_time
        print(f"Training time: {elapsed:.2f} s")
        return recorded_loss_values, elapsed

    def zero_grad(self):
        """Clear gradients via the wrapped optimizer."""
        self.optimizer.zero_grad()

    def compute_loss(
        self,
        x_train_Nu: torch.Tensor,
        u_train_Nu: torch.Tensor,
        x_train_Nf: torch.Tensor,
        bc_weight: float = 10.0
    ) -> torch.Tensor:
        """
        Compute the total loss for the PINN using the Losses class and perform backpropagation.

        Args:
            X_train_Nu (torch.Tensor): Boundary points.
            U_train_Nu (torch.Tensor): Boundary values.
            X_train_Nf (torch.Tensor): Collocation points.
            bc_weight (float, optional): Weight for the boundary condition loss. Default is 10.0.
        """
        return self.loss_calculator.total_loss(
            x_train_Nu=x_train_Nu,
            u_train_Nu=u_train_Nu,
            x_train_Nf=x_train_Nf,
            bc_weight=bc_weight,
        )

    def loss_backward(self, loss: torch.Tensor):
        """
        Backpropagate the loss.
        Args:
            loss (torch.Tensor): The loss tensor to backpropagate.
        """
        loss.backward()

    def step(self, closure=None):
        """Perform an optimizer step (supports LBFGS closure)."""
        self.optimizer.step(closure)

    def eval(self):
        return super().eval()

    def get_individual_losses(self, X_train_Nu, U_train_Nu, X_train_Nf, p=2.0):
        """
        Get individual loss components for monitoring.

        Args:
            X_train_Nu (torch.Tensor): Boundary points
            U_train_Nu (torch.Tensor): Boundary values
            X_train_Nf (torch.Tensor): Collocation points

        Returns:
            dict: Dictionary containing individual loss components
        """
        boundary_loss = self.loss_calculator.boundary_loss(
            X_train_Nu, U_train_Nu)
        pde_loss = self.loss_calculator.pde_loss(X_train_Nf)

        return {
            'boundary_loss': boundary_loss.item(),
            'pde_loss': pde_loss.item()
        }

    def forward(self, x):
        """
        Forward pass through the PINN.
        """
        return self.model(x)
