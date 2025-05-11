import torch
import torch.nn as nn
from .networks import NeuralNet
from src.training.losses import Losses

class PINN(nn.Module):
    """
    Physics-Informed Neural Network (PINN) class.
    This class encapsulates the neural network from NeuralNet and the loss functions from Losses.
    """
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int, 
        output_dim: int, 
        num_hidden_layers: int, 
        activation: str = "tanh",
        device: str = "cpu"
    ) -> None:
        super(PINN, self).__init__()
        self.network = NeuralNet(
            input_dim, hidden_dim, output_dim, num_hidden_layers, activation
        )
        self.device = device
        # Initialize the loss calculator
        self.loss_calculator = Losses(model=self.network, device=device)

    def forward(self, x):
        """
        Forward pass through the PINN.
        """
        return self.network(x)

    def compute_loss(self, X_train_Nu, U_train_Nu, X_train_Nf, p=2.0, bc_weight=10.0):
        """
        Compute the total loss for the PINN using the Losses class.

        Args:
            X_train_Nu (torch.Tensor): Boundary points
            U_train_Nu (torch.Tensor): Boundary values
            X_train_Nf (torch.Tensor): Collocation points
            p (float): p-value for the p-Laplacian
            bc_weight (float): Weight for the boundary condition loss

        Returns:
            torch.Tensor: Total loss
        """
        return self.loss_calculator.total_loss(
            X_train_Nu=X_train_Nu,
            U_train_Nu=U_train_Nu,
            X_train_Nf=X_train_Nf,
            p=p,
            bc_weight=bc_weight
        )
    
    def get_individual_losses(self, X_train_Nu, U_train_Nu, X_train_Nf, p=2.0):
        """
        Get individual loss components for monitoring.

        Args:
            X_train_Nu (torch.Tensor): Boundary points
            U_train_Nu (torch.Tensor): Boundary values
            X_train_Nf (torch.Tensor): Collocation points
            p (float): p-value for the p-Laplacian

        Returns:
            dict: Dictionary containing individual loss components
        """
        boundary_loss = self.loss_calculator.boundary_loss(X_train_Nu, U_train_Nu)
        pde_loss = self.loss_calculator.pde_loss(X_train_Nf, p)
        
        return {
            'boundary_loss': boundary_loss.item(),
            'pde_loss': pde_loss.item()
        }