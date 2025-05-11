import torch
import torch.nn as nn
from src import NeuralNet, Losses
from src.physics.boundary_and_collocation_points import TrainingDataGenerator
from src.training.optimizer import Optimizer

class PINN(nn.Module):
    """
    Physics-Informed Neural Network (PINN) class.
    This class encapsulates the neural network from NeuralNet and the loss functions from Losses.
    """
    def __init__(
        self, 
        # Args for the neural network
        input_dim: int, 
        hidden_dim: int, 
        output_dim: int, 
        num_hidden_layers: int, 
        activation: str = "tanh",
        device: str = "cpu",
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
        super(PINN, self).__init__()
        self.device = device
        self.model = NeuralNet(
            input_dim, hidden_dim, output_dim, num_hidden_layers, activation
        ).to(device)

        # Initialize the loss calculator
        self.loss_calculator = Losses(model=self.model, device=device)

        # Initialize training data
        self.data_generator = TrainingDataGenerator(Nd=Nd, Nc=Nc)
        Xb, Ub, Xf = self.data_generator.generate_training_data()
        self.X_train_Nu = torch.tensor(Xb, dtype=torch.float32, device=device)
        self.U_train_Nu = torch.tensor(Ub, dtype=torch.float32, device=device)
        self.X_train_Nf = torch.tensor(Xf, dtype=torch.float32, device=device)

        # Initialize optimizer
        self.optimizer = Optimizer(
            model=self.model,
            optimizer_type=optimizer_type,
            lr=lr,
            weight_decay=weight_decay,
            scheduler_type=scheduler_type,
            **opt_kwargs
        )

    def train(self, mode = True):
        return super().train(mode)
    
    def zero_grad(self):
        """Clear gradients via the wrapped optimizer."""
        self.optimizer.zero_grad()

    def compute_loss(self, X_train_Nu, U_train_Nu, X_train_Nf, p=2.0, bc_weight=10.0):
        """
        Compute the total loss for the PINN using the Losses class and perform backpropagation.

        Args:
            X_train_Nu (torch.Tensor): Boundary points.
            U_train_Nu (torch.Tensor): Boundary values.
            X_train_Nf (torch.Tensor): Collocation points.
            p (float, optional): p-value for the p-Laplacian. Default is 2.0.
            bc_weight (float, optional): Weight for the boundary condition loss. Default is 10.0.
        """

        loss = self.loss_calculator.total_loss(
            X_train_Nu=X_train_Nu,
            U_train_Nu=U_train_Nu,
            X_train_Nf=X_train_Nf,
            p=p,
            bc_weight=bc_weight,
        )
        
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

    
    def forward(self, x):
        """
        Forward pass through the PINN.
        """
        return self.model(x)