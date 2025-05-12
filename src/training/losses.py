import torch
import torch.nn as nn
import torch.autograd as autograd

class Losses(nn.Module):
    """
    A class to compute loss functions for training PINNs.
    """
    def __init__(self, model: torch.nn.Module, device='cpu', reduction='mean') -> None:
        """
        Initialize the loss functions.

        Args:
            model: The neural network model
            device: Torch device (cpu or cuda)
            reduction (str): Specifies the reduction to apply to the output ('mean' or 'sum')
        """
        super(Losses, self).__init__()
        self.model = model
        self.device = device
        self.loss_function = nn.MSELoss(reduction=reduction)

    def boundary_loss(self, X_train_Nu, U_train_Nu) -> torch.Tensor:
        """
        Calculate loss at boundary points.

        Args:
            X_train_Nu (torch.Tensor): Boundary coordinates
            U_train_Nu (torch.Tensor): True boundary values

        Returns:
            torch.Tensor: Boundary loss
        """
        predicted_u = self.model.forward(X_train_Nu)
        return self.loss_function(predicted_u, U_train_Nu)

    def compute_p_laplacian(self, u, x, p):
        """
        Compute the p-Laplacian operator: div(|∇u|^(p-2)∇u)

        Args:
            u (torch.Tensor): Model output
            x (torch.Tensor): Input coordinates
            p (float): p-value for the p-Laplacian

        Returns:
            torch.Tensor: The p-Laplacian of u
        """
        # Compute first derivatives (gradients)
        u_grad = autograd.grad(
            u, x, 
            grad_outputs=torch.ones_like(u).to(self.device),
            retain_graph=True, create_graph=True
        )[0]

        u_x = u_grad[:, [0]]
        u_y = u_grad[:, [1]]

        # Compute |∇u|^(p-2)
        grad_norm_pow = torch.sqrt(u_x**2 + u_y**2)**(p-2)

        # Compute |∇u|^(p-2)∇u components
        weighted_u_x = grad_norm_pow * u_x
        weighted_u_y = grad_norm_pow * u_y

        # Compute divergence of (|∇u|^(p-2)∇u)
        div_x = autograd.grad(
            weighted_u_x, x,
            grad_outputs=torch.ones_like(u_x).to(self.device),
            retain_graph=True, create_graph=True
        )[0][:, [0]]

        div_y = autograd.grad(
            weighted_u_y, x,
            grad_outputs=torch.ones_like(u_y).to(self.device),
            retain_graph=True, create_graph=True
        )[0][:, [1]]

        # Return the full p-Laplacian
        return div_x + div_y

    def pde_loss(self, X_train_Nf, p):
        """
        Calculate the PDE residual loss for the p-Laplacian equation.

        Args:
            X_train_Nf (torch.Tensor): Collocation points
            p (float): p-value for the p-Laplacian

        Returns:
            torch.Tensor: PDE residual loss
        """
        # Create a copy with gradient tracking
        x = X_train_Nf.clone()
        x.requires_grad = True

        # Forward pass
        u = self.model(x)

        # Compute p-Laplacian
        p_laplacian = self.compute_p_laplacian(u, x, p)

        # PDE: -div(|∇u|^(p-2)∇u) = 1
        residual = -p_laplacian - 1.0

        # Zero residual is the target
        return torch.mean(residual**2)

    def total_loss(self, X_train_Nu, U_train_Nu, X_train_Nf, p, bc_weight=10.0):
        """
        Calculate the total loss as a weighted sum of boundary and PDE losses.

        Args:
            X_train_Nu (torch.Tensor): Boundary points
            U_train_Nu (torch.Tensor): Boundary values
            X_train_Nf (torch.Tensor): Collocation points
            p (float): p-value for the p-Laplacian
            bc_weight (float): Weight for the boundary condition loss

        Returns:
            torch.Tensor: Total loss
        """
        loss_u = self.boundary_loss(X_train_Nu, U_train_Nu)
        loss_f = self.pde_loss(X_train_Nf, p)

        return bc_weight * loss_u + loss_f