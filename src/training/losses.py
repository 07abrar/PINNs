import torch
import torch.nn as nn
import torch.autograd as autograd

class Losses(nn.Module):
    """
    A class to compute loss functions for training PINNs.
    """
    def __init__(self,
        pde_residual: callable,
        model: torch.nn.Module,
        device='cpu',
        reduction='mean') -> None:
        """
        Initialize the loss functions.

        Args:
            pde_residual (callable): The PDE residual function
            model: The neural network model
            device: Torch device (cpu or cuda)
            reduction (str): Specifies the reduction to apply to the output ('mean' or 'sum')
        """
        super(Losses, self).__init__()
        self.pde_residual = pde_residual
        self.model = model
        self.device = device
        self.loss_function = nn.MSELoss(reduction=reduction)

    def boundary_loss(self, x_train_Nu, u_train_Nu) -> torch.Tensor:
        """
        Calculate loss at boundary points.

        Args:
            X_train_Nu (torch.Tensor): Boundary coordinates
            U_train_Nu (torch.Tensor): True boundary values

        Returns:
            torch.Tensor: Boundary loss
        """
        predicted_u = self.model.forward(x_train_Nu)
        return self.loss_function(predicted_u, u_train_Nu)

    def pde_loss(self, x_train_Nf):
        """
        Calculate the PDE residual loss.
        This function computes the PDE residual loss from self.pde_residual with
        respect to the collocation points.
        The PDE residual is defined as the difference between the predicted
        solution and the true solution.

        Args:
            X_train_Nf (torch.Tensor): Collocation points

        Returns:
            torch.Tensor: PDE residual loss
        """
        u_pred = self.model.forward(x_train_Nf)
        residual_function = self.pde_residual(x_train_Nf, u_pred)
        true_solution = torch.zeros_like(u_pred, dtype=torch.float).to(self.device)
        return self.loss_function(residual_function, true_solution)





    def total_loss(self, x_train_Nu, u_train_Nu, x_train_Nf, bc_weight=10.0):
        """
        Calculate the total loss as a weighted sum of boundary and PDE losses.

        Args:
            X_train_Nu (torch.Tensor): Boundary points
            U_train_Nu (torch.Tensor): Boundary values
            X_train_Nf (torch.Tensor): Collocation points
            bc_weight (float): Weight for the boundary condition loss

        Returns:
            torch.Tensor: Total loss
        """
        loss_u = self.boundary_loss(x_train_Nu, u_train_Nu)
        loss_f = self.pde_loss(x_train_Nf)

        return bc_weight * loss_u + loss_f