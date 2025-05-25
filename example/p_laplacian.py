import os
import sys
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.autograd as autograd

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)
from src.pinns import PINN
from src.utils.visualization import TrainingDataVisualizer as TDV

# reproducibility
torch.manual_seed(31)

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Example 1: 2D p-poisson equation
def pde_residual(coords, u_pred):
    """
    Compute the residual of the 2D p-Laplacian Poisson equation

        −∇·( |∇u|^(p-2) ∇u ) = 1

    Args:
        coords (torch.Tensor): shape (N,2) input points (x,y), requires_grad=True
        u_pred (torch.Tensor): shape (N,1) model's predicted u at coords

    Returns:
        torch.Tensor: PDE residual r(x,y) = −∇·( |∇u|^(p-2) ∇u ) - 1
    """
    p = 2.0  # p-Laplacian exponent
    # 1) Compute gradient ∇u = (∂u/∂x, ∂u/∂y)
    u_grad = autograd.grad(
        outputs=u_pred,
        inputs=coords,
        grad_outputs=torch.ones_like(u_pred).to(device),
        retain_graph=True,
        create_graph=True
    )[0]
    u_grad_x = u_grad[:, 0]
    u_grad_y = u_grad[:, 1]

    # 2) Compute |∇u|^(p−2)
    grad_norm_power = (u_grad_x**2 + u_grad_y**2)**((p - 2) / 2)

    # 3) Form flux q = |∇u|^(p−2) ∇u
    flux_x = grad_norm_power * u_grad_x
    flux_y = grad_norm_power * u_grad_y

    # 4) Compute divergence ∇·q = ∂q_x/∂x + ∂q_y/∂y
    div_q_x = autograd.grad(
        outputs=flux_x,
        inputs=coords,
        grad_outputs=torch.ones_like(flux_x).to(device),
        retain_graph=True,
        create_graph=True
    )[0][:, 0]
    div_q_y = autograd.grad(
        outputs=flux_y,
        inputs=coords,
        grad_outputs=torch.ones_like(flux_y).to(device),
        retain_graph=True,
        create_graph=True
    )[0][:, 1]

    p_laplacian = div_q_x + div_q_y

    # 5) PDE residual: −∇·( |∇u|^(p−2) ∇u ) − 1
    return -p_laplacian - 1

# 2) instantiate PINN
model = PINN(
    pde_residual=pde_residual,
    input_dim=2,
    hidden_dim=50,
    output_dim=1,
    num_hidden_layers=4,
    device=device,
    Nd=50, # number of boundary points
    Nc=1000, # number of collocation points
    optimizer_type = "adam",
    lr = 1e-4,
)
model.double()  # convert model to double precision
model.to(device)
print("Using device:", next(model.parameters()).device)

# grab the data generated in PINN.__init__
x_train_Nu = model.x_train_Nu
u_train_Nu = model.u_train_Nu
x_train_Nf = model.x_train_Nf

TDV.training_data_plot(x_train_Nu, x_train_Nf)

loss_values, elapsed_time = model.execute_training_loop(loss_threshold=1e-4, bc_weight=10)
print('Training time: %.2f s' % (elapsed_time))

# Define a grid over input domain
n = 100  # grid resolution
x = np.linspace(-1, 1, n)
y = np.linspace(-1, 1, n)
X, Y = np.meshgrid(x, y)
XY = np.stack([X.ravel(), Y.ravel()], axis=1)
XY_tensor = torch.tensor(XY, dtype=torch.double).to(device)

# Predict using the trained model
with torch.no_grad():
    u_pred = model.model.forward(XY_tensor).cpu().numpy().reshape(n, n)

# Compute the real solution
p = 2.0  # Make sure this matches your PDE
N = 2
C = (p - 1) / p * N ** (1 / (1 - p))
u_real = np.zeros_like(X)
u_real[:, :] = C * (1 - np.sqrt(X[:, :] ** 2 + Y[:, :] ** 2) ** (p / (p - 1)))

# Mask for circular domain
def circular_mask(x, y):
    return x ** 2 + y ** 2 <= 1 ** 2  # Points inside a circle of radius 1

mask = circular_mask(X, Y)

# Masked arrays for plotting
u_pred_masked = np.ma.array(u_pred, mask=~mask)
u_real_masked = np.ma.array(u_real, mask=~mask)
error_masked = np.ma.array(np.abs(u_pred - u_real), mask=~mask)

# Specify the directory to save the model and graph
current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
save_dir = os.path.join(repo_root, "saved_model_and_graph", current_time)
os.makedirs(save_dir, exist_ok=True)

# Save the model
optimizer_type = getattr(model.optimizer, "optimizer_type", "adam")
save_model_path = os.path.join(save_dir, f"{optimizer_type}.pt")
torch.save(model.state_dict(), save_model_path)
print(f"Model saved to {save_model_path}")

# Save the loss function graph
plt.figure(figsize=(8, 5))
plt.title("Loss function")
plt.semilogy(loss_values, label="Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
save_loss_graph = os.path.join(save_dir, "loss_function_graph.png")
plt.savefig(save_loss_graph, bbox_inches="tight", dpi=150)
print(f"Loss function graph saved to {save_model_path}")

# Save the predictions and real solution
plt.figure(figsize=(16, 4))

# Plot Model Prediction
plt.subplot(1, 3, 1)
contour1 = plt.contourf(X, Y, u_pred_masked, levels=50, cmap='hsv')
plt.colorbar(contour1, label='u_pred')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Model Prediction')

# Plot Real Solution
plt.subplot(1, 3, 2)
contour2 = plt.contourf(X, Y, u_real_masked, levels=50, cmap='hsv')
plt.colorbar(contour2, label='u_real')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Real Solution')

# Plot Absolute Error
plt.subplot(1, 3, 3)
contour3 = plt.contourf(X, Y, error_masked, levels=50, cmap='inferno')
plt.colorbar(contour3, label='|u_pred - u_real|')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Absolute Error')

save_predictions_path = os.path.join(save_dir, "predictions_and_error.png")
plt.tight_layout()
plt.savefig(save_predictions_path, bbox_inches="tight", dpi=450)