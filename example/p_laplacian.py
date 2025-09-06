from datetime import datetime
import os

import numpy as np
import torch
from torch import autograd

import src as pinns


def my_pde_residual(coords: torch.Tensor, u_pred: torch.Tensor) -> torch.Tensor:
    """
    Compute the residual of the 2D p-Laplacian Poisson equation

        −∇·( |∇u|^(p-2) ∇u ) = 1

    Args:
        coords (torch.Tensor): shape (N,2) input points (x,y), requires_grad=True
        u_pred (torch.Tensor): shape (N,1) model's predicted u at coords

    Returns:
        torch.Tensor: PDE residual r(x,y) = −∇·( |∇u|^(p-2) ∇u ) - 1
    """
    device = coords.device
    p = 2.0  # p-Laplacian exponent
    # 1) Compute gradient ∇u = (∂u/∂x, ∂u/∂y)
    u_grad = autograd.grad(
        outputs=u_pred,
        inputs=coords,
        grad_outputs=torch.ones_like(u_pred).to(device),
        retain_graph=True,
        create_graph=True,
    )[0]
    u_grad_x = u_grad[:, 0]
    u_grad_y = u_grad[:, 1]

    # 2) Compute |∇u|^(p−2)
    grad_norm_power = (u_grad_x**2 + u_grad_y**2) ** ((p - 2) / 2)

    # 3) Form flux q = |∇u|^(p−2) ∇u
    flux_x = grad_norm_power * u_grad_x
    flux_y = grad_norm_power * u_grad_y

    # 4) Compute divergence ∇·q = ∂q_x/∂x + ∂q_y/∂y
    div_q_x = autograd.grad(
        outputs=flux_x,
        inputs=coords,
        grad_outputs=torch.ones_like(flux_x).to(device),
        retain_graph=True,
        create_graph=True,
    )[0][:, 0]
    div_q_y = autograd.grad(
        outputs=flux_y,
        inputs=coords,
        grad_outputs=torch.ones_like(flux_y).to(device),
        retain_graph=True,
        create_graph=True,
    )[0][:, 1]

    p_laplacian = div_q_x + div_q_y

    # 5) PDE residual: −∇·( |∇u|^(p−2) ∇u ) − 1
    return -p_laplacian - 1


# 1. Define domain
domain = pinns.CircularDomain(
    center=(0, 0),
    radius=1,
    training_data={"boundary": 100, "collocation": 1000},
)
# or
# domain = pinns.PolygonDomain(vertices=[(0, 0), (1, 1), (1, 0)])
# domain = pinns.RectangularDomain(x_range=(0, 1), y_range=(0, 1))

domain.training_data_plot()

# 2. Define problem
problem = pinns.PDEProblem(
    residual_fn=my_pde_residual, boundary_conditions={"dirichlet": 0.0}
)

# 3. Create PINN (network + loss only)
model = pinns.NeuralNet(
    input_dim=2,
    hidden_dim=50,
    output_dim=1,
    num_hidden_layers=4,
    activation="tanh",
)

# 4. Create trainer with strategy
trainer = pinns.Trainer(
    model=model,
    problem=problem,
    domain=domain,
    optimizer_config={"type": "adam", "lr": 1e-2},
    strategy="standard",  # or pinns.AdaptiveSamplingStrategy()
)
# 5. Train
results = trainer.train(epochs=10000, loss_threshold=1e-5)

# 6. Fix later

# Define a grid over input domain
n = 100  # grid resolution
x = np.linspace(-1, 1, n)
y = np.linspace(-1, 1, n)
X, Y = np.meshgrid(x, y)
XY = np.stack([X.ravel(), Y.ravel()], axis=1)
XY_tensor = torch.tensor(XY, dtype=model.dtype).to(model.device)

# Predict using the trained model
with torch.no_grad():
    u_pred = model.forward(XY_tensor).cpu().numpy().reshape(n, n)

# Compute the real solution
p = 2.0  # Make sure this matches your PDE
N = 2
C = (p - 1) / p * N ** (1 / (1 - p))
u_real = np.zeros_like(X)
u_real[:, :] = C * (1 - np.sqrt(X[:, :] ** 2 + Y[:, :] ** 2) ** (p / (p - 1)))

circular_mask = domain.visualization_mask(X, Y)

# Use visualization helpers for plotting and saving
current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
save_dir = os.path.join(repo_root, "saved_model_and_graph", current_time)
os.makedirs(save_dir, exist_ok=True)

# Save the model
# optimizer_type = getattr(trainer.optimizer, "adam")
optimizer_type = "adam"
save_model_path = os.path.join(save_dir, f"{optimizer_type}.pt")
torch.save(model.state_dict(), save_model_path)
print(f"Model saved to {save_model_path}")

# Loss curve
save_loss_graph = os.path.join(save_dir, "loss_function_graph.png")
pinns.loss_curve(
    results["loss_history"],
    save_path=save_loss_graph,
    title="Loss function",
    show=False,
    dpi=150,
)

# Predictions and error figure
save_predictions_path = os.path.join(save_dir, "predictions_and_error.png")
pinns.prediction_and_error(
    X=X,
    Y=Y,
    u_pred=u_pred,
    u_real=u_real,
    mask=circular_mask,
    save_path=save_predictions_path,
    cmap_pred="hsv",
    cmap_real="hsv",
    cmap_err="inferno",
    levels=50,
    dpi=450,
    show=False,
)
