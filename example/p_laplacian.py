from datetime import datetime
import os

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

# 6. Visualization and saving

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

# Prediction surface
save_surface_path = os.path.join(save_dir, "prediction_surface.png")
pinns.prediction_surface(
    model=model,
    domain=domain,
    mask_fn=domain.visualization_mask,
    save_path=save_surface_path,
    cmap="hsv",
    dpi=450,
    show=False,
)

# Domain loss heatmap
save_domain_loss_path = os.path.join(save_dir, "domain_loss_heatmap.png")
pinns.domain_loss_heatmap(
    model=model,
    problem=problem,
    domain=domain,
    mask_fn=domain.visualization_mask,
    save_path=save_domain_loss_path,
    cmap="inferno",
    dpi=450,
    show=False,
)
