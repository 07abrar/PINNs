"""Visualization helpers for PINN experiments."""

from typing import Any, Callable, Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch


def training_data_plot(
    boundary_points: np.ndarray,
    collocation_points: np.ndarray,
    mask: Callable[[np.ndarray, np.ndarray], np.ndarray],
    save_path: str | None = None,
) -> None:
    """Visualize the boundary and collocation points."""

    plt.figure(figsize=(8, 8))
    plt.title("Boundary Data Points and Collocation Points")

    # Plot boundary points
    plt.scatter(
        boundary_points[:, 0],
        boundary_points[:, 1],
        marker="x",
        c="b",
        label="Boundary Points (BDP)",
    )

    # Plot collocation points
    plt.scatter(
        collocation_points[:, 0],
        collocation_points[:, 1],
        marker=".",
        c="r",
        label="Collocation Points (CP)",
    )

    all_points = np.vstack([boundary_points, collocation_points])
    x_min, x_max = all_points[:, 0].min() - 0.1, all_points[:, 0].max() + 0.1
    y_min, y_max = all_points[:, 1].min() - 0.1, all_points[:, 1].max() + 0.1
    x = np.linspace(x_min, x_max, 100)
    y = np.linspace(y_min, y_max, 100)
    x_grid, y_grid = np.meshgrid(x, y)
    mask_values = mask(x_grid, y_grid)

    plt.contour(x_grid, y_grid, mask_values, levels=[
                0.5], colors="green", linewidths=1)
    plt.axis("equal")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)

    # Save or show the plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


def loss_curve(
    loss_values: Iterable[float] | np.ndarray,
    ax: plt.Axes | None = None,
    save_path: str | None = None,
    title: str = "Loss function",
    show: bool = False,
    dpi: int = 150,
):
    """Plot training loss curve."""

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure

    ax.set_title(title)
    ax.semilogy(loss_values, label="Training Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=dpi)
        print(f"Loss function graph saved to {save_path}")
    if show:
        plt.show()

    if ax is None:
        plt.close(fig)
    return fig, ax


def prediction_and_error(
    X: np.ndarray,
    Y: np.ndarray,
    u_pred: np.ndarray,
    u_real: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    save_path: str | None = None,
    cmap_pred: str = "hsv",
    cmap_real: str = "hsv",
    cmap_err: str = "inferno",
    levels: int = 50,
    dpi: int = 300,
    show: bool = False,
) -> None:
    """
    Plot model prediction, optional exact solution, and absolute error.

    Args:
        X, Y: 2D meshgrid arrays
        u_pred: 2D array of predictions matching X, Y shape
        u_real: optional 2D array of exact solution
        mask: optional boolean mask with True for valid domain points
        save_path: optional file path to save the figure
    """
    # Apply mask if provided
    if mask is not None:
        u_pred_plot = np.ma.array(u_pred, mask=~mask)
        u_real_plot = np.ma.array(
            u_real, mask=~mask) if u_real is not None else None
        err_plot = (
            np.ma.array(np.abs(u_pred - u_real), mask=~mask)
            if u_real is not None
            else None
        )
    else:
        u_pred_plot = u_pred
        u_real_plot = u_real
        err_plot = np.abs(u_pred - u_real) if u_real is not None else None

    cols = 3 if u_real is not None else 1
    # plt.figure(figsize=(6 * cols, 4))
    plt.figure(figsize=(16, 4))

    # Prediction
    plt.subplot(1, cols, 1)
    c1 = plt.contourf(X, Y, u_pred_plot, levels=levels, cmap=cmap_pred)
    plt.colorbar(c1, label="u_pred")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Model Prediction")

    if u_real is not None:
        # Exact
        plt.subplot(1, cols, 2)
        c2 = plt.contourf(X, Y, u_real_plot, levels=levels, cmap=cmap_real)
        plt.colorbar(c2, label="u_real")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Real Solution")

        # Error
        plt.subplot(1, cols, 3)
        c3 = plt.contourf(X, Y, err_plot, levels=levels, cmap=cmap_err)
        plt.colorbar(c3, label="|u_pred - u_real|")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Absolute Error")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=dpi)
        print(f"Predictions figure saved to {save_path}")
    if show and not save_path:
        plt.show()
    plt.close()


def __prepare_plotting_grid(
    domain: Any,
    mask_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    resolution: float = 50,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a meshgrid and mask covering the domain."""

    boundary_points = domain.boundary_points
    collocation_points = domain.collocation_points
    all_points = np.vstack([boundary_points, collocation_points])
    x_min, x_max = all_points[:, 0].min() - 0.1, all_points[:, 0].max() + 0.1
    y_min, y_max = all_points[:, 1].min() - 0.1, all_points[:, 1].max() + 0.1
    span = max(x_max - x_min, y_max - y_min)
    n = max(int(span * resolution), 2)
    x = np.linspace(x_min, x_max, n)
    y = np.linspace(y_min, y_max, n)
    X, Y = np.meshgrid(x, y)
    mask = mask_fn(X, Y)
    return X, Y, mask


def prediction_surface(
    model: torch.nn.Module,
    domain: Any,
    plot_type: str = "2d",
    save_path: str | None = None,
    cmap: str = "viridis",
    dpi: int = 300,
    show: bool = False,
) -> None:
    """Plot the model prediction as a 2D or 3D surface."""

    mask_fn = domain.visualization_mask
    if mask_fn is None:
        raise ValueError("mask_fn must be provided")

    X, Y, mask = __prepare_plotting_grid(domain, mask_fn)
    XY = np.stack([X.ravel(), Y.ravel()], axis=1)
    param = next(model.parameters())
    XY_tensor = torch.tensor(XY, dtype=param.dtype, device=param.device)

    with torch.no_grad():
        u_pred = model.forward(XY_tensor).cpu().numpy().reshape(X.shape)

    u_pred = np.where(mask, u_pred, np.nan)

    fig = plt.figure(figsize=(8, 8))

    if plot_type in ["3d", "3D"]:
        ax = fig.add_subplot(1, 1, 1, projection="3d")
        surf = ax.plot_surface(X, Y, u_pred, cmap=cmap, linewidth=0)
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label="u_pred")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("u")
        ax.set_title("Model Prediction")
    elif plot_type in ["2d", "2D"]:
        ax = fig.add_subplot(1, 1, 1)
        contour = ax.contourf(X, Y, u_pred, levels=50, cmap=cmap)
        fig.colorbar(contour, ax=ax, shrink=0.6, aspect=14, label="u_pred")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(True)
        ax.set_title("Model Prediction")
    else:
        raise ValueError("plot_type must be either '2d' or '3d'")

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=dpi)
        print(f"Prediction surface saved to {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def domain_loss_heatmap(
    model: torch.nn.Module,
    problem: Any,
    domain: Any,
    save_path: str | None = None,
    cmap: str = "inferno",
    dpi: int = 300,
    show: bool = False,
) -> None:
    """Plot squared residual (domain loss) as a 2D heatmap."""

    mask_fn = domain.visualization_mask
    if mask_fn is None:
        raise ValueError("mask_fn must be provided")

    X, Y, mask = __prepare_plotting_grid(domain, mask_fn)
    XY = np.stack([X.ravel(), Y.ravel()], axis=1)
    param = next(model.parameters())
    XY_tensor = torch.tensor(XY, dtype=param.dtype,
                             device=param.device, requires_grad=True)

    u_pred_tensor = model.forward(XY_tensor)
    if problem is not None:
        residual = problem.compute_residual(XY_tensor, u_pred_tensor)
        domain_loss = residual.detach().cpu().numpy().reshape(X.shape) ** 2
    else:
        domain_loss = np.zeros_like(X)

    domain_loss = np.where(mask, domain_loss, np.nan)

    fig, ax = plt.subplots(figsize=(6, 5))
    c = ax.contourf(X, Y, domain_loss, levels=50, cmap=cmap)
    fig.colorbar(c, ax=ax, label="Domain Loss")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Domain Loss Heatmap")

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=dpi)
        print(f"Domain loss heatmap saved to {save_path}")
    if show:
        plt.show()
    plt.close(fig)
