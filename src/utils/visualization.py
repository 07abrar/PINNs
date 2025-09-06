"""Visualization helpers for PINN experiments."""

from typing import Callable, Iterable

import matplotlib.pyplot as plt
import numpy as np


def training_data_plot(
    boundary_points: np.ndarray,
    collocation_points: np.ndarray,
    mask: Callable[[np.ndarray, np.ndarray], np.ndarray],
    save_path: str | None = None,
) -> None:
    """
    Visualize the boundary and collocation points.
    """

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
                0.5], colors="green", linewidths=2)
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
    save_path: str | None = None,
    title: str = "Loss function",
    show: bool = False,
    dpi: int = 150,
) -> None:
    plt.figure(figsize=(8, 5))
    plt.title(title)
    plt.semilogy(loss_values, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=dpi)
        print(f"Loss function graph saved to {save_path}")
    if show and not save_path:
        plt.show()
    plt.close()


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
