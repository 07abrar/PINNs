import matplotlib.pyplot as plt
# Add numpy for masking and array ops used in plotting helpers
import numpy as np


class TrainingDataVisualizer:
    """
    Class to visualize the point distribution of boundary and collocation points.
    """
    @staticmethod
    def training_data_plot(X_train_Nu, X_train_Nf, save_path=None):
        """
        Visualize the boundary and collocation points.

        Args:
            X_train_Nu (np.ndarray): Boundary points (x, y).
            X_train_Nf (np.ndarray): Combined boundary and collocation points (x, y).
            save_path (str): File path to save the plot (optional).
        """
        plt.figure(figsize=(8, 8))
        plt.title("Boundary Data Points and Collocation Points")
        # Plot boundary points
        plt.scatter(
            X_train_Nu[:, 0], X_train_Nu[:, 1],
            marker="x", c="b", label="Boundary Points (BDP)"
        )
        # Plot collocation points
        plt.scatter(
            X_train_Nf[:, 0], X_train_Nf[:, 1],
            marker=".", c="r", label="Collocation Points (CP)"
        )
        # Draw unit circle
        circle = plt.Circle((0, 0), 1, color='green', fill=False)
        plt.gca().add_artist(circle)

        plt.axis('equal')
        plt.legend()
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid(True)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()

    # New: loss curve plotting helper
    @staticmethod
    def loss_curve(loss_values, save_path=None, title="Loss function", show=False, dpi=150):
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

    # New: predictions vs exact and error plotting helper
    @staticmethod
    def prediction_and_error(
        X,
        Y,
        u_pred,
        u_real=None,
        mask=None,
        save_path=None,
        cmap_pred='hsv',
        cmap_real='hsv',
        cmap_err='inferno',
        levels=50,
        dpi=300,
        show=False,
    ):
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
                if u_real is not None else None
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
        plt.colorbar(c1, label='u_pred')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Model Prediction')

        if u_real is not None:
            # Exact
            plt.subplot(1, cols, 2)
            c2 = plt.contourf(X, Y, u_real_plot, levels=levels, cmap=cmap_real)
            plt.colorbar(c2, label='u_real')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title('Real Solution')

            # Error
            plt.subplot(1, cols, 3)
            c3 = plt.contourf(X, Y, err_plot, levels=levels, cmap=cmap_err)
            plt.colorbar(c3, label='|u_pred - u_real|')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title('Absolute Error')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=dpi)
            print(f"Predictions figure saved to {save_path}")
        if show and not save_path:
            plt.show()
        plt.close()
