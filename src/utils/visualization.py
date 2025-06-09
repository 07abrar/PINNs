import matplotlib.pyplot as plt


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
