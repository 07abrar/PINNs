import numpy as np
from scipy.stats import qmc


class BoundaryPoints:
    """
    Class to generate boundary points for PINNs.

    Attributes:
        Nd (int): Number of boundary points.
    """

    def __init__(self, Nd: int = 50):
        """
        Initialize the class with the number of boundary points.

        Args:
            Nd (int): Number of boundary points.
        """
        self.Nd = Nd

    def generate_boundary_points(self):
        """
        Generate boundary points on a unit circle.

        Returns:
            boundary_train (np.ndarray): Boundary points (x, y).
            boundary_train_u (np.ndarray): Boundary condition values (u).
        """
        engine = qmc.LatinHypercube(d=1)
        bd = engine.random(n=self.Nd)
        theta = bd[:, 0] * 2 * np.pi
        data_boundary = np.zeros([self.Nd, 3])
        data_boundary[:, 0] = np.cos(theta)
        data_boundary[:, 1] = np.sin(theta)
        data_boundary[:, 2] = 0  # Boundary condition value (u)
        boundary_train = data_boundary[:, :2]
        boundary_train_u = data_boundary[:, 2][:, None]
        return boundary_train, boundary_train_u


class CollocationPoints:
    """
    Class to generate collocation points for PINNs.

    Attributes:
        Nc (int): Number of collocation points.
    """

    def __init__(self, Nc=1000):
        """
        Initialize the class with the number of collocation points.

        Args:
            Nc (int): Number of collocation points.
        """
        self.Nc = Nc

    def generate_collocation_points(self):
        """
        Generate collocation points inside a unit circle.

        Returns:
            X_train_Nf1 (np.ndarray): Collocation points (x, y).
        """
        engine = qmc.LatinHypercube(d=2)
        internal_points = engine.random(n=self.Nc)
        radius = np.sqrt(internal_points[:, 0])
        angles = internal_points[:, 1] * 2 * np.pi
        X_train_Nf1 = np.zeros([self.Nc, 2])
        X_train_Nf1[:, 0] = radius * np.cos(angles)
        X_train_Nf1[:, 1] = radius * np.sin(angles)
        return X_train_Nf1


class TrainingDataGenerator:
    """
    Class to combine boundary and collocation points for training data.
    """

    def __init__(self, Nd=50, Nc=1000):
        """
        Initialize the class with the number of boundary and collocation points.

        Args:
            Nd (int): Number of boundary points.
            Nc (int): Number of collocation points.
        """
        self.boundary_points = BoundaryPoints(Nd)
        self.collocation_points = CollocationPoints(Nc)

    def generate_training_data(self):
        """
        Generate training data by combining boundary and collocation points.

        Returns:
            X_train_Nu (np.ndarray): Boundary points (x, y).
            U_train_Nu (np.ndarray): Boundary condition values (u).
            X_train_Nf (np.ndarray): Combined boundary and collocation points (x, y).
        """
        X_train_Nu, U_train_Nu = self.boundary_points.generate_boundary_points()
        X_train_Nf1 = self.collocation_points.generate_collocation_points()
        X_train_Nf = np.vstack([X_train_Nu, X_train_Nf1])
        return X_train_Nu, U_train_Nu, X_train_Nf
