import math
import numpy as np
import scipy.linalg
from typing import Tuple, Optional
from src.kinematics import VehicleDimensions, VehicleState

class ReverseLQRController:
    """
    Finite/infinite-horizon LQR stabilizer for multi-trailer reverse motion.
    Linearizes hitch dynamics around nominal operating conditions:
    state e = [delta_beta1, delta_beta2]
    control delta_phi
    """
    def __init__(self, dims: Optional[VehicleDimensions] = None):
        self.dims = dims or VehicleDimensions()
        # Q weights for [delta_beta1, delta_beta2]
        self.Q = np.diag([10.0, 25.0])
        # R weight for steering effort
        self.R = np.array([[1.0]])

    def compute_linearized_matrices(self, beta1: float, beta2: float, phi: float, v0: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes Jacobian matrices A = df/d(beta), B = df/d(phi)
        for reverse speed v0 < 0.
        """
        L0, L1, L2 = self.dims.L0, self.dims.L1, self.dims.L2
        cos_b1 = math.cos(beta1)
        sin_b1 = math.sin(beta1)
        cos_b2 = math.cos(beta2)
        sin_b2 = math.sin(beta2)
        cos_phi = math.cos(phi)

        # Pure rolling hitch dynamics Jacobians:
        # dot(beta1) = (v0/L0)*tan(phi) - (v0/L1)*sin(beta1)
        # dot(beta2) = (v0/L1)*sin(beta1) - (v0*cos(beta1)/L2)*sin(beta2)
        #
        # d(dot_beta1)/d(beta1) = -(v0 / L1) * cos(beta1)
        # d(dot_beta1)/d(beta2) = 0
        a11 = -(v0 / L1) * cos_b1
        a12 = 0.0

        # d(dot_beta2)/d(beta1) = (v0 / L1) * cos(beta1) + (v0 / L2) * sin_b1 * sin_b2
        # d(dot_beta2)/d(beta2) = -(v0 * cos_b1 / L2) * cos_b2
        a21 = (v0 / L1) * cos_b1 + (v0 / L2) * sin_b1 * sin_b2
        a22 = -(v0 * cos_b1 / L2) * cos_b2

        A = np.array([[a11, a12],
                      [a21, a22]], dtype=np.float64)

        # d(dot_beta1)/d(phi) = v0 / (L0 * cos^2(phi))
        # d(dot_beta2)/d(phi) = 0
        b1 = v0 / (L0 * max(0.01, cos_phi)**2)
        b2 = 0.0

        B = np.array([[b1],
                      [b2]], dtype=np.float64)

        return A, B

    def compute_gain(self, beta1: float, beta2: float, phi: float, v0: float = -1.5) -> np.ndarray:
        """
        Solves continuous algebraic Riccati equation:
        A^T P + P A - P B R^-1 B^T P + Q = 0
        Returns gain K: u = -K * e
        """
        A, B = self.compute_linearized_matrices(beta1, beta2, phi, v0)

        try:
            P = scipy.linalg.solve_continuous_are(A, B, self.Q, self.R)
            K = np.linalg.inv(self.R) @ (B.T @ P)
            return K
        except Exception:
            # Fallback heuristic stabilization gain
            return np.array([[1.5, 3.0]])

    def corrective_steering(
        self,
        nominal_phi: float,
        nominal_beta1: float,
        nominal_beta2: float,
        actual_beta1: float,
        actual_beta2: float,
        v0: float = -1.5
    ) -> float:
        """
        Computes stabilizing steering command:
        phi* = clip(phi_nom - K * e, -max_steer, +max_steer)
        """
        e = np.array([actual_beta1 - nominal_beta1, actual_beta2 - nominal_beta2])
        K = self.compute_gain(nominal_beta1, nominal_beta2, nominal_phi, v0)
        delta_phi = float((K @ e).item() if hasattr(K @ e, 'item') else (K @ e)[0])

        phi_star = nominal_phi - delta_phi
        max_rad = self.dims.max_steer_rad
        return max(-max_rad, min(max_rad, phi_star))
