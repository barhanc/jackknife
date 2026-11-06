import math
import numpy as np
import scipy.linalg
from typing import Tuple, Optional
from src.kinematics import VehicleDimensions

class ReverseLQRController:
    """
    Stabilized finite/infinite-horizon LQR controller for multi-trailer reversing.
    Linearizes hitch dynamics around nominal reversing operating points:
      state error: e = [delta_beta1, delta_beta2]
      control error: u = delta_phi

    Tuned with high steering penalty R=25.0 and balanced Q=diag([3.0, 6.0])
    to prevent explosive gain peaks and saturated steering slamming on fast chicanes.
    """
    def __init__(
        self,
        dims: Optional[VehicleDimensions] = None,
        q_diag: Tuple[float, float] = (3.0, 6.0),
        r_val: float = 25.0,
        max_delta_phi_deg: float = 18.0
    ):
        self.dims = dims or VehicleDimensions()
        # Balanced state weights on hitch deviations: [delta_beta1, delta_beta2]
        self.Q = np.diag([q_diag[0], q_diag[1]])
        # High control weight penalizing violent counter-steering
        self.R = np.array([[r_val]])
        # Maximum allowed perturbation steering offset (+/- 18 degrees)
        self.max_delta_phi = math.radians(max_delta_phi_deg)

    def compute_linearized_matrices(
        self,
        beta1: float,
        beta2: float,
        phi: float,
        v0: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes continuous Jacobian matrices A = df/d(beta), B = df/d(phi)
        for reverse longitudinal speed v0 < 0.
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
        a11 = -(v0 / L1) * cos_b1
        a12 = 0.0

        a21 = (v0 / L1) * cos_b1 + (v0 / L2) * sin_b1 * sin_b2
        a22 = -(v0 * cos_b1 / L2) * cos_b2

        A = np.array([
            [a11, a12],
            [a21, a22]
        ], dtype=np.float64)

        b1 = v0 / (L0 * max(0.01, cos_phi)**2)
        b2 = 0.0

        B = np.array([
            [b1],
            [b2]
        ], dtype=np.float64)

        return A, B

    def compute_gain(
        self,
        beta1: float,
        beta2: float,
        phi: float,
        v0: float = -1.5
    ) -> np.ndarray:
        """
        Solves continuous algebraic Riccati equation:
          A^T P + P A - P B R^-1 B^T P + Q = 0
        Returns optimal feedback gain K: u = -K * e
        """
        A, B = self.compute_linearized_matrices(beta1, beta2, phi, v0)

        try:
            P = scipy.linalg.solve_continuous_are(A, B, self.Q, self.R)
            K = np.linalg.inv(self.R) @ (B.T @ P)
            return K
        except Exception:
            # Well-damped heuristic stabilization gain fallback
            return np.array([[-3.2, 2.9]])

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
          phi* = clip(nominal_phi - clip(K * e, -max_delta, max_delta), -max_steer, +max_steer)
        """
        e = np.array([actual_beta1 - nominal_beta1, actual_beta2 - nominal_beta2])
        K = self.compute_gain(nominal_beta1, nominal_beta2, nominal_phi, v0)
        ke = K @ e
        delta_phi = float(ke.item() if hasattr(ke, 'item') else ke[0])

        # Clamping delta_phi avoids saturated slamming on fast chicanes
        bounded_delta = float(np.clip(delta_phi, -self.max_delta_phi, self.max_delta_phi))

        phi_star = nominal_phi - bounded_delta
        max_rad = self.dims.max_steer_rad
        return float(np.clip(phi_star, -max_rad, max_rad))
