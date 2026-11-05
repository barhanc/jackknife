import math
import numpy as np
import scipy.linalg
from typing import Tuple, Optional
from src.kinematics import VehicleDimensions

class ReverseLQRController:
    """
    Stabilized LQR controller with Non-Linear Articulation Protection for multi-trailer reversing.
    Uses continuous Riccati optimal feedback for small/moderate tracking errors,
    paired with strict non-linear priority protection when articulation angles approach danger boundaries.
    """
    def __init__(self, dims: Optional[VehicleDimensions] = None):
        self.dims = dims or VehicleDimensions()
        # Balanced weights on articulation angles: [delta_beta1, delta_beta2]
        self.Q = np.diag([4.0, 8.0])
        # Balanced control weight (R=8.0 prevents both gain explosion and sluggish under-steering)
        self.R = np.array([[8.0]])
        # Maximum allowed perturbation steering offset (+/- 22 degrees)
        self.max_delta_phi = math.radians(28.0)

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
            return np.array([[-3.6, 3.8]])

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
        Computes stabilizing steering command with non-linear anti-jackknife protection:
        1. Evaluates LQR feedback around nominal operating point
        2. Applies Non-Linear Cascade Protection during chicane inflections (opposite signs)
        3. Enforces hard safety limits
        """
        e1 = actual_beta1 - nominal_beta1
        e2 = actual_beta2 - nominal_beta2
        e = np.array([e1, e2])

        K = self.compute_gain(nominal_beta1, nominal_beta2, nominal_phi, v0)
        ke = K @ e
        delta_phi = float(ke.item() if hasattr(ke, 'item') else ke[0])

        # Non-Linear Inflection & Anti-Jackknife Cascade Protection:
        # In severe S-curve inflections (where Trailer 2 is heavily articulated AND opposing Trailer 1)
        # or when beta2 approaches the physical cliff (> 44 deg, where jackknife is at 70 deg),
        # linear LQR gains can reinforce each other and command full lock in the wrong direction.
        # We invoke cascaded outer/inner loop control:
        # 1. Outer loop sets the safe target articulation for Trailer 1 (b1_ref) needed to stabilize Trailer 2.
        # 2. Inner loop steers the tractor cab to track b1_ref without overshooting.
        is_severe_inflection = (actual_beta1 * actual_beta2 < -0.05) and (abs(actual_beta2) > math.radians(26.0))
        is_near_jackknife = abs(actual_beta2) > math.radians(44.0)
        if is_severe_inflection or is_near_jackknife:
            sin_b1_eq = (self.dims.L1 / self.dims.L2) * math.sin(nominal_beta2)
            sin_b1_cmd = sin_b1_eq + 1.2 * e2
            sin_b1_cmd = float(np.clip(sin_b1_cmd, -0.72, 0.72))  # Allow Trailer 1 up to 46 deg
            b1_ref = math.asin(sin_b1_cmd)
            err_b1 = actual_beta1 - b1_ref
            phi_star = nominal_phi + 1.8 * err_b1
        else:
            delta_phi = float(np.clip(delta_phi, -self.max_delta_phi, self.max_delta_phi))
            phi_star = nominal_phi - delta_phi

        max_rad = self.dims.max_steer_rad
        return float(np.clip(phi_star, -max_rad, max_rad))
