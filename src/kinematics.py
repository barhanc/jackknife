import math
import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Dict, Any, Optional

@dataclass
class VehicleDimensions:
    L0: float = 3.5    # Tractor wheelbase (m)
    L1: float = 6.0    # Trailer 1 hitch-to-axle distance (m)
    L2: float = 6.0    # Trailer 2 hitch-to-axle distance (m)
    cab_width: float = 2.4
    cab_length: float = 5.0
    cab_front_overhang: float = 1.0
    cab_rear_overhang: float = 0.5
    trailer1_width: float = 2.5
    trailer1_length: float = 7.5
    trailer1_rear_overhang: float = 1.5
    trailer2_width: float = 2.5
    trailer2_length: float = 7.5
    trailer2_rear_overhang: float = 1.5
    max_steer_rad: float = math.radians(35.0)  # +/- 35 degrees
    max_steer_rate: float = math.radians(45.0) # +/- 45 deg/sec
    crit_hitch_rad: float = math.radians(70.0) # 70 degrees jackknife threshold


@dataclass
class VehicleState:
    # Pose of Tractor drive axle
    x0: float = 0.0
    y0: float = 0.0
    theta0: float = 0.0

    # Articulation angles
    beta1: float = 0.0  # theta0 - theta1
    beta2: float = 0.0  # theta1 - theta2

    # Steering angle
    phi: float = 0.0

    @property
    def theta1(self) -> float:
        return self.theta0 - self.beta1

    @property
    def theta2(self) -> float:
        return self.theta1 - self.beta2

    def wrap_angles(self):
        self.theta0 = math.atan2(math.sin(self.theta0), math.cos(self.theta0))
        self.beta1 = math.atan2(math.sin(self.beta1), math.cos(self.beta1))
        self.beta2 = math.atan2(math.sin(self.beta2), math.cos(self.beta2))
        self.phi = max(-VehicleDimensions.max_steer_rad, min(VehicleDimensions.max_steer_rad, self.phi))


class TractorDoubleTrailerKinematics:
    """
    Standard B-Double / Tractor + 2 Semi-trailers kinematic model under pure rolling conditions.
    State representation: [x0, y0, theta0, beta1, beta2]
    Tractor drive axle at (x0, y0) with heading theta0.
    Trailer 1 axle at (x1, y1) with heading theta1 = theta0 - beta1.
    Trailer 2 axle at (x2, y2) with heading theta2 = theta1 - beta2.
    """
    def __init__(self, dims: Optional[VehicleDimensions] = None):
        self.dims = dims or VehicleDimensions()

    def get_trailer1_axle(self, state: VehicleState) -> Tuple[float, float, float]:
        """Returns (x1, y1, theta1) of Trailer 1 axle."""
        theta1 = state.theta0 - state.beta1
        x1 = state.x0 - self.dims.L1 * math.cos(theta1)
        y1 = state.y0 - self.dims.L1 * math.sin(theta1)
        return x1, y1, theta1

    def get_trailer2_axle(self, state: VehicleState) -> Tuple[float, float, float]:
        """Returns (x2, y2, theta2) of Trailer 2 rear axle (the primary target tracking axle)."""
        x1, y1, theta1 = self.get_trailer1_axle(state)
        theta2 = theta1 - state.beta2
        x2 = x1 - self.dims.L2 * math.cos(theta2)
        y2 = y1 - self.dims.L2 * math.sin(theta2)
        return x2, y2, theta2

    def derivatives(self, state_arr: np.ndarray, v0: float, phi: float) -> np.ndarray:
        """
        Calculates time derivatives [dx0/dt, dy0/dt, dtheta0/dt, dbeta1/dt, dbeta2/dt]
        given tractor speed v0 and steering angle phi.
        """
        x0, y0, theta0, beta1, beta2 = state_arr
        L0, L1, L2 = self.dims.L0, self.dims.L1, self.dims.L2

        tan_phi = math.tan(phi)
        cos_b1 = math.cos(beta1)
        sin_b1 = math.sin(beta1)
        cos_b2 = math.cos(beta2)
        sin_b2 = math.sin(beta2)

        # Tractor drive axle translational velocity
        dx0 = v0 * math.cos(theta0)
        dy0 = v0 * math.sin(theta0)

        # Pure non-holonomic no-slip rolling constraints:
        # For trailer i, hitch is at preceding body with longitudinal speed v_{i-1}.
        # Zero lateral tire slip (v_axle . n_lateral = 0) strictly dictates:
        #   dtheta_1/dt = (v0 / L1) * sin(beta1)
        #   dtheta_2/dt = (v1 / L2) * sin(beta2) = (v0 * cos(beta1) / L2) * sin(beta2)
        # (Note: An extra cos(beta_i) factor previously caused artificial lateral wheel skidding at large angles).
        dtheta0 = (v0 / L0) * tan_phi
        dtheta1 = (v0 / L1) * sin_b1
        dtheta2 = (v0 * cos_b1 / L2) * sin_b2

        # Hitch angle derivatives:
        # dot(beta1) = dot(theta0) - dot(theta1)
        # dot(beta2) = dot(theta1) - dot(theta2)
        dbeta1 = dtheta0 - dtheta1
        dbeta2 = dtheta1 - dtheta2

        return np.array([dx0, dy0, dtheta0, dbeta1, dbeta2], dtype=np.float64)

    def step_rk4(self, state: VehicleState, v0: float, target_phi: float, dt: float) -> VehicleState:
        """
        Performs 4th-order Runge-Kutta integration step with rate-limited steering.
        """
        # Update steering angle with rate limit
        steer_diff = target_phi - state.phi
        max_change = self.dims.max_steer_rate * dt
        if abs(steer_diff) > max_change:
            phi_new = state.phi + math.copysign(max_change, steer_diff)
        else:
            phi_new = target_phi
        phi_new = max(-self.dims.max_steer_rad, min(self.dims.max_steer_rad, phi_new))

        # RK4 integration of kinematic state
        y = np.array([state.x0, state.y0, state.theta0, state.beta1, state.beta2], dtype=np.float64)

        k1 = self.derivatives(y, v0, phi_new)
        k2 = self.derivatives(y + 0.5 * dt * k1, v0, phi_new)
        k3 = self.derivatives(y + 0.5 * dt * k2, v0, phi_new)
        k4 = self.derivatives(y + dt * k3, v0, phi_new)

        y_next = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        new_state = VehicleState(
            x0=float(y_next[0]),
            y0=float(y_next[1]),
            theta0=float(y_next[2]),
            beta1=float(y_next[3]),
            beta2=float(y_next[4]),
            phi=phi_new
        )
        new_state.wrap_angles()
        return new_state

    def is_jackknifed(self, state: VehicleState) -> bool:
        crit = self.dims.crit_hitch_rad
        return abs(state.beta1) >= crit or abs(state.beta2) >= crit

    def jackknife_proximity(self, state: VehicleState) -> float:
        """
        Returns stability proximity meter rho in [0.0, 1.0]:
        1.0 = perfectly straight
        0.0 = jackknife limit reached
        """
        crit = self.dims.crit_hitch_rad
        ratio = max(abs(state.beta1) / crit, abs(state.beta2) / crit)
        return max(0.0, 1.0 - ratio)

    def get_body_polygons(self, state: VehicleState) -> Dict[str, np.ndarray]:
        """
        Computes 2D polygon vertices for rendering Tractor Cab, Trailer 1, and Trailer 2.
        Returns dict with keys: 'cab', 'trailer1', 'trailer2', 'wheels', 'hitches'.
        """
        dims = self.dims
        x0, y0, th0 = state.x0, state.y0, state.theta0
        x1, y1, th1 = self.get_trailer1_axle(state)
        x2, y2, th2 = self.get_trailer2_axle(state)

        def make_box(center_x, center_y, heading, length, width, rear_axle_offset):
            # Center of box relative to axle
            # Axle is located at rear_axle_offset from the rear edge of the body
            c_x = center_x + (length / 2.0 - rear_axle_offset) * math.cos(heading)
            c_y = center_y + (length / 2.0 - rear_axle_offset) * math.sin(heading)

            cos_h, sin_h = math.cos(heading), math.sin(heading)
            hl = length / 2.0
            hw = width / 2.0

            local_corners = np.array([
                [hl, hw],
                [hl, -hw],
                [-hl, -hw],
                [-hl, hw]
            ])
            R = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
            rotated = local_corners @ R.T
            rotated[:, 0] += c_x
            rotated[:, 1] += c_y
            return rotated

        # Cab: drive axle is (x0, y0). Cab front axle is at (x0 + L0*cos(th0), y0 + L0*sin(th0))
        # Overall cab length has rear overhang cab_rear_overhang behind drive axle
        cab_poly = make_box(x0, y0, th0, dims.cab_length, dims.cab_width, dims.cab_rear_overhang)

        # Trailer 1: axle at (x1, y1), hitch is at (x0, y0)
        trailer1_poly = make_box(x1, y1, th1, dims.trailer1_length, dims.trailer1_width, dims.trailer1_rear_overhang)

        # Trailer 2: axle at (x2, y2), hitch is at (x1, y1)
        trailer2_poly = make_box(x2, y2, th2, dims.trailer2_length, dims.trailer2_width, dims.trailer2_rear_overhang)

        # Wheels coordinates
        steer_th = th0 + state.phi
        cos_s, sin_s = math.cos(steer_th), math.sin(steer_th)
        cos_t, sin_t = math.cos(th0), math.sin(th0)

        # Front steering wheels
        fx = x0 + dims.L0 * cos_t
        fy = y0 + dims.L0 * sin_t
        f_left = np.array([fx - dims.cab_width * 0.45 * sin_t, fy + dims.cab_width * 0.45 * cos_t])
        f_right = np.array([fx + dims.cab_width * 0.45 * sin_t, fy - dims.cab_width * 0.45 * cos_t])

        return {
            'cab': cab_poly,
            'trailer1': trailer1_poly,
            'trailer2': trailer2_poly,
            'hitches': {
                'hitch1': np.array([x0, y0]),
                'hitch2': np.array([x1, y1])
            },
            'axles': {
                'tractor_drive': np.array([x0, y0]),
                'tractor_steer': np.array([fx, fy]),
                'trailer1': np.array([x1, y1]),
                'trailer2': np.array([x2, y2])
            },
            'steer_wheels': {
                'left': f_left,
                'right': f_right,
                'heading': steer_th
            }
        }
