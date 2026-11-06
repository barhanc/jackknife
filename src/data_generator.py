import math
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Dict, Any, Optional

from src.kinematics import TractorDoubleTrailerKinematics, VehicleDimensions, VehicleState
from src.path_generator import Path, generate_synthetic_path
from src.lqr_controller import ReverseLQRController
from src.model import steer_to_class_idx, NUM_STEER_CLASSES

class ReversingDataset(Dataset):
    """
    In-memory PyTorch Dataset for Multi-Trailer Reversing ControlTransformer.
    Contains (lookahead_tokens: 64x7, hitch_state: 4, steer_class: int).
    """
    def __init__(self, tokens: np.ndarray, hitch_states: np.ndarray, steer_classes: np.ndarray):
        self.tokens = torch.from_numpy(tokens).float()
        self.hitch_states = torch.from_numpy(hitch_states).float()
        self.steer_classes = torch.from_numpy(steer_classes).long()

    def __len__(self):
        return len(self.steer_classes)

    def __getitem__(self, idx):
        return self.tokens[idx], self.hitch_states[idx], self.steer_classes[idx]


class SyntheticDataGenerator:
    """
    Generates training data via closed-form time-reversal of forward trajectories
    plus LQR contraction funnel perturbations.
    """
    def __init__(self, dims: Optional[VehicleDimensions] = None):
        self.dims = dims or VehicleDimensions()
        self.kinematics = TractorDoubleTrailerKinematics(self.dims)
        self.lqr = ReverseLQRController(self.dims)

    def generate_single_trajectory(
        self,
        path_type: str = "smooth_random",
        length_m: float = 75.0,
        dt: float = 0.05,
        v_forward: float = 1.5
    ) -> List[Dict[str, Any]]:
        """
        Rolls out forward trajectory following a smooth path, then records states.
        """
        path = generate_synthetic_path(length_m=length_m, ds=0.8, path_type=path_type)
        if len(path.x) < 20:
            return []

        # Start tractor at initial path point with optional mild hitch misalignments
        rng = np.random.default_rng()
        b1_init = float(rng.uniform(-0.06, 0.06)) if rng.random() < 0.6 else 0.0
        b2_init = float(rng.uniform(-0.06, 0.06)) if rng.random() < 0.6 else 0.0
        state = VehicleState(
            x0=path.x[0],
            y0=path.y[0],
            theta0=path.headings[0],
            beta1=b1_init,
            beta2=b2_init,
            phi=0.0
        )

        steps = int((length_m - 15.0) / (v_forward * dt))
        history = []

        lookahead_dist = 4.5  # Pure pursuit lookahead for tractor cab
        fwd_nearest_idx = 0

        for step in range(steps):
            # Compute Pure Pursuit steering for tractor cab
            # Cab front axle location
            cos_t0, sin_t0 = math.cos(state.theta0), math.sin(state.theta0)
            fx = state.x0 + self.dims.L0 * cos_t0
            fy = state.y0 + self.dims.L0 * sin_t0

            # Target point along path ahead of front axle with monotonic curve tracking
            fwd_nearest_idx, _ = path.update_progression(fx, fy, fwd_nearest_idx)
            target_idx = min(len(path.x) - 1, fwd_nearest_idx + int(lookahead_dist / path.ds))
            gx = path.x[target_idx]
            gy = path.y[target_idx]

            # Vector to goal in tractor frame
            dx = gx - fx
            dy = gy - fy
            local_x = cos_t0 * dx + sin_t0 * dy
            local_y = -sin_t0 * dx + cos_t0 * dy

            dist_to_goal = math.hypot(local_x, local_y)
            if dist_to_goal < 0.5:
                # Steer towards tangent heading
                alpha = path.headings[target_idx] - state.theta0
                alpha = math.atan2(math.sin(alpha), math.cos(alpha))
                target_phi = math.atan2(2.0 * self.dims.L0 * math.sin(alpha), lookahead_dist)
            else:
                target_phi = math.atan2(2.0 * self.dims.L0 * local_y, dist_to_goal**2)

            target_phi = max(-self.dims.max_steer_rad, min(self.dims.max_steer_rad, target_phi))

            # Record state
            x1, y1, th1 = self.kinematics.get_trailer1_axle(state)
            x2, y2, th2 = self.kinematics.get_trailer2_axle(state)

            history.append({
                'state': VehicleState(state.x0, state.y0, state.theta0, state.beta1, state.beta2, state.phi),
                't2_axle': (x2, y2, th2),
                'phi': state.phi,
                'target_phi': target_phi
            })

            # Advance kinematics forward
            state = self.kinematics.step_rk4(state, v0=v_forward, target_phi=target_phi, dt=dt)
            if self.kinematics.is_jackknifed(state):
                break

        return history

    def extract_reversing_samples(
        self,
        forward_history: List[Dict[str, Any]],
        perturb_ratio: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Applies time-reversal to forward history and extracts (tokens, hitch_state, steer_class).
        Adds LQR perturbation recovery funnels.
        """
        N = len(forward_history)
        if N < 50:
            return np.zeros((0, 64, 7)), np.zeros((0, 4)), np.zeros(0, dtype=np.int64)

        # In time-reversal, Trailer 2 rear axle traces the exact curve in reverse.
        # Let's build the exact reverse path traced by Trailer 2 rear axle
        t2_x = np.array([h['t2_axle'][0] for h in forward_history])
        t2_y = np.array([h['t2_axle'][1] for h in forward_history])

        # Time-reversed order
        rev_t2_x = t2_x[::-1]
        rev_t2_y = t2_y[::-1]
        reverse_path = Path(rev_t2_x, rev_t2_y, ds=0.8)

        tokens_list = []
        states_list = []
        classes_list = []
        rev_curr_idx = 0

        # Iterate through reversed time steps
        for i in range(15, N - 15):
            fwd_entry = forward_history[N - 1 - i]
            state: VehicleState = fwd_entry['state']
            x2, y2, th2 = fwd_entry['t2_axle']
            nominal_phi = fwd_entry['phi']

            # Monotonic 1D Frenet tracking along the curve
            rev_curr_idx, _ = reverse_path.update_progression(x2, y2, rev_curr_idx)

            # Extract 32 lookahead tokens strictly ALONG THE CURVE ahead of Trailer 2
            tokens = reverse_path.get_lookahead_tokens(
                x_t2=x2,
                y_t2=y2,
                theta_t2=th2,
                num_tokens=64,
                ds=0.8,
                current_idx=rev_curr_idx
            )

            # 1. Nominal sample
            hitch_state_nom = np.array([
                math.sin(state.beta1),
                math.cos(state.beta1),
                math.sin(state.beta2),
                math.cos(state.beta2)
            ], dtype=np.float32)

            class_nom = steer_to_class_idx(nominal_phi)

            tokens_list.append(tokens)
            states_list.append(hitch_state_nom)
            classes_list.append(class_nom)

            # 2. Multi-Regime Perturbation Recovery Funnels
            if np.random.rand() < perturb_ratio:
                # Regimes: 60% small/medium tracking noise (2°-8°), 40% large articulation stress (10°-30°)
                if np.random.rand() < 0.40:
                    # Coherent high-articulation perturbation within recoverable domain
                    sign = float(np.random.choice([-1.0, 1.0]))
                    scale_b1 = np.random.uniform(math.radians(10.0), math.radians(26.0))
                    scale_b2 = np.random.uniform(math.radians(6.0), math.radians(18.0))
                    noise_b1 = sign * scale_b1
                    # Coherent sign prevents opposite-sign geometric lock
                    noise_b2 = sign * scale_b2 * np.random.uniform(0.4, 0.9)
                else:
                    noise_scale = np.random.uniform(math.radians(2.0), math.radians(8.0))
                    noise_b1 = np.random.normal(0.0, noise_scale)
                    noise_b2 = np.random.normal(0.0, noise_scale)

                perturbed_b1 = state.beta1 + noise_b1
                perturbed_b2 = state.beta2 + noise_b2

                # Enforce physical angle limits before jackknife boundary (max 60 deg)
                limit_rad = math.radians(58.0)
                perturbed_b1 = float(np.clip(perturbed_b1, -limit_rad, limit_rad))
                perturbed_b2 = float(np.clip(perturbed_b2, -limit_rad, limit_rad))

                # LQR corrective steering
                phi_star = self.lqr.corrective_steering(
                    nominal_phi=nominal_phi,
                    nominal_beta1=state.beta1,
                    nominal_beta2=state.beta2,
                    actual_beta1=perturbed_b1,
                    actual_beta2=perturbed_b2,
                    v0=-1.5
                )

                hitch_state_pert = np.array([
                    math.sin(perturbed_b1),
                    math.cos(perturbed_b1),
                    math.sin(perturbed_b2),
                    math.cos(perturbed_b2)
                ], dtype=np.float32)

                class_pert = steer_to_class_idx(phi_star)

                tokens_list.append(tokens)
                states_list.append(hitch_state_pert)
                classes_list.append(class_pert)

        return (
            np.array(tokens_list, dtype=np.float32),
            np.array(states_list, dtype=np.float32),
            np.array(classes_list, dtype=np.int64)
        )

    def generate_dataset(
        self,
        num_trajectories: int = 160,
        perturb_ratio: float = 0.55
    ) -> ReversingDataset:
        """
        Generates full synthetic dataset spanning diverse user-like and geometric path types.
        """
        path_types = [
            "random_fourier",
            "yard_grand_tour",
            "yard_serpentine_slalom",
            "yard_double_figure8",
            "yard_perimeter_clover",
            "self_intersecting_figure8",
            "loop_the_loop",
            "sharp_switchbacks",
            "cloverleaf_intersect",
            "serpentine_knot",
            "dock_with_360_turnaround",
            "high_curvature_chicane",
            "continuous_arc",
            "figure8_inflection",
            "multi_scale_s",
            "segmented_dock",
            "jagged_corners",
            "compound_kink",
            "user_wobbly",
            "rapid_chicane",
            "asymmetric_bezier",
            "piecewise_clothoid",
            "z_track",
            "parallel_lane_shift",
            "hairpin",
            "s_curve",
            "straight"
        ]
        all_tokens = []
        all_states = []
        all_classes = []

        for i in range(num_trajectories):
            ptype = str(np.random.choice(path_types))
            # Diverse path lengths: 65m for short maneuvers up to 320m for giant yard-filling trajectories
            if ptype in ["yard_grand_tour", "yard_double_figure8", "yard_perimeter_clover"]:
                length = np.random.uniform(220.0, 320.0)
            elif ptype in ["yard_serpentine_slalom", "self_intersecting_figure8", "cloverleaf_intersect", "serpentine_knot", "random_fourier"]:
                length = np.random.uniform(140.0, 240.0)
            elif ptype in ["loop_the_loop", "dock_with_360_turnaround", "figure8_inflection", "continuous_arc"]:
                length = np.random.uniform(110.0, 165.0)
            elif ptype in ["multi_scale_s", "segmented_dock", "sharp_switchbacks", "z_track", "parallel_lane_shift"]:
                length = np.random.uniform(85.0, 130.0)
            else:
                length = np.random.uniform(65.0, 100.0)
            hist = self.generate_single_trajectory(path_type=ptype, length_m=length)
            toks, sts, cls = self.extract_reversing_samples(hist, perturb_ratio=perturb_ratio)

            if len(cls) > 0:
                all_tokens.append(toks)
                all_states.append(sts)
                all_classes.append(cls)

        merged_tokens = np.concatenate(all_tokens, axis=0)
        merged_states = np.concatenate(all_states, axis=0)
        merged_classes = np.concatenate(all_classes, axis=0)

        return ReversingDataset(merged_tokens, merged_states, merged_classes)
