import math
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Dict, Any, Optional

from src.kinematics import TractorDoubleTrailerKinematics, VehicleDimensions, VehicleState
from src.path_generator import Path, generate_synthetic_path
from src.lqr_controller import ReverseLQRController
from src.model import steer_to_soft_target

PATH_TYPES = [
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

class ReversingDataset(Dataset):
    """
    In-memory PyTorch Dataset for Multi-Trailer Reversing ControlPFN.
    Contains:
      - lookahead_tokens: (N, 64, 7) float32
      - hitch_states: (N, 4) float32
      - steer_targets: (N, 15) float32 Gaussian soft target distributions
    """
    def __init__(self, tokens: np.ndarray, hitch_states: np.ndarray, steer_classes: np.ndarray):
        self.tokens = torch.from_numpy(tokens).float()
        self.hitch_states = torch.from_numpy(hitch_states).float()
        self.steer_targets = torch.from_numpy(steer_classes).float() if steer_classes.ndim == 2 else torch.from_numpy(steer_classes).long()

    def __len__(self) -> int:
        return len(self.steer_targets)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.tokens[idx], self.hitch_states[idx], self.steer_targets[idx]


class SyntheticDataGenerator:
    """
    Generates training data via closed-form time-reversal of forward trajectories
    plus tuned LQR contraction funnel perturbations.
    """
    def __init__(self, dims: Optional[VehicleDimensions] = None):
        self.dims = dims or VehicleDimensions()
        self.kinematics = TractorDoubleTrailerKinematics(self.dims)
        # Consistent, non-conflicting optimal Riccati teacher
        self.lqr = ReverseLQRController(self.dims, q_diag=(3.5, 7.0), r_val=18.0, max_delta_phi_deg=22.0)
        self.c_y = 0.16  # Calibrated restoring stiffness (clean recovery without phase lag resonance)

    def generate_single_trajectory(
        self,
        path_type: str = "smooth_random",
        length_m: float = 75.0,
        dt: float = 0.05,
        v_forward: float = 1.5
    ) -> List[Dict[str, Any]]:
        """
        Rolls out forward trajectory following a smooth path, recording states.
        """
        path = generate_synthetic_path(length_m=length_m, ds=0.8, path_type=path_type)
        if len(path.x) < 20:
            return []

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
        history: List[Dict[str, Any]] = []

        lookahead_dist = 4.5
        fwd_nearest_idx = 0

        for _ in range(steps):
            cos_t0, sin_t0 = math.cos(state.theta0), math.sin(state.theta0)
            fx = state.x0 + self.dims.L0 * cos_t0
            fy = state.y0 + self.dims.L0 * sin_t0

            fwd_nearest_idx, _ = path.update_progression(fx, fy, fwd_nearest_idx)
            target_idx = min(len(path.x) - 1, fwd_nearest_idx + int(lookahead_dist / path.ds))
            gx = path.x[target_idx]
            gy = path.y[target_idx]

            dx = gx - fx
            dy = gy - fy
            local_x = cos_t0 * dx + sin_t0 * dy
            local_y = -sin_t0 * dx + cos_t0 * dy

            dist_to_goal = math.hypot(local_x, local_y)
            if dist_to_goal < 0.5:
                alpha = path.headings[target_idx] - state.theta0
                alpha = math.atan2(math.sin(alpha), math.cos(alpha))
                target_phi = math.atan2(2.0 * self.dims.L0 * math.sin(alpha), lookahead_dist)
            else:
                target_phi = math.atan2(2.0 * self.dims.L0 * local_y, dist_to_goal**2)

            target_phi = max(-self.dims.max_steer_rad, min(self.dims.max_steer_rad, target_phi))

            x1, y1, th1 = self.kinematics.get_trailer1_axle(state)
            x2, y2, th2 = self.kinematics.get_trailer2_axle(state)

            history.append({
                'state': VehicleState(state.x0, state.y0, state.theta0, state.beta1, state.beta2, state.phi),
                't2_axle': (x2, y2, th2),
                'phi': state.phi,
                'target_phi': target_phi
            })

            state = self.kinematics.step_rk4(state, v0=v_forward, target_phi=target_phi, dt=dt)
            if self.kinematics.is_jackknifed(state):
                break

        return history

    def extract_reversing_samples(
        self,
        forward_history: List[Dict[str, Any]],
        perturb_ratio: float = 0.5,
        lqr_controller: Optional[ReverseLQRController] = None,
        c_y: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Applies time-reversal to forward history and extracts (tokens, hitch_state, steer_class).
        Injects tuned LQR perturbation recovery funnels with optional randomized teacher persona.
        """
        N = len(forward_history)
        if N < 50:
            return np.zeros((0, 64, 7), dtype=np.float32), np.zeros((0, 4), dtype=np.float32), np.zeros((0, 15), dtype=np.float32)

        lqr = lqr_controller or self.lqr
        restoring_cy = c_y if c_y is not None else self.c_y

        t2_x = np.array([h['t2_axle'][0] for h in forward_history])
        t2_y = np.array([h['t2_axle'][1] for h in forward_history])

        rev_t2_x = t2_x[::-1]
        rev_t2_y = t2_y[::-1]
        reverse_path = Path(rev_t2_x, rev_t2_y, ds=0.8)

        tokens_list = []
        states_list = []
        classes_list = []
        rev_curr_idx = 0

        limit_rad = math.radians(52.0)

        for i in range(15, N - 15):
            fwd_entry = forward_history[N - 1 - i]
            state: VehicleState = fwd_entry['state']
            x2, y2, th2 = fwd_entry['t2_axle']
            nominal_phi = fwd_entry['phi']

            rev_curr_idx, _ = reverse_path.update_progression(x2, y2, rev_curr_idx)

            tokens = reverse_path.get_lookahead_tokens(
                x_t2=x2,
                y_t2=y2,
                theta_t2=th2,
                num_tokens=64,
                ds=0.8,
                current_idx=rev_curr_idx
            )

            # 1. Nominal on-manifold sample
            hitch_state_nom = np.array([
                math.sin(state.beta1),
                math.cos(state.beta1),
                math.sin(state.beta2),
                math.cos(state.beta2)
            ], dtype=np.float32)

            class_nom = steer_to_soft_target(nominal_phi)

            tokens_list.append(tokens)
            states_list.append(hitch_state_nom)
            classes_list.append(class_nom)

            # 2. Articulation Perturbations (Stabilized Contraction Funnels)
            if np.random.rand() < perturb_ratio:
                # Sub-regime A (40%): Opposite-sign inflection stress (S-curves / chicane reversals)
                # Sub-regime B (60%): Normal articulation tracking jitter
                if np.random.rand() < 0.40:
                    sign1 = float(np.random.choice([-1.0, 1.0]))
                    sign2 = -sign1 # opposite sign!
                    scale_b1 = float(np.random.uniform(math.radians(8.0), math.radians(18.0)))
                    scale_b2 = float(np.random.uniform(math.radians(8.0), math.radians(16.0)))
                    noise_b1 = sign1 * scale_b1
                    noise_b2 = sign2 * scale_b2
                else:
                    noise_scale = float(np.random.uniform(math.radians(2.0), math.radians(7.0)))
                    noise_b1 = float(np.random.normal(0.0, noise_scale))
                    noise_b2 = float(np.random.normal(0.0, noise_scale))

                perturbed_b1 = state.beta1 + noise_b1
                perturbed_b2 = state.beta2 + noise_b2

                # Enforce physical angle limits safely before critical threshold
                limit_rad = math.radians(52.0)
                perturbed_b1 = float(np.clip(perturbed_b1, -limit_rad, limit_rad))
                perturbed_b2 = float(np.clip(perturbed_b2, -limit_rad, limit_rad))

                # Stabilized LQR corrective steering
                phi_star = lqr.corrective_steering(
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

                class_pert = steer_to_soft_target(phi_star)

                tokens_list.append(tokens)
                states_list.append(hitch_state_pert)
                classes_list.append(class_pert)

            # 3. Lateral Cross-Track & Heading Offset Perturbations (Path Re-centering Funnels)
            # Gives the transformer tokens with non-zero y_rel and realistic off-track poses!
            if np.random.rand() < 0.45:
                dy_local = float(np.random.uniform(-1.2, 1.2))
                dth_local = float(np.random.uniform(-math.radians(10.0), math.radians(10.0)))

                x2_pert = x2 - dy_local * math.sin(th2)
                y2_pert = y2 + dy_local * math.cos(th2)
                th2_pert = th2 + dth_local

                tokens_off = reverse_path.get_lookahead_tokens(
                    x_t2=x2_pert,
                    y_t2=y2_pert,
                    theta_t2=th2_pert,
                    num_tokens=64,
                    ds=0.8,
                    current_idx=rev_curr_idx
                )

                # Corrective curvature: steer trailer 2 towards centerline
                y_rel_off = float(tokens_off[0, 1])
                delta_b2_des = float(np.clip(restoring_cy * y_rel_off, -math.radians(35.0), math.radians(35.0)))

                b2_target = state.beta2 + delta_b2_des
                b1_target = state.beta1 + (self.dims.L1 / self.dims.L2) * delta_b2_des
                phi_target = nominal_phi - (self.dims.L0 / self.dims.L2) * delta_b2_des

                # Add small tracking noise to actual state
                act_b1 = float(state.beta1 + np.random.normal(0.0, math.radians(3.0)))
                act_b2 = float(state.beta2 + np.random.normal(0.0, math.radians(3.0)))
                act_b1 = float(np.clip(act_b1, -limit_rad, limit_rad))
                act_b2 = float(np.clip(act_b2, -limit_rad, limit_rad))

                phi_star_off = lqr.corrective_steering(
                    nominal_phi=phi_target,
                    nominal_beta1=b1_target,
                    nominal_beta2=b2_target,
                    actual_beta1=act_b1,
                    actual_beta2=act_b2,
                    v0=-1.5
                )

                hitch_state_off = np.array([
                    math.sin(act_b1),
                    math.cos(act_b1),
                    math.sin(act_b2),
                    math.cos(act_b2)
                ], dtype=np.float32)

                class_off = steer_to_soft_target(phi_star_off)

                tokens_list.append(tokens_off)
                states_list.append(hitch_state_off)
                classes_list.append(class_off)

        if not classes_list:
            return np.zeros((0, 64, 7), dtype=np.float32), np.zeros((0, 4), dtype=np.float32), np.zeros((0, 15), dtype=np.float32)

        return (
            np.array(tokens_list, dtype=np.float32),
            np.array(states_list, dtype=np.float32),
            np.array(classes_list, dtype=np.float32)
        )

    def generate_dataset(
        self,
        num_trajectories: int = 160,
        perturb_ratio: float = 0.55
    ) -> ReversingDataset:
        """
        Generates full synthetic dataset spanning diverse path types.
        """
        all_tokens = []
        all_states = []
        all_classes = []

        for _ in range(num_trajectories):
            ptype = str(np.random.choice(PATH_TYPES))
            if ptype in ["yard_grand_tour", "yard_double_figure8", "yard_perimeter_clover"]:
                length = float(np.random.uniform(220.0, 320.0))
            elif ptype in ["yard_serpentine_slalom", "self_intersecting_figure8", "cloverleaf_intersect", "serpentine_knot", "random_fourier"]:
                length = float(np.random.uniform(140.0, 240.0))
            elif ptype in ["loop_the_loop", "dock_with_360_turnaround", "figure8_inflection", "continuous_arc"]:
                length = float(np.random.uniform(110.0, 165.0))
            elif ptype in ["multi_scale_s", "segmented_dock", "sharp_switchbacks", "z_track", "parallel_lane_shift"]:
                length = float(np.random.uniform(85.0, 130.0))
            else:
                length = float(np.random.uniform(65.0, 100.0))

            hist = self.generate_single_trajectory(path_type=ptype, length_m=length)
            toks, sts, cls = self.extract_reversing_samples(
                hist,
                perturb_ratio=perturb_ratio,
                lqr_controller=self.lqr,
                c_y=self.c_y
            )

            if len(cls) > 0:
                all_tokens.append(toks)
                all_states.append(sts)
                all_classes.append(cls)

        if not all_classes:
            return ReversingDataset(np.zeros((0, 64, 7)), np.zeros((0, 4)), np.zeros((0, 15)))

        merged_tokens = np.concatenate(all_tokens, axis=0)
        merged_states = np.concatenate(all_states, axis=0)
        merged_classes = np.concatenate(all_classes, axis=0)

        return ReversingDataset(merged_tokens, merged_states, merged_classes)
