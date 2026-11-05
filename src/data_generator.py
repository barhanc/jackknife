import math
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Dict, Any, Optional

from src.kinematics import TractorDoubleTrailerKinematics, VehicleDimensions, VehicleState
from src.path_generator import Path, generate_synthetic_path
from src.model import steer_to_soft_target, NUM_STEER_CLASSES

PATH_TYPES = [
    "chicane",
    "rapid_chicane",
    "segmented_dock",
    "parallel_lane_shift",
    "multi_scale_s",
    "sharp_switchbacks",
    "yard_serpentine_slalom",
    "random_fourier",
    "smooth_random"
]

class ReversingDataset(Dataset):
    """
    In-memory PyTorch Dataset for Multi-Trailer Reversing RL-PFN.
    Contains:
      - lookahead_tokens: (N, 64, 7) float32
      - hitch_states: (N, 4) float32
      - steer_targets: (N, 15) float32 Gaussian soft target distributions
    """
    def __init__(self, tokens: np.ndarray, hitch_states: np.ndarray, steer_classes: np.ndarray):
        self.tokens = torch.from_numpy(tokens).float()
        self.hitch_states = torch.from_numpy(hitch_states).float()
        self.steer_targets = torch.from_numpy(steer_classes).float() if steer_classes.ndim == 2 else torch.from_numpy(steer_classes).long()

    def __len__(self):
        return len(self.steer_targets)

    def __getitem__(self, idx):
        return self.tokens[idx], self.hitch_states[idx], self.steer_targets[idx]


class SyntheticDataGenerator:
    """
    Generates diverse, dynamically exact multi-trailer reversing demonstrations
    via closed-form time-reversal of natural non-holonomic trajectories.
    Eliminates all heuristic LQR linearizations and synthetic noise corruption.
    """
    def __init__(self, dims: Optional[VehicleDimensions] = None):
        self.dims = dims or VehicleDimensions()
        self.kinematics = TractorDoubleTrailerKinematics(self.dims)

    def generate_single_trajectory(
        self,
        path_type: Optional[str] = None,
        length_m: float = 80.0,
        dt: float = 0.05,
        v_forward: float = 1.5
    ) -> List[Dict[str, Any]]:
        """
        Rolls out an exact forward trajectory along a procedurally sampled path,
        incorporating natural entry offsets and recovery convergence.
        """
        rng = np.random.default_rng()
        if path_type is None:
            path_type = str(rng.choice(PATH_TYPES))

        path = generate_synthetic_path(length_m=length_m, ds=0.8, path_type=path_type)
        if len(path.x) < 25:
            return []

        # 40% nominal path alignment, 60% diverse offset convergence
        if rng.random() < 0.60:
            lat_offset = float(rng.uniform(-1.8, 1.8))
            heading_offset = float(rng.uniform(-0.35, 0.35))  # ±20 degrees
            b1_init = float(rng.uniform(-0.25, 0.25))          # ±14 degrees
            b2_init = float(rng.uniform(-0.25, 0.25))          # ±14 degrees
        else:
            lat_offset = 0.0
            heading_offset = 0.0
            b1_init = float(rng.uniform(-0.05, 0.05)) if rng.random() < 0.5 else 0.0
            b2_init = float(rng.uniform(-0.05, 0.05)) if rng.random() < 0.5 else 0.0

        th0_init = path.headings[0] + heading_offset
        norm_x = -math.sin(path.headings[0])
        norm_y = math.cos(path.headings[0])

        x0_init = path.x[0] + lat_offset * norm_x
        y0_init = path.y[0] + lat_offset * norm_y

        state = VehicleState(
            x0=x0_init,
            y0=y0_init,
            theta0=th0_init,
            beta1=b1_init,
            beta2=b2_init,
            phi=0.0
        )

        steps = int((length_m - 12.0) / (v_forward * dt))
        history = []

        lookahead_dist = 4.5  # Pure pursuit lookahead for tractor cab
        fwd_nearest_idx = 0

        for step in range(steps):
            # Compute Pure Pursuit steering for tractor cab
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

            # Record vehicle multi-body state
            x1, y1, th1 = self.kinematics.get_trailer1_axle(state)
            x2, y2, th2 = self.kinematics.get_trailer2_axle(state)

            history.append({
                'state': VehicleState(state.x0, state.y0, state.theta0, state.beta1, state.beta2, state.phi),
                't2_axle': (x2, y2, th2),
                'phi': state.phi,
                'target_phi': target_phi
            })

            # Advance kinematics forward using RK4
            state = self.kinematics.step_rk4(state, v0=v_forward, target_phi=target_phi, dt=dt)
            if self.kinematics.is_jackknifed(state):
                break

        return history

    def extract_reversing_samples(
        self,
        forward_history: List[Dict[str, Any]],
        sigma_deg: float = 4.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Applies time-reversal to forward history and extracts exact (tokens, hitch_state, steer_target) samples.
        Each sample is an uncorrupted, kinematically exact physical state.
        """
        N = len(forward_history)
        if N < 50:
            return np.zeros((0, 64, 7), dtype=np.float32), np.zeros((0, 4), dtype=np.float32), np.zeros((0, 15), dtype=np.float32)

        # Build reverse path traced by Trailer 2 rear axle
        t2_x = np.array([h['t2_axle'][0] for h in forward_history])
        t2_y = np.array([h['t2_axle'][1] for h in forward_history])

        rev_t2_x = t2_x[::-1]
        rev_t2_y = t2_y[::-1]
        reverse_path = Path(rev_t2_x, rev_t2_y, ds=0.8)

        tokens_list = []
        states_list = []
        targets_list = []
        rev_curr_idx = 0

        # Discard start and end turnaround transients
        for i in range(12, N - 12):
            fwd_entry = forward_history[N - 1 - i]
            state: VehicleState = fwd_entry['state']
            x2, y2, th2 = fwd_entry['t2_axle']
            nominal_phi = fwd_entry['phi']

            rev_curr_idx, _ = reverse_path.update_progression(x2, y2, rev_curr_idx)

            # Lookahead tokens ahead of Trailer 2 in reverse
            tokens = reverse_path.get_lookahead_tokens(
                x_t2=x2,
                y_t2=y2,
                theta_t2=th2,
                num_tokens=64,
                ds=0.8,
                current_idx=rev_curr_idx
            )

            hitch_state = np.array([
                math.sin(state.beta1),
                math.cos(state.beta1),
                math.sin(state.beta2),
                math.cos(state.beta2)
            ], dtype=np.float32)

            soft_target = steer_to_soft_target(nominal_phi, sigma_deg=sigma_deg)

            tokens_list.append(tokens)
            states_list.append(hitch_state)
            targets_list.append(soft_target)

        return (
            np.array(tokens_list, dtype=np.float32),
            np.array(states_list, dtype=np.float32),
            np.array(targets_list, dtype=np.float32)
        )

    def generate_dataset(
        self,
        num_trajectories: int = 160,
        length_m: float = 80.0,
        **kwargs
    ) -> ReversingDataset:
        """
        Generates full synthetic dataset by sampling diverse procedural trajectories.
        """
        all_tokens = []
        all_states = []
        all_targets = []

        for _ in range(num_trajectories):
            fwd_hist = self.generate_single_trajectory(length_m=length_m)
            if len(fwd_hist) >= 50:
                toks, states, targets = self.extract_reversing_samples(fwd_hist)
                if len(toks) > 0:
                    all_tokens.append(toks)
                    all_states.append(states)
                    all_targets.append(targets)

        if not all_tokens:
            raise RuntimeError("Failed to generate any valid training trajectories.")

        merged_tokens = np.concatenate(all_tokens, axis=0)
        merged_states = np.concatenate(all_states, axis=0)
        merged_targets = np.concatenate(all_targets, axis=0)

        return ReversingDataset(merged_tokens, merged_states, merged_targets)
