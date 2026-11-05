import math
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Dict, Any, Optional

from src.kinematics import TractorDoubleTrailerKinematics, VehicleDimensions, VehicleState
from src.path_generator import Path, generate_synthetic_path
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
    Pure-Path / No-Teacher Synthetic Data Generator for Multi-Trailer Reversing.
    
    Eliminates heuristic teacher controllers (LQR, cascade clamps, hand-tuned gains).
    Uses 100% physically exact kinematic time-reversal:
      - Nominal Trajectories: forward rollouts on 27 path families time-reversed for
        exact centerline following and high-curvature holding (up to beta2 ~ 45 deg).
      - Merging Trajectories: forward rollouts on feeder paths peeling off from main roads.
        In time-reversal, Trailer 2 starts off-track (y_rel != 0, dtheta != 0) and
        naturally executes the non-linear multi-body counter-steer maneuver that merges
        onto the main target road with zero terminal error.
    """
    def __init__(self, dims: Optional[VehicleDimensions] = None):
        self.dims = dims or VehicleDimensions()
        self.kinematics = TractorDoubleTrailerKinematics(self.dims)

    def build_feeder_and_main_paths(
        self,
        length_m: float = 80.0,
        ds: float = 0.8,
        path_type: str = "s_curve",
        merge_type: str = "nominal",
        max_offset: float = 1.5
    ) -> Tuple[Path, Path]:
        """
        Builds a main target path and an associated feeder rollout path.
        """
        main_path = generate_synthetic_path(length_m=length_m, ds=ds, path_type=path_type)
        N = len(main_path.x)
        if N < 25 or merge_type == "nominal":
            return main_path, main_path

        offset_profile = np.zeros(N)

        if merge_type == "peel_off":
            # Forward starts on path, peels off to max_offset at end.
            # In reverse: Trailer 2 starts off-track by max_offset, smoothly merges to path!
            split_idx = int(N * np.random.uniform(0.15, 0.40))
            ramp_len = N - split_idx
            s_norm = np.linspace(0.0, 1.0, ramp_len)
            smooth_ramp = (3.0 * (s_norm**2) - 2.0 * (s_norm**3)) * max_offset
            offset_profile[split_idx:] = smooth_ramp

        elif merge_type == "lane_change":
            # Forward transitions from -max_offset/2 across 0 to +max_offset/2.
            # In reverse: transitions across the main road centerline.
            s_norm = np.linspace(0.0, 1.0, N)
            smooth_ramp = (3.0 * (s_norm**2) - 2.0 * (s_norm**3)) * max_offset - (max_offset * 0.5)
            offset_profile[:] = smooth_ramp

        elif merge_type == "angled_merge":
            # Forward starts offset and merges onto main path.
            # In reverse: Trailer 2 starts on main path and diverges onto offset angled track.
            split_idx = int(N * np.random.uniform(0.30, 0.60))
            s_norm = np.linspace(0.0, 1.0, split_idx)
            smooth_ramp = (1.0 - (3.0 * (s_norm**2) - 2.0 * (s_norm**3))) * max_offset
            offset_profile[:split_idx] = smooth_ramp
            offset_profile[split_idx:] = 0.0

        normals_x = -np.sin(main_path.headings)
        normals_y = np.cos(main_path.headings)
        feeder_x = main_path.x + offset_profile * normals_x
        feeder_y = main_path.y + offset_profile * normals_y

        feeder_path = Path(feeder_x, feeder_y, ds=ds)
        return main_path, feeder_path

    def generate_single_trajectory(
        self,
        path_type: str = "smooth_random",
        length_m: float = 75.0,
        dt: float = 0.05,
        v_forward: float = 1.5,
        merge_type: str = "nominal",
        max_offset: float = 1.5
    ) -> Tuple[List[Dict[str, Any]], Path]:
        """
        Rolls out forward trajectory following feeder_path, recording kinematically exact states.
        Returns the recorded history along with the main target path.
        """
        main_path, feeder_path = self.build_feeder_and_main_paths(
            length_m=length_m,
            ds=0.8,
            path_type=path_type,
            merge_type=merge_type,
            max_offset=max_offset
        )
        if len(feeder_path.x) < 20:
            return [], main_path

        rng = np.random.default_rng()
        b1_init = float(rng.uniform(-0.06, 0.06)) if rng.random() < 0.5 else 0.0
        b2_init = float(rng.uniform(-0.06, 0.06)) if rng.random() < 0.5 else 0.0
        state = VehicleState(
            x0=feeder_path.x[0],
            y0=feeder_path.y[0],
            theta0=feeder_path.headings[0],
            beta1=b1_init,
            beta2=b2_init,
            phi=0.0
        )

        steps = int((feeder_path.total_length - 12.0) / (v_forward * dt))
        history: List[Dict[str, Any]] = []

        lookahead_dist = 4.5
        fwd_nearest_idx = 0

        for _ in range(steps):
            cos_t0, sin_t0 = math.cos(state.theta0), math.sin(state.theta0)
            fx = state.x0 + self.dims.L0 * cos_t0
            fy = state.y0 + self.dims.L0 * sin_t0

            fwd_nearest_idx, _ = feeder_path.update_progression(fx, fy, fwd_nearest_idx)
            target_idx = min(len(feeder_path.x) - 1, fwd_nearest_idx + int(lookahead_dist / feeder_path.ds))
            gx = feeder_path.x[target_idx]
            gy = feeder_path.y[target_idx]

            dx = gx - fx
            dy = gy - fy
            local_x = cos_t0 * dx + sin_t0 * dy
            local_y = -sin_t0 * dx + cos_t0 * dy

            dist_to_goal = math.hypot(local_x, local_y)
            if dist_to_goal < 0.5:
                alpha = feeder_path.headings[target_idx] - state.theta0
                alpha = math.atan2(math.sin(alpha), math.cos(alpha))
                target_phi = math.atan2(2.0 * self.dims.L0 * math.sin(alpha), lookahead_dist)
            else:
                target_phi = math.atan2(2.0 * self.dims.L0 * local_y, dist_to_goal**2)

            target_phi = max(-self.dims.max_steer_rad, min(self.dims.max_steer_rad, target_phi))

            x2, y2, th2 = self.kinematics.get_trailer2_axle(state)

            history.append({
                'state': VehicleState(state.x0, state.y0, state.theta0, state.beta1, state.beta2, state.phi),
                't2_axle': (x2, y2, th2),
                'phi': state.phi,
                'target_phi': target_phi
            })

            state = self.kinematics.step_rk4(state, v0=v_forward, target_phi=target_phi, dt=dt)
            if self.kinematics.is_jackknifed(state) or abs(state.beta1) > math.radians(55.0) or abs(state.beta2) > math.radians(55.0):
                break

        return history, main_path

    def extract_reversing_samples(
        self,
        forward_history: List[Dict[str, Any]],
        main_path: Path
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Applies time-reversal to forward history and extracts (tokens, hitch_state, steer_class).
        Lookahead tokens are evaluated against the reversed main target path, naturally
        teaching both centerline tracking and off-track re-centering without a teacher.
        """
        N = len(forward_history)
        if N < 50:
            return np.zeros((0, 64, 7), dtype=np.float32), np.zeros((0, 4), dtype=np.float32), np.zeros((0, 15), dtype=np.float32)

        # Track progression along main_path throughout forward rollout to find true terminal station
        m_idx = 0
        for entry in forward_history:
            x2, y2, _ = entry['t2_axle']
            m_idx, _ = main_path.update_progression(x2, y2, m_idx)

        slice_end = min(len(main_path.x), m_idx + 8)
        rev_x = main_path.x[:slice_end][::-1]
        rev_y = main_path.y[:slice_end][::-1]
        rev_main_path = Path(rev_x, rev_y, ds=0.8)

        tokens_list = []
        states_list = []
        classes_list = []
        rev_curr_idx = 0

        for i in range(12, N - 12):
            fwd_entry = forward_history[N - 1 - i]
            state: VehicleState = fwd_entry['state']
            x2, y2, th2 = fwd_entry['t2_axle']
            phi = fwd_entry['phi']

            rev_curr_idx, _ = rev_main_path.update_progression(x2, y2, rev_curr_idx)

            tokens = rev_main_path.get_lookahead_tokens(
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

            steer_target = steer_to_soft_target(phi)

            tokens_list.append(tokens)
            states_list.append(hitch_state)
            classes_list.append(steer_target)

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
        perturb_ratio: float = 0.60
    ) -> ReversingDataset:
        """
        Generates full synthetic dataset spanning diverse path types and merge geometries.
        perturb_ratio sets the fraction of merging/re-centering trajectories (vs nominal).
        """
        all_tokens = []
        all_states = []
        all_classes = []

        merge_types = ["peel_off", "lane_change", "angled_merge"]
        merge_weights = [0.60, 0.25, 0.15]

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

            is_merge = (np.random.rand() < perturb_ratio)
            if is_merge:
                mtype = str(np.random.choice(merge_types, p=merge_weights))
                max_offset = float(np.random.uniform(0.5, 2.2) * np.random.choice([-1.0, 1.0]))
            else:
                mtype = "nominal"
                max_offset = 0.0

            hist, main_p = self.generate_single_trajectory(
                path_type=ptype,
                length_m=length,
                merge_type=mtype,
                max_offset=max_offset
            )
            toks, sts, cls = self.extract_reversing_samples(hist, main_p)

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
