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

    def __len__(self):
        return len(self.steer_targets)

    def __getitem__(self, idx):
        return self.tokens[idx], self.hitch_states[idx], self.steer_targets[idx]


def compute_flatness_states(path: Path, dims: VehicleDimensions) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluates the exact analytical Differential Flatness equations for a B-Double
    multi-trailer vehicle where Trailer 2's rear axle (x2, y2) is the flat output.

    For any C^2 reference curve parameterized by arc length s:
      1. tan(beta2) = L2 * kappa2
      2. kappa1 = cos(beta2) * [kappa2 + dbeta2/ds2]
         tan(beta1) = L1 * kappa1
      3. kappa0 = cos(beta1) * [kappa1 + dbeta1/ds1]
         tan(phi)   = L0 * kappa0

    Returns:
      beta1 (np.ndarray): Trailer 1 hitch angles along curve (rad)
      beta2 (np.ndarray): Trailer 2 hitch angles along curve (rad)
      phi   (np.ndarray): Tractor steering angles along curve (rad)
    """
    ds = path.ds
    N = len(path.x)
    s2 = np.arange(N, dtype=np.float64) * ds
    k2 = path.curvatures.astype(np.float64)
    dk2_ds2 = np.gradient(k2, s2)

    # 1. Trailer 2 -> Trailer 1 Flatness Map
    tan_b2 = dims.L2 * k2
    b2 = np.arctan(tan_b2)
    cos_b2 = np.cos(b2)
    db2_ds2 = (dims.L2 * dk2_ds2) / (1.0 + tan_b2**2)

    # 2. Trailer 1 -> Tractor Axle Flatness Map
    k1 = cos_b2 * (k2 + db2_ds2)
    tan_b1 = dims.L1 * k1
    b1 = np.arctan(tan_b1)
    cos_b1 = np.cos(b1)

    s1 = np.cumsum(np.maximum(0.1, cos_b2) * ds)
    dk1_ds1 = np.gradient(k1, s1)
    db1_ds1 = (dims.L1 * dk1_ds1) / (1.0 + tan_b1**2)

    # 3. Tractor Axle -> Front Steer Wheel Flatness Map
    k0 = cos_b1 * (k1 + db1_ds1)
    tan_phi = dims.L0 * k0

    phi = np.clip(np.arctan(tan_phi), -dims.max_steer_rad, dims.max_steer_rad).astype(np.float32)
    b1 = np.clip(b1, -dims.crit_hitch_rad * 0.95, dims.crit_hitch_rad * 0.95).astype(np.float32)
    b2 = np.clip(b2, -dims.crit_hitch_rad * 0.95, dims.crit_hitch_rad * 0.95).astype(np.float32)

    return b1, b2, phi


def create_recovery_path(
    base_path: Path,
    lat_offset: float = 1.5,
    heading_offset: float = 0.15,
    transition_len: float = 20.0,
    ds: float = 0.4
) -> Path:
    """
    Generates a C^2 smooth quintic polynomial recovery transition that smoothly
    connects an off-nominal pose (x_start, y_start, theta_start) back onto the
    reference centerline path without any geometric kinks.
    """
    x0, y0 = float(base_path.x[0]), float(base_path.y[0])
    th0 = float(base_path.headings[0])
    nx = -math.sin(th0)
    ny = math.cos(th0)

    num_trans = max(10, int(transition_len / ds))
    t = np.linspace(0.0, 1.0, num_trans, dtype=np.float64)
    # Quintic polynomial: h(0)=1, h(1)=0, h'(0)=h'(1)=0, h''(0)=h''(1)=0
    poly = 1.0 - (10.0 * t**3 - 15.0 * t**4 + 6.0 * t**5)

    trans_x: List[float] = []
    trans_y: List[float] = []
    for k in range(num_trans):
        prog = float(t[k]) * transition_len
        bx = x0 + prog * math.cos(th0)
        by = y0 + prog * math.sin(th0)
        curr_offset = lat_offset * float(poly[k])
        trans_x.append(bx + curr_offset * nx)
        trans_y.append(by + curr_offset * ny)

    start_idx = min(len(base_path.x) - 1, int(transition_len / base_path.ds))
    full_x = np.concatenate([np.array(trans_x, dtype=np.float64), base_path.x[start_idx:]])
    full_y = np.concatenate([np.array(trans_y, dtype=np.float64), base_path.y[start_idx:]])
    return Path(full_x, full_y, ds=ds)


class SyntheticDataGenerator:
    """
    Generates diverse, mathematically exact multi-trailer reversing demonstrations
    using Differential Flatness. Eliminates all heuristic LQR linearizations,
    artificial noise injection, and backward-in-time drift paradoxes.
    """
    def __init__(self, dims: Optional[VehicleDimensions] = None):
        self.dims = dims or VehicleDimensions()
        self.kinematics = TractorDoubleTrailerKinematics(self.dims)

    def extract_flatness_samples(
        self,
        base_path: Path,
        actual_path: Optional[Path] = None,
        sigma_deg: float = 4.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extracts exact (tokens, hitch_state, steer_target) samples along actual_path
        with reference to base_path.
        """
        tracked_path = actual_path if actual_path is not None else base_path
        N = len(tracked_path.x)
        if N < 35:
            return (
                np.zeros((0, 64, 7), dtype=np.float32),
                np.zeros((0, 4), dtype=np.float32),
                np.zeros((0, 15), dtype=np.float32)
            )

        b1, b2, phi = compute_flatness_states(tracked_path, self.dims)

        # In reverse, Trailer 2 body heading points forward towards Trailer 1: th2 = th_path - pi
        th2_arr = tracked_path.headings - math.pi

        tokens_list = []
        states_list = []
        targets_list = []
        base_curr_idx = 0

        # Sample interior points to avoid boundary gradient edge effects
        for i in range(4, N - 12):
            x2 = float(tracked_path.x[i])
            y2 = float(tracked_path.y[i])
            th2 = float(th2_arr[i])

            base_curr_idx, _ = base_path.update_progression(x2, y2, base_curr_idx)

            # Lookahead tokens ahead of Trailer 2 in Trailer 2's body frame
            tokens = base_path.get_lookahead_tokens(
                x_t2=x2,
                y_t2=y2,
                theta_t2=th2,
                num_tokens=64,
                ds=0.8,
                current_idx=base_curr_idx
            )

            hitch_state = np.array([
                math.sin(float(b1[i])),
                math.cos(float(b1[i])),
                math.sin(float(b2[i])),
                math.cos(float(b2[i]))
            ], dtype=np.float32)

            soft_target = steer_to_soft_target(float(phi[i]), sigma_deg=sigma_deg)

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
        num_trajectories: int = 120,
        length_m: float = 80.0,
        sigma_deg: float = 4.0,
        **kwargs
    ) -> ReversingDataset:
        """
        Generates full synthetic dataset by sampling diverse procedural paths
        and computing closed-form Differential Flatness solutions.
        Includes 50% nominal path following and 50% offset recovery trajectories.
        """
        rng = np.random.default_rng()
        all_tokens = []
        all_states = []
        all_targets = []

        for _ in range(num_trajectories):
            ptype = str(rng.choice(PATH_TYPES))
            base_path = generate_synthetic_path(
                length_m=length_m,
                ds=0.4,
                path_type=ptype,
                allow_rotation=True
            )
            if len(base_path.x) < 35:
                continue

            # 50% nominal path following, 50% offset recovery
            if rng.random() < 0.50:
                lat_offset = float(rng.uniform(-2.2, 2.2))
                heading_offset = float(rng.uniform(-0.30, 0.30))  # ±17 degrees
                trans_len = float(rng.uniform(16.0, 26.0))
                actual_path = create_recovery_path(
                    base_path=base_path,
                    lat_offset=lat_offset,
                    heading_offset=heading_offset,
                    transition_len=trans_len,
                    ds=0.4
                )
            else:
                actual_path = None

            toks, states, tgts = self.extract_flatness_samples(
                base_path=base_path,
                actual_path=actual_path,
                sigma_deg=sigma_deg
            )

            if len(toks) > 0:
                all_tokens.append(toks)
                all_states.append(states)
                all_targets.append(tgts)

        if not all_tokens:
            raise RuntimeError("Failed to generate any valid training trajectories.")

        merged_tokens = np.concatenate(all_tokens, axis=0)
        merged_states = np.concatenate(all_states, axis=0)
        merged_targets = np.concatenate(all_targets, axis=0)

        return ReversingDataset(merged_tokens, merged_states, merged_targets)
