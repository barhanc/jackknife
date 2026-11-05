import math
import numpy as np
from typing import List, Tuple, Optional
from scipy.ndimage import gaussian_filter1d

MAX_PHYSICAL_CURVATURE = 0.12  # Minimum radius R >= 8.33m (Permits high-curvature turns & tight maneuvers)

class Path:
    """
    Equidistant arc-length reparameterized path with physical curvature constraints.
    Stores stations with (x, y, headings, curvatures, s).
    """
    def __init__(self, x: np.ndarray, y: np.ndarray, ds: float = 0.8, constrain_curvature: bool = True):
        self.ds = ds
        self.constrain_curvature = constrain_curvature
        if len(x) < 2:
            self.x = np.array(x, dtype=np.float64)
            self.y = np.array(y, dtype=np.float64)
            self.headings = np.zeros_like(self.x)
            self.curvatures = np.zeros_like(self.x)
            self.s = np.zeros_like(self.x)
            self.total_length = 0.0
            return

        self._build_equidistant(x, y, ds)

    def _build_equidistant(self, raw_x: np.ndarray, raw_y: np.ndarray, ds: float):
        # 1. Remove consecutive duplicates & degenerate points
        dx = np.diff(raw_x)
        dy = np.diff(raw_y)
        dist = np.hypot(dx, dy)
        keep = np.ones(len(raw_x), dtype=bool)
        keep[1:] = dist > 1e-3
        pts_x = np.array(raw_x)[keep]
        pts_y = np.array(raw_y)[keep]

        if len(pts_x) < 2:
            pts_x = np.array([raw_x[0], raw_x[0] + 1.0])
            pts_y = np.array([raw_y[0], raw_y[0]])

        # 2. Cumulative distance along raw points
        seg_dist = np.hypot(np.diff(pts_x), np.diff(pts_y))
        cum_dist = np.insert(np.cumsum(seg_dist), 0, 0.0)
        total_len = cum_dist[-1]
        self.total_length = float(total_len)

        # 3. Equidistant sample points
        num_samples = max(2, int(math.ceil(total_len / ds)) + 1)
        s_samples = np.linspace(0.0, total_len, num_samples)

        interp_x = np.interp(s_samples, cum_dist, pts_x)
        interp_y = np.interp(s_samples, cum_dist, pts_y)

        # 4. Optional Gaussian filter for raw mouse jitter removal
        if self.constrain_curvature and len(interp_x) > 6:
            interp_x = gaussian_filter1d(interp_x, sigma=1.5, mode='nearest')
            interp_y = gaussian_filter1d(interp_y, sigma=1.5, mode='nearest')

        self.x = interp_x
        self.y = interp_y
        self.s = s_samples

        # 5. Tangent headings
        dx_s = np.gradient(self.x, s_samples)
        dy_s = np.gradient(self.y, s_samples)
        self.headings = np.arctan2(dy_s, dx_s)

        # 6. Smooth curvature calculation with physical clamping
        unwrapped_th = np.unwrap(self.headings)
        dth_s = np.gradient(unwrapped_th, s_samples)

        if self.constrain_curvature:
            # Clamp to physical vehicle limits
            self.curvatures = np.clip(dth_s, -MAX_PHYSICAL_CURVATURE, MAX_PHYSICAL_CURVATURE)
        else:
            self.curvatures = np.clip(dth_s, -0.35, 0.35)

    def update_progression(self, x: float, y: float, current_idx: int) -> Tuple[int, float]:
        """
        Pure 1D parametric Frenet frame tracking along the curve.
        Computes station progression and cross-track error using strictly the 1D parameter
        s along the curve, projecting movement onto local tangent and normal vectors.
        Completely immune to self-intersections (Figure-8s, loops, coils) because it
        operates in the curve's 1D arc-length coordinate system and never performs
        2D Euclidean nearest-neighbor searches across intersecting branches!
        Returns: (new_index, cross_track_error)
        """
        N = len(self.x)
        if N < 2:
            return 0, 0.0

        idx = max(0, min(N - 1, current_idx))

        # Project position onto local tangent vector
        dx = x - self.x[idx]
        dy = y - self.y[idx]
        th = self.headings[idx]
        t_x, t_y = math.cos(th), math.sin(th)

        delta_s = dx * t_x + dy * t_y
        steps = int(round(delta_s / self.ds))

        target_idx = idx + steps
        new_idx = max(0, min(N - 1, target_idx))

        # Cross-track error is the perpendicular distance from curve at new_idx
        dx_new = x - self.x[new_idx]
        dy_new = y - self.y[new_idx]
        th_new = self.headings[new_idx]
        n_x, n_y = -math.sin(th_new), math.cos(th_new)
        lat_err = abs(dx_new * n_x + dy_new * n_y)

        # If vehicle drives forward before start of path (e.g. forward into yard):
        if target_idx < 0:
            # Longitudinal forward distance past start of path
            dx0 = x - self.x[0]
            dy0 = y - self.y[0]
            th0 = self.headings[0]
            fwd_dist = -(dx0 * math.cos(th0) + dy0 * math.sin(th0))
            # Allow up to 1.5m forward pull-up margin along road axis; beyond that, distance counts as off-course
            fwd_overshoot = max(0.0, fwd_dist - 1.5)
            cross_track = math.hypot(lat_err, fwd_overshoot * 1.5)
        else:
            cross_track = lat_err

        return new_idx, cross_track

    def find_nearest_index(
        self,
        x: float,
        y: float,
        prev_idx: Optional[int] = None,
        heading_hint: Optional[float] = None,
        window_backward: int = 5,
        window_forward: int = 8
    ) -> int:
        """
        Finds the closest station index along the curve:
        - When prev_idx is provided: searches strictly within a local progress window
          [prev_idx - window_backward, prev_idx + window_forward]. This preserves
          monotonic arc-length progression and prevents jumping between intersecting
          branches in figure-8s, loops, and compound chicanes!
        - When heading_hint is provided: penalizes candidate stations whose tangent
          heading points in the opposite direction.
        """
        N = len(self.x)
        if N < 2:
            return 0

        if prev_idx is not None and 0 <= prev_idx < N:
            start_i = max(0, prev_idx - window_backward)
            end_i = min(N, prev_idx + window_forward)
            sub_x = self.x[start_i:end_i]
            sub_y = self.y[start_i:end_i]
            dists_sq = (sub_x - x) ** 2 + (sub_y - y) ** 2

            if heading_hint is not None:
                sub_th = self.headings[start_i:end_i]
                dth = np.arctan2(np.sin(sub_th - heading_hint), np.cos(sub_th - heading_hint))
                cost = dists_sq + 30.0 * (dth ** 2)
                best_sub = int(np.argmin(cost))
            else:
                best_sub = int(np.argmin(dists_sq))
            return start_i + best_sub
        else:
            dists_sq = (self.x - x) ** 2 + (self.y - y) ** 2
            if heading_hint is not None:
                dth = np.arctan2(np.sin(self.headings - heading_hint), np.cos(self.headings - heading_hint))
                cost = dists_sq + 30.0 * (dth ** 2)
                return int(np.argmin(cost))
            return int(np.argmin(dists_sq))

    def get_lookahead_tokens(
        self,
        x_t2: float,
        y_t2: float,
        theta_t2: float,
        num_tokens: int = 64,
        ds: float = 0.8,
        corridor_w: float = 3.5,
        current_idx: Optional[int] = None
    ) -> np.ndarray:
        """
        Extracts 32 lookahead tokens (K=32, 7 dims each) in Trailer 2 local reference frame,
        strictly tracking progression ALONG THE CURVE ahead of Trailer 2.
        Tokens: [x_rel, y_rel, cos(theta_rel), sin(theta_rel), kappa, w_L, w_R]
        """
        if len(self.x) < 2:
            return np.zeros((num_tokens, 7), dtype=np.float32)

        if current_idx is None:
            # Infer expected path heading: in reverse, Trailer 2 travels backwards along path (theta_path ~ theta_t2 + pi)
            expected_th = math.atan2(math.sin(theta_t2 + math.pi), math.cos(theta_t2 + math.pi))
            nearest_idx = self.find_nearest_index(x_t2, y_t2, heading_hint=expected_th)
        else:
            nearest_idx = current_idx

        total_stations = len(self.x)

        indices = []
        for k in range(num_tokens):
            idx = nearest_idx + k
            if idx >= total_stations:
                idx = total_stations - 1
            indices.append(idx)

        stations_x = self.x[indices]
        stations_y = self.y[indices]
        stations_th = self.headings[indices]
        stations_k = self.curvatures[indices]

        dx = stations_x - x_t2
        dy = stations_y - y_t2

        cos_t = math.cos(theta_t2)
        sin_t = math.sin(theta_t2)

        x_rel = cos_t * dx + sin_t * dy
        y_rel = -sin_t * dx + cos_t * dy

        theta_rel = np.arctan2(np.sin(stations_th - theta_t2), np.cos(stations_th - theta_t2))

        tokens = np.zeros((num_tokens, 7), dtype=np.float32)
        tokens[:, 0] = x_rel
        tokens[:, 1] = y_rel
        tokens[:, 2] = np.cos(theta_rel)
        tokens[:, 3] = np.sin(theta_rel)
        tokens[:, 4] = stations_k
        tokens[:, 5] = corridor_w
        tokens[:, 6] = corridor_w

        return tokens


def smooth_and_constrain_user_path(raw_points: List[Tuple[float, float]], ds: float = 0.8, min_radius: float = 12.5) -> Path:
    """
    Takes raw freehand mouse points from user drawing and applies:
    1. Distance-based deduplication and arc-length resampling.
    2. Physical curvature relaxation: iteratively smooths path until local radius R >= min_radius (12.5m).
    3. Anchors start and end points so user's intended start/dock positions are preserved.
    """
    if len(raw_points) < 3:
        raw_x = np.array([p[0] for p in raw_points])
        raw_y = np.array([p[1] for p in raw_points])
        return Path(raw_x, raw_y, ds=ds)

    pts = np.array(raw_points, dtype=np.float64)

    # 1. Subsample closely spaced points
    d = np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1]))
    keep = np.ones(len(pts), dtype=bool)
    keep[1:] = d > 0.4
    pts = pts[keep]

    if len(pts) < 3:
        return Path(pts[:, 0], pts[:, 1], ds=ds)

    seg_dist = np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1]))
    cum_dist = np.insert(np.cumsum(seg_dist), 0, 0.0)
    total_len = cum_dist[-1]
    if total_len < 6.0:
        return Path(pts[:, 0], pts[:, 1], ds=ds)

    num_pts = max(5, int(math.ceil(total_len / ds)) + 1)
    s = np.linspace(0.0, total_len, num_pts)
    px = np.interp(s, cum_dist, pts[:, 0])
    py = np.interp(s, cum_dist, pts[:, 1])

    p0 = np.array([px[0], py[0]])
    p_end = np.array([px[-1], py[-1]])

    kappa_max = 1.0 / max(5.0, min_radius)
    cur_x, cur_y = px, py
    sigma = 1.5
    for _ in range(35):
        sx = gaussian_filter1d(cur_x, sigma=sigma, mode='nearest')
        sy = gaussian_filter1d(cur_y, sigma=sigma, mode='nearest')
        sx[0], sy[0] = p0
        sx[-1], sy[-1] = p_end

        dx = np.gradient(sx, s)
        dy = np.gradient(sy, s)
        th = np.arctan2(dy, dx)
        curv = np.gradient(np.unwrap(th), s)
        cur_x, cur_y = sx, sy
        if np.max(np.abs(curv)) <= kappa_max:
            break
        sigma += 0.8

    return Path(cur_x, cur_y, ds=ds, constrain_curvature=True)


def generate_synthetic_path(
    length_m: float = 80.0,
    ds: float = 0.8,
    max_curvature: float = MAX_PHYSICAL_CURVATURE,
    path_type: str = "smooth_random",
    allow_rotation: bool = True
) -> Path:
    """
    Generates diverse, continuously non-repeating synthetic paths across a wide spectrum of maneuvers:
    - 'random_fourier': Infinite organic paths generated by multi-harmonic Fourier frequency decomposition.
    - 'yard_grand_tour': Procedural tour traversing 8-14 yard sectors with chicanes and interior doglegs.
    - 'yard_serpentine_slalom': Procedural multi-gate slalom with variable gates, spacing, and lateral offsets.
    - 'yard_double_figure8': Dual harmonic Lissajous figure-8 with continuous aspect ratio and modulation.
    - 'yard_perimeter_clover': Multi-intersection cloverleaf with randomized harmonic parameters.
    - 'self_intersecting_figure8': Continuous Lissajous Lemniscate crossing over itself with randomized leads.
    - 'loop_the_loop': Circular 360+ degree loop crossing its own approach road with random radius and sweep.
    - 'sharp_switchbacks': Multi-hairpin mountain pass maneuver with individual turn radii and straights.
    - 'cloverleaf_intersect': Multi-loop intersecting ribbon with continuous frequencies and phases.
    - 'serpentine_knot': Multi-harmonic Lissajous ribbon knot with randomized frequencies.
    - 'dock_with_360_turnaround': 360-degree loop turnaround into a 90-degree dock approach.
    - 'high_curvature_chicane': Aggressive near-limit double kink (R ~ 10m).
    - 'continuous_arc': Sustained circular arc / roundabout (R in [12m, 48m], sweep 60-300 deg).
    - 'figure8_inflection': Opposite direction loops with transition straight for hitch angle reversal.
    - 'multi_scale_s': Multi-scale S-curves spanning tight chicanes to sweeping lane changes.
    - 'segmented_dock': Realistic alley dock with straight approach, dogleg turn, and bay alignment.
    - 'rapid_chicane': Sudden obstacle avoidance jog.
    - 'jagged_corners': Sharp piecewise straight industrial yard roads with abrupt turns.
    - 'compound_kink': Lateral jog around yard obstacles.
    - 'user_wobbly': Simulates human hand jitter and slight wobbly freehand strokes.
    - 'asymmetric_bezier': Higher-order Bezier curve with randomized control points.
    - 'piecewise_clothoid': Trapezoidal curvature profile with random piecewise segments.
    - 'hairpin': Variable loading bay hairpin with continuous turn angle and radius.
    - 'straight': Straight docking bay with baseline noise and drift.
    """
    num_pts = int(length_m / ds) + 1
    s = np.linspace(0.0, length_m, num_pts)
    rng = np.random.default_rng()

    if path_type == "straight":
        # Straight line with subtle randomized steering drift
        drift_amp = rng.uniform(-0.006, 0.006)
        curv = drift_amp * np.sin(np.linspace(0, rng.uniform(1.0, 3.0) * math.pi, num_pts))
        th = np.cumsum(curv * ds)
        x = np.cumsum(np.cos(th) * ds)
        y = np.cumsum(np.sin(th) * ds)

    elif path_type == "z_track":
        # Diagonal Z-chicane dogleg lane shift (diagonal traverse with S/Z offset jog)
        span_diag = rng.uniform(48.0, 64.0)
        slope = rng.uniform(0.35, 0.55)
        offset_w = rng.uniform(10.0, 18.0)
        pts = [
            (span_diag, span_diag * slope + offset_w),
            (span_diag * 0.65, span_diag * 0.65 * slope + offset_w),
            (span_diag * 0.35, span_diag * 0.35 * slope + offset_w),
            (span_diag * 0.10, span_diag * 0.10 * slope + offset_w * 0.6),
            (-span_diag * 0.10, -span_diag * 0.10 * slope),
            (-span_diag * 0.35, -span_diag * 0.35 * slope),
            (-span_diag * 0.65, -span_diag * 0.65 * slope),
            (-span_diag, -span_diag * slope)
        ]
        path_obj = smooth_and_constrain_user_path(pts, ds=ds, min_radius=12.5)
        x, y = path_obj.x, path_obj.y

    elif path_type == "parallel_lane_shift":
        # Horizontal / parallel lane change with sharp S-jog chicane transition
        lane_dist = rng.uniform(35.0, 55.0)
        lateral_offset = rng.uniform(7.0, 15.0) * float(rng.choice([-1.0, 1.0]))
        jog_dx = rng.uniform(16.0, 26.0)
        pts = [
            (lane_dist, lateral_offset),
            (jog_dx * 0.6, lateral_offset),
            (jog_dx * 0.2, lateral_offset * 0.8),
            (-jog_dx * 0.2, 0.0),
            (-jog_dx * 0.6, 0.0),
            (-lane_dist, 0.0)
        ]
        path_obj = smooth_and_constrain_user_path(pts, ds=ds, min_radius=12.5)
        x, y = path_obj.x, path_obj.y

    elif path_type == "random_fourier":
        # Infinite non-repeating curves generated by random Fourier harmonics
        num_harmonics = rng.integers(2, 6)
        curv = np.zeros_like(s)
        for h in range(1, num_harmonics + 1):
            freq = rng.uniform(0.015, 0.075)
            amp = rng.uniform(0.015, 0.045) / h
            phase = rng.uniform(0.0, 2.0 * math.pi)
            curv += amp * np.sin(freq * s + phase)
        curv = np.clip(curv, -max_curvature, max_curvature)
        th = np.cumsum(curv * ds)
        x = np.cumsum(np.cos(th) * ds)
        y = np.cumsum(np.sin(th) * ds)

    elif path_type == "continuous_arc":
        # Sustained circular arc / roundabout (R in [12m, 48m], sweep 60 to 300 degrees)
        target_r = rng.uniform(12.0, 48.0)
        target_k = float(rng.choice([-1.0, 1.0])) * (1.0 / target_r)
        lead_in = int(round(rng.uniform(8.0, 18.0) / ds))
        ramp_len = int(round(rng.uniform(10.0, 20.0) / ds))
        lead_out = int(round(rng.uniform(8.0, 18.0) / ds))
        hold_len = max(10, num_pts - lead_in - lead_out - 2 * ramp_len)
        pieces = [
            np.zeros(lead_in),
            np.linspace(0.0, target_k, ramp_len),
            np.full(hold_len, target_k),
            np.linspace(target_k, 0.0, ramp_len),
            np.zeros(lead_out)
        ]
        all_k = np.concatenate(pieces)
        curv = all_k[:num_pts] if len(all_k) >= num_pts else np.pad(all_k, (0, num_pts - len(all_k)))
        th = np.cumsum(curv * ds)
        x = np.cumsum(np.cos(th) * ds)
        y = np.cumsum(np.sin(th) * ds)

    elif path_type == "figure8_inflection":
        # Opposite direction turns separated by transition straight with independent radii
        r1 = rng.uniform(18.0, 44.0)
        r2 = rng.uniform(18.0, 44.0)
        k1 = 1.0 / r1
        k2 = -1.0 / r2
        dir_sign = float(rng.choice([-1.0, 1.0]))
        n_lead = int(round(rng.uniform(10.0, 20.0) / ds))
        n_cross = int(round(rng.uniform(20.0, 45.0) / ds))
        n_ramp = int(round(rng.uniform(12.0, 20.0) / ds))
        rem = max(16, num_pts - n_lead - n_cross - 4 * n_ramp - int(round(10.0 / ds)))
        n_hold = max(8, rem // 2)
        pieces = [
            np.zeros(n_lead),
            np.linspace(0.0, dir_sign * k1, n_ramp),
            np.full(n_hold, dir_sign * k1),
            np.linspace(dir_sign * k1, 0.0, n_ramp),
            np.zeros(n_cross),
            np.linspace(0.0, dir_sign * k2, n_ramp),
            np.full(n_hold, dir_sign * k2),
            np.linspace(dir_sign * k2, 0.0, n_ramp),
            np.zeros(max(4, num_pts - (n_lead + 4 * n_ramp + 2 * n_hold + n_cross)))
        ]
        all_k = np.concatenate(pieces)
        curv = all_k[:num_pts] if len(all_k) >= num_pts else np.pad(all_k, (0, num_pts - len(all_k)))
        th = np.cumsum(curv * ds)
        x = np.cumsum(np.cos(th) * ds)
        y = np.cumsum(np.sin(th) * ds)

    elif path_type == "multi_scale_s":
        # Multi-scale S-curve with continuous wavelength and amplitude
        wavelength = rng.uniform(28.0, 130.0)
        amp = rng.uniform(3.5, 16.0)
        phase = rng.uniform(0.0, 2.0 * math.pi)
        curv = (amp * (2.0 * math.pi / wavelength)**2) * np.sin(2.0 * math.pi * s / wavelength + phase)
        curv = np.clip(curv, -max_curvature, max_curvature)
        th = np.cumsum(curv * ds)
        x = np.cumsum(np.cos(th) * ds)
        y = np.cumsum(np.sin(th) * ds)

    elif path_type == "jagged_corners":
        # Sharp multi-segment warehouse road with randomized angles and leg lengths
        num_legs = rng.integers(3, 8)
        pts = [(0.0, 0.0)]
        cur_x, cur_y = 0.0, 0.0
        cur_th = 0.0
        leg_len = length_m / num_legs
        for _ in range(num_legs):
            cur_th += rng.uniform(-math.radians(65.0), math.radians(65.0))
            cur_x += leg_len * rng.uniform(0.8, 1.2) * math.cos(cur_th)
            cur_y += leg_len * rng.uniform(0.8, 1.2) * math.sin(cur_th)
            pts.append((cur_x, cur_y))
        path_obj = smooth_and_constrain_user_path(pts, ds=ds, min_radius=10.0)
        x, y = path_obj.x, path_obj.y

    elif path_type == "compound_kink":
        # Sharp obstacle bypass with continuous lateral offset and kink width
        step_y = float(rng.choice([-1.0, 1.0])) * rng.uniform(3.0, 10.0)
        k_w = rng.uniform(14.0, 26.0)
        p_start = length_m * rng.uniform(0.2, 0.35)
        pts = [
            (0.0, 0.0),
            (p_start, 0.0),
            (p_start + k_w, step_y),
            (length_m * rng.uniform(0.7, 0.85), step_y),
            (length_m, step_y)
        ]
        path_obj = smooth_and_constrain_user_path(pts, ds=ds, min_radius=10.0)
        x, y = path_obj.x, path_obj.y

    elif path_type == "user_wobbly":
        # Continuous human drawing jitter with variable turning impulses
        pts = [(0.0, 0.0)]
        cx, cy = 0.0, 0.0
        num_segs = max(4, int(length_m / rng.uniform(7.0, 12.0)))
        th = 0.0
        for _ in range(num_segs):
            th += rng.uniform(-math.radians(40.0), math.radians(40.0))
            step_d = rng.uniform(6.0, 14.0)
            cx += step_d * math.cos(th)
            cy += step_d * math.sin(th)
            pts.append((cx, cy))
        path_obj = smooth_and_constrain_user_path(pts, ds=ds, min_radius=10.0)
        x, y = path_obj.x, path_obj.y

    elif path_type == "segmented_dock":
        # Straight -> Smooth Turn (30-135 deg) -> Dock alignment straight + optional counter-turn
        curv = np.zeros_like(s)
        turn_start = int(len(s) * rng.uniform(0.18, 0.35))
        turn_len = int(rng.uniform(16.0, 36.0) / ds)
        turn_end = min(len(s) - 8, turn_start + turn_len)
        direction = float(rng.choice([-1.0, 1.0]))
        turn_k = direction * rng.uniform(0.035, max_curvature)
        tau = np.linspace(0.0, 1.0, max(2, turn_end - turn_start))
        curv[turn_start:turn_end] = turn_k * np.sin(np.pi * tau)**2
        if rng.random() < 0.6:
            c_start = min(len(s) - 15, turn_end + int(rng.uniform(10.0, 20.0) / ds))
            c_len = int(rng.uniform(14.0, 26.0) / ds)
            c_end = min(len(s) - 4, c_start + c_len)
            c_tau = np.linspace(0.0, 1.0, max(2, c_end - c_start))
            curv[c_start:c_end] = -turn_k * rng.uniform(0.5, 0.95) * np.sin(np.pi * c_tau)**2
        th = np.cumsum(curv * ds)
        x = np.cumsum(np.cos(th) * ds)
        y = np.cumsum(np.sin(th) * ds)

    elif path_type == "rapid_chicane":
        # Sudden obstacle avoidance with continuous placement, width, and direction
        curv = np.zeros_like(s)
        chicane_center = int(len(s) * rng.uniform(0.35, 0.60))
        w = int(len(s) * rng.uniform(0.08, 0.16))
        dir_sign = float(rng.choice([-1.0, 1.0]))
        k_mag = rng.uniform(0.045, max_curvature)
        curv[chicane_center - w : chicane_center] = dir_sign * k_mag
        curv[chicane_center : chicane_center + w] = -dir_sign * k_mag
        th = np.cumsum(curv * ds)
        x = np.cumsum(np.cos(th) * ds)
        y = np.cumsum(np.sin(th) * ds)

    elif path_type == "asymmetric_bezier":
        # 5-point quintic Bezier curve with continuous lateral control point offsets
        p0 = np.array([0.0, 0.0])
        p1 = np.array([length_m * rng.uniform(0.2, 0.35), rng.uniform(-15.0, 15.0)])
        p2 = np.array([length_m * rng.uniform(0.5, 0.65), rng.uniform(-18.0, 18.0)])
        p3 = np.array([length_m * rng.uniform(0.75, 0.90), rng.uniform(-15.0, 15.0)])
        p4 = np.array([length_m, rng.uniform(-10.0, 10.0)])
        t = np.linspace(0.0, 1.0, num_pts)
        bx = (1-t)**4 * p0[0] + 4*(1-t)**3*t * p1[0] + 6*(1-t)**2*t**2 * p2[0] + 4*(1-t)*t**3 * p3[0] + t**4 * p4[0]
        by = (1-t)**4 * p0[1] + 4*(1-t)**3*t * p1[1] + 6*(1-t)**2*t**2 * p2[1] + 4*(1-t)*t**3 * p3[1] + t**4 * p4[1]
        path_obj = smooth_and_constrain_user_path(list(zip(bx, by)), ds=ds, min_radius=10.0)
        x, y = path_obj.x, path_obj.y

    elif path_type == "piecewise_clothoid":
        # Linear curvature ramps with randomized segment counts and durations
        curv = np.zeros_like(s)
        num_segments = rng.integers(3, 8)
        splits = np.sort(rng.choice(len(s) - 2, num_segments, replace=False)) + 1
        splits = np.insert(splits, [0, len(splits)], [0, len(s)])
        for i in range(len(splits) - 1):
            target_k = rng.uniform(-max_curvature, max_curvature) if i % 2 == 1 else 0.0
            idx0, idx1 = splits[i], splits[i+1]
            curv[idx0:idx1] = target_k
        curv = gaussian_filter1d(curv, sigma=rng.uniform(1.5, 3.0))
        th = np.cumsum(curv * ds)
        x = np.cumsum(np.cos(th) * ds)
        y = np.cumsum(np.sin(th) * ds)

    elif path_type == "self_intersecting_figure8":
        # Parametric Lissajous / Lemniscate figure-8 with continuous scales and leads
        a = rng.uniform(28.0, 56.0)
        b = rng.uniform(18.0, 38.0)
        phase_jitter = rng.uniform(-0.15, 0.15)
        t = np.linspace(-np.pi * rng.uniform(0.92, 0.98), np.pi * rng.uniform(0.92, 0.98), num_pts)
        loop_x = a * np.sin(t)
        loop_y = b * np.sin(2.0 * t + phase_jitter) * 0.5
        th0 = math.atan2(loop_y[1] - loop_y[0], loop_x[1] - loop_x[0])
        lead_s = np.linspace(-rng.uniform(14.0, 24.0), 0.0, max(5, int(18.0 / ds)))
        lead_x = loop_x[0] + lead_s * math.cos(th0)
        lead_y = loop_y[0] + lead_s * math.sin(th0)
        th1 = math.atan2(loop_y[-1] - loop_y[-2], loop_x[-1] - loop_x[-2])
        dock_s = np.linspace(0.0, rng.uniform(14.0, 24.0), max(5, int(18.0 / ds)))
        dock_x = loop_x[-1] + dock_s * math.cos(th1)
        dock_y = loop_y[-1] + dock_s * math.sin(th1)
        x = np.concatenate([lead_x[:-1], loop_x, dock_x[1:]])
        y = np.concatenate([lead_y[:-1], loop_y, dock_y[1:]])

    elif path_type == "loop_the_loop":
        # 360-deg circular loop with continuous radius, direction, and overlap
        r = rng.uniform(11.0, 18.5)
        dir_sign = float(rng.choice([-1.0, 1.0]))
        lead_len = rng.uniform(16.0, 26.0)
        lead_s = np.linspace(-lead_len, 0.0, max(5, int(lead_len / ds)))
        x_lead = lead_s
        y_lead = np.zeros_like(lead_s)
        overlap = rng.uniform(0.4, 1.6)
        phi = np.linspace(0.0, 2.0 * np.pi + overlap, max(15, int((2.0 * np.pi + overlap) * r / ds)))
        x_circ = r * np.sin(phi)
        y_circ = dir_sign * r * (1.0 - np.cos(phi))
        end_phi = phi[-1]
        exit_len = rng.uniform(16.0, 26.0)
        exit_s = np.linspace(0.0, exit_len, max(5, int(exit_len / ds)))
        x_exit = x_circ[-1] + exit_s * math.cos(end_phi)
        y_exit = y_circ[-1] + exit_s * dir_sign * math.sin(end_phi)
        x = np.concatenate([x_lead[:-1], x_circ, x_exit[1:]])
        y = np.concatenate([y_lead[:-1], y_circ, y_exit[1:]])

    elif path_type == "sharp_switchbacks":
        # Aggressive high-curvature hairpins with continuous turn radii and straights
        num_hairpins = rng.integers(2, 5)
        pieces = [np.zeros(max(4, int(14.0 / ds)))]
        dir_turn = float(rng.choice([-1.0, 1.0]))
        for _ in range(num_hairpins):
            r_h = rng.uniform(10.2, 14.5)
            k_h = dir_turn * (1.0 / r_h)
            arc_len = math.pi * r_h * rng.uniform(0.70, 0.98)
            pieces.append(np.full(max(4, int(arc_len / ds)), k_h))
            pieces.append(np.zeros(max(4, int(rng.uniform(12.0, 24.0) / ds))))
            dir_turn = -dir_turn
        all_k = np.concatenate(pieces)
        th = np.cumsum(all_k * ds)
        x = np.cumsum(np.cos(th) * ds)
        y = np.cumsum(np.sin(th) * ds)

    elif path_type == "cloverleaf_intersect":
        # Multi-loop cloverleaf with continuous coefficients and phase shifts
        a = rng.uniform(34.0, 52.0)
        b = rng.uniform(18.0, 32.0)
        c = rng.uniform(6.0, 14.0)
        w_t = rng.uniform(0.85, 1.15)
        t = np.linspace(-np.pi * rng.uniform(0.88, 0.96), np.pi * rng.uniform(0.88, 0.96), num_pts)
        x = a * np.sin(w_t * t)
        y = b * np.sin(2.0 * w_t * t) + c * np.sin(w_t * t)

    elif path_type == "serpentine_knot":
        # Multi-harmonic Lissajous ribbon knot with randomized frequencies
        a1, a2 = rng.uniform(28.0, 44.0), rng.uniform(8.0, 16.0)
        b1 = rng.uniform(18.0, 30.0)
        w1, w2, w3 = rng.uniform(1.2, 1.7), rng.uniform(0.6, 1.0), rng.uniform(1.8, 2.4)
        phi_k = rng.uniform(0.0, math.pi)
        t = np.linspace(-2.2, 2.2, num_pts)
        x = a1 * np.sin(t * w1) + a2 * np.cos(t * w2 + phi_k)
        y = b1 * np.cos(t * w3)

    elif path_type == "dock_with_360_turnaround":
        # 360-deg turnaround with continuous radius, direction, and exit dock length
        r = rng.uniform(11.5, 17.5)
        turn_span = 2.0 * np.pi + rng.uniform(0.3, math.pi * 0.75)
        phi = np.linspace(0.0, turn_span, max(20, int(turn_span * r / ds)))
        dir_sign = float(rng.choice([-1.0, 1.0]))
        x_circ = r * np.sin(phi)
        y_circ = dir_sign * r * (1.0 - np.cos(phi))
        lead_s = np.linspace(-rng.uniform(14.0, 22.0), 0.0, max(5, int(18.0 / ds)))
        x_lead = lead_s
        y_lead = np.zeros_like(lead_s)
        end_phi = phi[-1]
        dock_len = rng.uniform(16.0, 26.0)
        dock_s = np.linspace(0.0, dock_len, max(5, int(dock_len / ds)))
        x_dock = x_circ[-1] + dock_s * math.cos(end_phi)
        y_dock = y_circ[-1] + dock_s * dir_sign * math.sin(end_phi)
        x = np.concatenate([x_lead[:-1], x_circ, x_dock[1:]])
        y = np.concatenate([y_lead[:-1], y_circ, y_dock[1:]])

    elif path_type == "high_curvature_chicane":
        # Rapid, aggressive double kink with near-limit curvature (R ~ 10m)
        r_c = rng.uniform(10.0, 13.5)
        k_c = 1.0 / r_c
        dir_c = float(rng.choice([-1.0, 1.0]))
        w_kink = max(5, int(rng.uniform(10.0, 16.0) / ds))
        curv = np.zeros(num_pts)
        mid = num_pts // 2
        curv[mid - w_kink : mid] = dir_c * k_c
        curv[mid : mid + w_kink] = -dir_c * k_c
        th = np.cumsum(curv * ds)
        x = np.cumsum(np.cos(th) * ds)
        y = np.cumsum(np.sin(th) * ds)

    elif path_type == "yard_grand_tour":
        # Truly procedural grand tour visiting 8 to 14 sectors in randomized sequence
        num_sectors = rng.integers(8, 14)
        angles = np.linspace(0.0, 2.0 * math.pi, num_sectors, endpoint=False)
        if rng.random() < 0.5:
            angles = angles[::-1]
        angles = (angles + rng.uniform(0.0, 2.0 * math.pi)) % (2.0 * math.pi)
        angles = np.sort(angles) if rng.random() < 0.5 else np.sort(angles)[::-1]
        pts = []
        for a in angles:
            r_frac = rng.uniform(0.40, 1.0)
            rx = r_frac * rng.uniform(48.0, 62.0)
            ry = r_frac * rng.uniform(22.0, 32.0)
            pts.append((rx * math.cos(a), ry * math.sin(a)))
        last_x, last_y = pts[-1]
        pts.append((last_x * 0.4 + rng.uniform(-10.0, 10.0), last_y * 0.4 + rng.uniform(-10.0, 10.0)))
        pts.append((rng.uniform(-20.0, 20.0), 0.0))
        path_obj = smooth_and_constrain_user_path(pts, ds=ds, min_radius=11.0)
        x, y = path_obj.x, path_obj.y

    elif path_type == "yard_serpentine_slalom":
        # Procedural multi-gate slalom snaking with variable gate counts, positions, and offsets
        s_dir = float(rng.choice([-1.0, 1.0]))
        num_gates = rng.integers(4, 9)
        x_start = rng.uniform(-64.0, -56.0)
        x_end = rng.uniform(56.0, 64.0)
        xs = np.linspace(x_start, x_end, num_gates) + rng.uniform(-3.5, 3.5, num_gates)
        pts = []
        for i, gx in enumerate(xs):
            gy = s_dir * (1 if i % 2 == 0 else -1) * rng.uniform(22.0, 32.0)
            pts.append((gx, gy))
        pts.append((x_end + rng.uniform(4.0, 8.0), 0.0))
        path_obj = smooth_and_constrain_user_path(pts, ds=ds, min_radius=11.0)
        x, y = path_obj.x, path_obj.y

    elif path_type == "yard_double_figure8":
        # Two interconnected figure-8 lobes with continuous parameters
        a1 = rng.uniform(48.0, 62.0)
        a2 = rng.uniform(4.0, 10.0)
        b = rng.uniform(24.0, 32.0)
        t = np.linspace(-np.pi, np.pi, max(200, int(length_m / ds)))
        x_raw = a1 * np.sin(t) + a2 * np.sin(3.0 * t)
        y_raw = b * np.sin(2.0 * t)
        path_obj = smooth_and_constrain_user_path(list(zip(x_raw, y_raw)), ds=ds, min_radius=10.5)
        x, y = path_obj.x, path_obj.y

    elif path_type == "yard_perimeter_clover":
        # Sprawling multi-intersection cloverleaf with continuous scales
        a = rng.uniform(50.0, 62.0)
        b1 = rng.uniform(24.0, 32.0)
        b2 = rng.uniform(8.0, 14.0)
        t = np.linspace(-np.pi, np.pi, max(220, int(length_m / ds)))
        x_raw = a * np.sin(t)
        y_raw = b1 * np.sin(2.0 * t) + b2 * np.sin(t)
        path_obj = smooth_and_constrain_user_path(list(zip(x_raw, y_raw)), ds=ds, min_radius=10.5)
        x, y = path_obj.x, path_obj.y

    elif path_type == "hairpin":
        turn_angle = rng.uniform(math.radians(75.0), math.radians(180.0))
        curv = (turn_angle / length_m) * 2.0 * np.sin(np.pi * s / length_m)**2
        curv = np.clip(curv, -max_curvature, max_curvature) * float(rng.choice([-1.0, 1.0]))
        th = np.cumsum(curv * ds)
        x = np.cumsum(np.cos(th) * ds)
        y = np.cumsum(np.sin(th) * ds)

    else:  # s_curve / smooth_random
        w1 = length_m * rng.uniform(0.4, 0.75)
        w2 = length_m * rng.uniform(0.8, 1.5)
        a1 = rng.uniform(3.5, 7.5)
        a2 = rng.uniform(1.5, 4.5)
        phi2 = rng.uniform(0.0, 2.0 * math.pi)
        curv = (a1 * (2.0 * math.pi / w1)**2) * np.sin(2.0 * math.pi * s / w1) + \
               (a2 * (2.0 * math.pi / w2)**2) * np.sin(2.0 * math.pi * s / w2 + phi2)
        curv = np.clip(curv, -max_curvature, max_curvature)
        th = np.cumsum(curv * ds)
        x = np.cumsum(np.cos(th) * ds)
        y = np.cumsum(np.sin(th) * ds)

    # Apply arbitrary 2D spatial rotation and reflection invariance
    if allow_rotation:
        rot_angle = rng.uniform(0.0, 2.0 * math.pi)
        flip_y = float(rng.choice([-1.0, 1.0]))
        cos_r, sin_r = math.cos(rot_angle), math.sin(rot_angle)
        x_rot = cos_r * x - sin_r * (flip_y * y)
        y_rot = sin_r * x + cos_r * (flip_y * y)
        x, y = x_rot, y_rot

    return Path(x, y, ds=ds, constrain_curvature=True)
