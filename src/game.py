import os
import sys
import math
import random
import time
from typing import Tuple, Optional, List, Dict, Any
import numpy as np
import pygame
import torch

os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')

from src.kinematics import TractorDoubleTrailerKinematics, VehicleDimensions, VehicleState
from src.path_generator import Path, smooth_and_constrain_user_path, generate_synthetic_path
from src.model import ControlTransformer

# ==============================================================================
# CLEAN 2.5D NIGHT DIORAMA PALETTE (Tactile Top-Down View)
# ==============================================================================
COLOR_NIGHT_BG        = (13, 17, 23)       # Deep nocturnal slate
COLOR_ROAD_SURF       = (20, 26, 35)       # Dark asphalt track
COLOR_PATH_CENTER     = (58, 102, 155)     # Target guidance centerline
COLOR_PATH_CORRIDOR   = (28, 36, 48)
COLOR_F1_RED          = (220, 36, 42)      # F1 racing red curb
COLOR_F1_RED_DARK     = (170, 24, 30)      # Shadow bevel on red curb
COLOR_F1_WHITE        = (242, 244, 250)    # F1 racing white curb
COLOR_F1_WHITE_DARK   = (195, 200, 210)    # Shadow bevel on white curb
COLOR_TRACK_LIMIT     = (235, 240, 250)    # White track limit boundary line

# High-Visibility Industrial Yard & Perimeter Colors
COLOR_HAZARD_YELLOW    = (246, 198, 36)     # High-vis safety yellow
COLOR_HAZARD_BLACK     = (18, 22, 28)       # Matte black hazard stripe
COLOR_PERIMETER_WALL   = (48, 62, 80)       # Reinforced concrete retaining wall
COLOR_PERIMETER_FENCE  = (72, 88, 112)      # Chain-link security wire
COLOR_YARD_PAD         = (22, 28, 38)       # Heavy-duty concrete depot apron
COLOR_YARD_EXPANSION   = (32, 40, 54)       # Slab expansion joint sealant
COLOR_DRAIN_GRATE      = (26, 34, 46)       # Linear storm drainage trench

# 2.5D Vehicle Palette (Clean Top-Down Architectural Styling)
COLOR_CAB_ROOF        = (242, 154, 46)     # Vibrant Industrial Ochre
COLOR_CAB_BEVEL       = (214, 130, 32)
COLOR_CAB_HOOD        = (225, 140, 38)     # Lower front engine hood
COLOR_CAB_GLASS       = (18, 26, 38)       # Tinted dark glass windshield
COLOR_CHROME          = (225, 230, 240)    # Grille & bumper
COLOR_EXHAUST         = (185, 190, 200)

COLOR_T1_ROOF         = (64, 100, 134)     # Lead Trailer: Nordic Slate Blue
COLOR_T1_BEVEL        = (50, 80, 108)
COLOR_T2_ROOF         = (68, 118, 88)      # Rear Trailer: Forest Sage Green
COLOR_T2_BEVEL        = (52, 94, 68)
COLOR_REEFER          = (218, 224, 232)    # Front refrigeration bulkhead unit

COLOR_CHASSIS         = (28, 32, 40)       # Exposed rear deck
COLOR_TIRE            = (22, 24, 30)       # Matte rubber
COLOR_RIM             = (145, 150, 160)    # Silver wheel rim
COLOR_SHADOW          = (8, 10, 14, 185)   # Directional drop-shadow
COLOR_HITCH           = (195, 200, 210)

# Soft Thematic Dynamic Lighting
COLOR_HEADLIGHT_BEAM  = (210, 235, 255)    # Transparent volumetric light cone
COLOR_HEADLIGHT_CORE  = (255, 255, 240)
COLOR_TAILLIGHT_RED   = (255, 45, 55)      # Red rear taillight glow
COLOR_TARGET_GREEN    = (60, 240, 145)     # Soft green terminus badge
COLOR_STATUS_GREEN    = (60, 240, 145)     # Status emerald
COLOR_ACCENT_AMBER    = (245, 175, 55)
COLOR_ACCENT_CYAN     = (65, 195, 235)

# Frosted Glass HUD
COLOR_HUD_BG          = (15, 19, 27, 230)
COLOR_HUD_BORDER      = (42, 54, 72)
COLOR_TEXT_WHITE      = (235, 242, 252)
COLOR_TEXT_DIM        = (140, 155, 175)


class TopDownCamera:
    """
    Fixed 2D orthogonal top-down camera anchored to a permanent, fixed rectangular yard.
    Yard dimensions: 160.0m x 100.0m ([-80, +80] x [-50, +50] meters).
    Origin (0, 0) is permanently locked at the center of the viewport.
    Guarantees 100% consistent physical scale and spatial intuition across all presets
    and user drawings without camera jump, shift, or zoom distortions.
    """
    def __init__(self, width: int, height: int, yard_w: float = 160.0, yard_h: float = 100.0):
        self.width = width
        self.height = height
        self.yard_w = yard_w
        self.yard_h = yard_h
        self.cx = 0.0
        self.cy = 0.0
        self.scale = 8.5
        self.update_scale()

    def update_scale(self):
        # Scale is computed so the fixed physical rectangle fits with standard clean padding
        margin_x = 48.0
        margin_y = 48.0
        scale_x = (self.width - margin_x) / self.yard_w
        scale_y = (self.height - margin_y) / self.yard_h
        self.scale = min(scale_x, scale_y)

    def fit_to_bounds(self, min_x: float = None, max_x: float = None, min_y: float = None, max_y: float = None, instant: bool = False):
        # Camera center is permanently locked to yard origin (0, 0)
        self.cx = 0.0
        self.cy = 0.0
        self.update_scale()

    def update(self):
        pass

    @property
    def target_x(self) -> float:
        return 0.0

    @property
    def target_y(self) -> float:
        return 0.0

    def project(self, wx: float, wy: float, z: float = 0.0) -> Tuple[int, int]:
        sx = int(self.width // 2 + (wx - self.cx) * self.scale)
        sy = int(self.height // 2 - (wy - self.cy) * self.scale)
        return sx, sy

    def unproject_ground(self, sx: int, sy: int) -> Tuple[float, float]:
        wx = self.cx + (sx - self.width // 2) / self.scale
        wy = self.cy + (self.height // 2 - sy) / self.scale
        return wx, wy


class JackknifeGame25D:
    def __init__(self, model_path: Optional[str] = "models/model.pt"):
        pygame.init()
        pygame.font.init()
        self.width, self.height = 1400, 900
        self.screen = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE | pygame.DOUBLEBUF)
        pygame.display.set_caption("Multi-Trailer Reversing Simulation | Autonomous ControlTransformer")
        self.clock = pygame.time.Clock()

        # Dynamic Resolution-Responsive Typography (Scales crisp and readable from 1080p to 4K)
        self.ui_scale = 1.0
        self._init_fonts()

        self.dims = VehicleDimensions()
        self.kinematics = TractorDoubleTrailerKinematics(self.dims)
        self.camera = TopDownCamera(self.width, self.height, yard_w=160.0, yard_h=100.0)

        # PyTorch Model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[ControlTransformer] = None
        self.model_path = model_path
        self.model_mtime = 0.0
        self.load_model(model_path)

        # Simulation & State
        self.state = VehicleState()
        self.mode = "AUTONOMOUS"  # "AUTONOMOUS" or "MANUAL"
        self.ai_save_active = False
        self.is_running = True
        self.is_paused = False
        self.has_arrived = False
        self.is_off_track = False
        self.dt = 0.02
        self.current_speed = 0.0
        self.target_steer = 0.0
        self.current_path_idx = 0  # Monotonic curve arc-length tracker (handles 8-shapes and loops)
        self.time_multiplier = 1   # Fast-forward simulation speed warp: 1x, 8x, 16x
        self.speed_options = [1, 8, 16]
        self.speed_btn_rect = pygame.Rect(0, 0, int(122 * self.ui_scale), int(30 * self.ui_scale))
        self.sample_btn_rect = pygame.Rect(0, 0, int(144 * self.ui_scale), int(30 * self.ui_scale))

        # Drawing mode
        self.drawing_points: List[Tuple[float, float]] = []
        self.is_drawing = False

        # Pre-allocated High-Performance Surfaces (Zero memory allocation in 4K render loop)
        self.background_surf: Optional[pygame.Surface] = None
        self.dynamic_alpha_surf = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        self.vignette_surf = None
        self._build_vignette()

        # Path & Procedural Yard
        self.path: Optional[Path] = None
        self.preset_idx = 2
        self.load_preset_path(self.preset_idx)

        # Telemetry & Atmosphere
        self.cross_track_err = 0.0
        self.inference_time_ms = 0.0
        self.attention_weights = np.ones(64) / 64.0
        self.action_probs = np.ones(15) / 15.0
        self.show_help = True
        self.show_rays = True
        self.frame_count = 0

    def _init_fonts(self):
        """
        Dynamically scales typography according to current display resolution.
        Ensures HUD text, meters, and diagnostics remain crisp and easily readable on 4K monitors.
        """
        self.ui_scale = max(1.0, min(self.width / 1400.0, self.height / 900.0))
        s = self.ui_scale
        self.font_big_alert = pygame.font.SysFont("ubuntusans,liberationsans,dejavusans", max(18, int(34 * s)), bold=True)
        self.font_big_sub   = pygame.font.SysFont("ubuntumono,liberationmono,dejavusansmono", max(11, int(15 * s)), bold=True)
        self.font_alert     = pygame.font.SysFont("ubuntusans,liberationsans,dejavusans", max(14, int(22 * s)), bold=True)
        self.font_title     = pygame.font.SysFont("ubuntusans,liberationsans,dejavusans", max(11, int(15 * s)), bold=True)
        self.font_mono      = pygame.font.SysFont("ubuntumono,liberationmono,dejavusansmono", max(10, int(13 * s)))
        self.font_mono_bold = pygame.font.SysFont("ubuntumono,liberationmono,dejavusansmono", max(10, int(14 * s)), bold=True)
        self.font_badge     = pygame.font.SysFont("ubuntusans,liberationsans,dejavusans", max(9, int(12 * s)), bold=True)
        self.font_sm        = pygame.font.SysFont("ubuntumono,liberationmono,dejavusansmono", max(8, int(11 * s)))

    def _build_vignette(self):
        self.vignette_surf = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        diag = math.hypot(self.width, self.height)
        center = (self.width // 2, self.height // 2)
        for rad_frac, alpha in [(0.70, 35), (0.80, 80), (0.90, 140), (1.02, 210)]:
            r = int(diag * 0.5 * rad_frac)
            pygame.draw.circle(self.vignette_surf, (8, 10, 14, alpha), center, r, int(diag * 0.12))

    def load_model(self, path: Optional[str]):
        if path and os.path.exists(path):
            try:
                mtime = os.path.getmtime(path)
                ckpt = torch.load(path, map_location=self.device)
                self.model = ControlTransformer().to(self.device)
                self.model.load_state_dict(ckpt['model_state_dict'])
                self.model.eval()
                self.model_mtime = mtime
                print(f"Loaded ControlTransformer checkpoint from {path} (Val Adj Acc: {ckpt.get('val_adj_acc', 0):.1f}%)")
            except Exception as e:
                print(f"Failed to load checkpoint: {e}")
                self.model = None

    def _create_centered_path(self, raw_x, raw_y, ds: float = 0.8) -> Path:
        cx = (np.min(raw_x) + np.max(raw_x)) / 2.0
        cy = (np.min(raw_y) + np.max(raw_y)) / 2.0
        return Path(np.array(raw_x) - cx, np.array(raw_y) - cy, ds=ds)

    def load_preset_path(self, preset_id: int):
        self.preset_idx = preset_id
        if preset_id == 1:
            # 90-Degree Standard Warehouse Alley Dock (68m, R >= 14m)
            s = np.linspace(0.0, 68.0, 86)
            curv = np.zeros_like(s)
            t_mask = (s >= 16.0) & (s <= 38.0)
            tau = (s[t_mask] - 16.0) / 22.0
            curv[t_mask] = (math.pi / 2.0 / 22.0) * 2.0 * np.sin(np.pi * tau)**2
            th = np.cumsum(curv * 0.8)
            x = np.cumsum(np.cos(th) * 0.8)
            y = np.cumsum(np.sin(th) * 0.8)
            self.path = self._create_centered_path(x, y, ds=0.8)
        elif preset_id == 2:
            # S-Curve (95m)
            s = np.linspace(0, 95.0, 120)
            y = 6.5 * np.sin(s * 0.075)
            self.path = self._create_centered_path(s, y, ds=0.8)
        elif preset_id == 3:
            # Diagonal "Z" Chicane Reversing Lane (128m)
            # Starting at North-East -> Long diagonal approach -> Z-jog dogleg chicane -> Southwest dock
            pts = [
                (58.0,  31.5),
                (42.0,  25.5),
                (24.0,  19.0),
                (8.0,   12.5),
                (-4.0,   0.0),
                (-18.0, -11.5),
                (-36.0, -18.5),
                (-56.0, -26.5)
            ]
            raw_path = smooth_and_constrain_user_path(pts, ds=0.8, min_radius=12.0)
            self.path = self._create_centered_path(raw_path.x, raw_path.y, ds=0.8)
        elif preset_id == 4:
            # 90-Degree Hairpin Loading Bay
            s = np.linspace(0, 70.0, 110)
            th = np.clip(s * 0.024, 0.0, math.pi / 2.0)
            x = np.cumsum(np.cos(th) * 0.8)
            y = np.cumsum(np.sin(th) * 0.8)
            self.path = self._create_centered_path(x, y, ds=0.8)
        elif preset_id == 5:
            # Extreme Precision Alley-Dock Gauntlet: Four Hard 90° Turns & Chicanes (162m)
            # Replicates Preset [1]'s brutal R_peak = 7.0m alley-dock curvature across a multi-turn labyrinth:
            # 1. Approach straight (16m) -> Hard 90° Right (R_peak = 7.0m)
            # 2. Short transition (12m)  -> Hard 90° Left (R_peak = 7.0m, Chicane exit)
            # 3. Intermediate straight (14m) -> Hard 90° Left (R_peak = 7.0m)
            # 4. Short transition (12m)  -> Hard 90° Right (R_peak = 7.0m)
            # 5. Final alley docking bay straight (18m)
            ds = 0.8
            def make_turn_curv(num_samples, angle_rad):
                tau = np.linspace(0.0, 1.0, num_samples)
                length = num_samples * ds
                return (angle_rad / length) * 2.0 * np.sin(np.pi * tau)**2

            n_turn = int(round(22.0 / ds))
            curv_parts = [
                np.zeros(int(round(16.0 / ds))),
                make_turn_curv(n_turn, math.pi / 2.0),
                np.zeros(int(round(12.0 / ds))),
                make_turn_curv(n_turn, -math.pi / 2.0),
                np.zeros(int(round(14.0 / ds))),
                make_turn_curv(n_turn, -math.pi / 2.0),
                np.zeros(int(round(12.0 / ds))),
                make_turn_curv(n_turn, math.pi / 2.0),
                np.zeros(int(round(18.0 / ds)))
            ]
            all_curv = np.concatenate(curv_parts)
            th = np.cumsum(all_curv * ds)
            x = np.cumsum(np.cos(th) * ds)
            y = np.cumsum(np.sin(th) * ds)
            self.path = self._create_centered_path(x, y, ds=0.8)

        self.generate_procedural_yard()
        self.reset_vehicle_to_path()

    def sample_random_prior_path(self):
        """
        Samples a physically-valid reversing trajectory from our synthetic prior distribution:
        - Massive yard-filling grand tours, 5-gate serpentine slaloms, dual figure-8s, and perimeter clovers.
        - Crazy self-intersecting loops, figure-8s, 360-degree roundabouts, sharp switchbacks,
          cloverleaf intersections, serpentine knots, and high-curvature maneuvers.
        - Automatically fitted and centered inside the fixed 160m x 100m yard rectangle,
          safely clear of perimeter hazard curbs and walls.
        - Automatically aligns vehicle at path start and updates static background.
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
            "multi_scale_s",
            "segmented_dock",
            "continuous_arc",
            "figure8_inflection",
            "compound_kink",
            "piecewise_clothoid",
            "rapid_chicane",
            "z_track",
            "jagged_corners"
        ]
        rng = np.random.default_rng()

        for _ in range(16):
            ptype = str(rng.choice(path_types))
            if ptype in ["yard_grand_tour", "yard_double_figure8", "yard_perimeter_clover"]:
                length_m = float(rng.uniform(220.0, 320.0))
            elif ptype in ["yard_serpentine_slalom", "self_intersecting_figure8", "cloverleaf_intersect", "serpentine_knot", "random_fourier"]:
                length_m = float(rng.uniform(140.0, 240.0))
            elif ptype in ["loop_the_loop", "dock_with_360_turnaround", "z_track"]:
                length_m = float(rng.uniform(105.0, 150.0))
            else:
                length_m = float(rng.uniform(70.0, 110.0))
            raw_path = generate_synthetic_path(length_m=length_m, ds=0.8, max_curvature=0.11, path_type=ptype)

            rx, ry = raw_path.x, raw_path.y

            # Align dominant progression along the wider X axis of the yard (160m X vs 100m Y)
            span_x = np.max(rx) - np.min(rx)
            span_y = np.max(ry) - np.min(ry)
            if span_y > span_x:
                rx, ry = -ry, rx

            # Random reflection for variety
            if rng.random() < 0.5:
                rx = -rx
            if rng.random() < 0.5:
                ry = -ry

            # Center trajectory in yard
            cx = (np.min(rx) + np.max(rx)) / 2.0
            cy = (np.min(ry) + np.max(ry)) / 2.0
            centered_x = rx - cx
            centered_y = ry - cy

            # Scale to fit inside [-64, 64]m X and [-35, 35]m Y
            cur_w = max(1e-3, np.ptp(centered_x))
            cur_h = max(1e-3, np.ptp(centered_y))
            max_scale = 1.35 if ptype in ["yard_grand_tour", "yard_serpentine_slalom", "yard_double_figure8", "yard_perimeter_clover"] else 1.15
            scale_fac = min(max_scale, 124.0 / cur_w, 68.0 / cur_h)
            centered_x *= scale_fac
            centered_y *= scale_fac

            # Compute initial vehicle alignment to ensure tractor cab is safely inside yard bounds
            th_start = math.atan2(centered_y[1] - centered_y[0], centered_x[1] - centered_x[0])
            cab_x = centered_x[0] + 11.5 * math.cos(th_start - math.pi)
            cab_y = centered_y[0] + 11.5 * math.sin(th_start - math.pi)

            all_xs = np.append(centered_x, cab_x)
            all_ys = np.append(centered_y, cab_y)

            # Strict clearance: stay within [-70, 70]m X and [-40, 40]m Y
            if (np.min(all_xs) >= -70.0 and np.max(all_xs) <= 70.0 and
                np.min(all_ys) >= -40.0 and np.max(all_ys) <= 40.0):
                self.preset_idx = 0
                self.path = Path(centered_x, centered_y, ds=0.8, constrain_curvature=False)
                print(f"Sampled random prior path: '{ptype}' (Length: {self.path.total_length:.1f}m, max_k={np.max(np.abs(self.path.curvatures)):.3f})")
                self.generate_procedural_yard()
                self.reset_vehicle_to_path()
                return

        # Fallback to guaranteed-safe centered S-curve if random proposals exceeded margins
        fallback = generate_synthetic_path(length_m=72.0, ds=0.8, max_curvature=0.08, path_type="multi_scale_s")
        self.preset_idx = 0
        self.path = self._create_centered_path(fallback.x, fallback.y, ds=0.8)
        print("Sampled random prior path: 'multi_scale_s' (fallback)")
        self.generate_procedural_yard()
        self.reset_vehicle_to_path()

    def reset_vehicle_to_path(self):
        if not self.path or len(self.path.x) < 4:
            return

        th_path = self.path.headings[0]
        th2 = th_path - math.pi
        x2, y2 = self.path.x[0], self.path.y[0]

        th1, th0 = th2, th2
        x1 = x2 + self.dims.L2 * math.cos(th1)
        y1 = y2 + self.dims.L2 * math.sin(th1)
        x0 = x1 + self.dims.L1 * math.cos(th0)
        y0 = y1 + self.dims.L1 * math.sin(th0)

        self.state = VehicleState(float(x0), float(y0), float(th0), 0.0, 0.0, 0.0)
        self.current_speed = 0.0
        self.target_steer = 0.0
        self.current_path_idx = 0
        self.has_arrived = False
        self.is_off_track = False
        self.auto_frame_camera(instant=True)
        self.rebuild_static_background()

        # Seamlessly reload fresh best model if background training updated the checkpoint
        if self.model_path and os.path.exists(self.model_path):
            try:
                if os.path.getmtime(self.model_path) > self.model_mtime:
                    self.load_model(self.model_path)
            except Exception:
                pass

    def auto_frame_camera(self, instant: bool = False):
        if not self.path or len(self.path.x) < 2:
            return
        x0, y0, _ = self.state.x0, self.state.y0, self.state.theta0
        x2, y2, _ = self.kinematics.get_trailer2_axle(self.state)
        all_x = list(self.path.x) + [x0, x2]
        all_y = list(self.path.y) + [y0, y2]
        self.camera.fit_to_bounds(min(all_x), max(all_x), min(all_y), max(all_y), instant=instant)

    def rebuild_static_background(self):
        """
        Pre-bakes the static environment (clean yard boundary, concrete pad, hazard curbs, and F1 track)
        into a single persistent surface for maximum 60fps performance on 4K displays.
        """
        if self.width <= 0 or self.height <= 0:
            return
        self.background_surf = pygame.Surface((self.width, self.height))
        self.draw_night_ground(self.background_surf)
        self.draw_path_static(self.background_surf)
        if self.vignette_surf is not None:
            self.background_surf.blit(self.vignette_surf, (0, 0))

    def generate_procedural_yard(self):
        pass

    def handle_input(self):
        keys = pygame.key.get_pressed()

        # Keyboard Driving in Manual Mode
        if self.mode == "MANUAL" and not self.ai_save_active:
            # Smooth, realistic truck power steering (delicate micro-taps, progressive turns)
            steer_rate = math.radians(22.0)
            if keys[pygame.K_a] or keys[pygame.K_LEFT]:
                self.steer_hold_time = getattr(self, "steer_hold_time", 0.0) + self.dt
                ramp = min(1.35, 0.55 + self.steer_hold_time * 0.55)
                self.target_steer = min(self.dims.max_steer_rad, self.target_steer + steer_rate * ramp * self.dt)
            elif keys[pygame.K_d] or keys[pygame.K_RIGHT]:
                self.steer_hold_time = getattr(self, "steer_hold_time", 0.0) + self.dt
                ramp = min(1.35, 0.55 + self.steer_hold_time * 0.55)
                self.target_steer = max(-self.dims.max_steer_rad, self.target_steer - steer_rate * ramp * self.dt)
            else:
                self.steer_hold_time = 0.0
                self.target_steer *= 0.90

            # Intuitive gas/reverse pedals (can drive forward even after docking!)
            if keys[pygame.K_w] or keys[pygame.K_UP]:
                self.has_arrived = False
                self.is_off_track = False
                self.current_speed = 2.0
            elif keys[pygame.K_s] or keys[pygame.K_DOWN]:
                self.has_arrived = False
                self.is_off_track = False
                self.current_speed = -1.5
            else:
                self.current_speed *= 0.85
                if abs(self.current_speed) < 0.05:
                    self.current_speed = 0.0
        elif self.mode == "AUTONOMOUS" or self.ai_save_active:
            self.current_speed = 0.0 if (self.has_arrived or self.is_off_track or self.is_paused) else -1.5

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.is_running = False
            elif event.type == pygame.VIDEORESIZE:
                self.width, self.height = event.w, event.h
                self.screen = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE | pygame.DOUBLEBUF)
                self.camera.width = self.width
                self.camera.height = self.height
                self.camera.update_scale()
                self._init_fonts()
                self._build_vignette()
                self.dynamic_alpha_surf = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
                self.auto_frame_camera(instant=True)
                self.rebuild_static_background()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    if self.speed_btn_rect.collidepoint(event.pos):
                        curr_idx = self.speed_options.index(self.time_multiplier)
                        self.time_multiplier = self.speed_options[(curr_idx + 1) % len(self.speed_options)]
                    elif self.sample_btn_rect.collidepoint(event.pos):
                        self.sample_random_prior_path()
                    else:
                        self.is_drawing = True
                        self.drawing_points = []
                        wx, wy = self.camera.unproject_ground(*event.pos)
                        half_w = self.camera.yard_w / 2.0 - 7.0
                        half_h = self.camera.yard_h / 2.0 - 7.0
                        wx = max(-half_w, min(half_w, wx))
                        wy = max(-half_h, min(half_h, wy))
                        self.drawing_points.append((wx, wy))
            elif event.type == pygame.MOUSEMOTION:
                if self.is_drawing:
                    wx, wy = self.camera.unproject_ground(*event.pos)
                    half_w = self.camera.yard_w / 2.0 - 7.0
                    half_h = self.camera.yard_h / 2.0 - 7.0
                    wx = max(-half_w, min(half_w, wx))
                    wy = max(-half_h, min(half_h, wy))
                    if len(self.drawing_points) > 0:
                        lx, ly = self.drawing_points[-1]
                        if math.hypot(wx - lx, wy - ly) > 0.5:
                            self.drawing_points.append((wx, wy))
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1 and self.is_drawing:
                    self.is_drawing = False
                    if len(self.drawing_points) >= 4:
                        self.path = smooth_and_constrain_user_path(self.drawing_points, ds=0.8)
                        self.generate_procedural_yard()
                        self.reset_vehicle_to_path()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_m:
                    self.mode = "AUTONOMOUS" if self.mode == "MANUAL" else "MANUAL"
                    self.ai_save_active = False
                    if self.mode == "MANUAL":
                        self.has_arrived = False
                        self.is_off_track = False
                elif event.key == pygame.K_r:
                    self.reset_vehicle_to_path()
                elif event.key == pygame.K_SPACE:
                    self.is_paused = not self.is_paused
                elif event.key == pygame.K_t:
                    curr_idx = self.speed_options.index(self.time_multiplier)
                    self.time_multiplier = self.speed_options[(curr_idx + 1) % len(self.speed_options)]
                elif event.key == pygame.K_p:
                    self.sample_random_prior_path()
                elif event.key == pygame.K_h:
                    self.show_help = not self.show_help
                elif event.key in [pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5]:
                    self.load_preset_path(event.key - pygame.K_0)

    def step_autonomous_policy(self):
        if not self.path:
            return

        x2, y2, th2 = self.kinematics.get_trailer2_axle(self.state)
        self.current_path_idx, self.cross_track_err = self.path.update_progression(
            x2, y2, self.current_path_idx
        )

        tokens = self.path.get_lookahead_tokens(
            x2, y2, th2, num_tokens=64, ds=0.8, current_idx=self.current_path_idx
        )
        hitch_state = np.array([
            math.sin(self.state.beta1), math.cos(self.state.beta1),
            math.sin(self.state.beta2), math.cos(self.state.beta2)
        ], dtype=np.float32)

        t_start = time.perf_counter()
        if self.model is not None:
            steer_cmd, attn, probs = self.model.predict_action(tokens, hitch_state, self.device, use_expectation=True)
            self.target_steer = steer_cmd
            if attn is not None:
                self.attention_weights = attn
            self.action_probs = probs
        else:
            self.target_steer = float(np.clip(-2.2 * self.state.beta1 - 1.6 * self.state.beta2, -self.dims.max_steer_rad, self.dims.max_steer_rad))

        self.inference_time_ms = (time.perf_counter() - t_start) * 1000.0

    def update_physics(self, run_inference: bool = True):
        if self.kinematics.is_jackknifed(self.state) or self.is_off_track:
            self.current_speed = 0.0
            return

        if self.path and len(self.path.x) > 4:
            x2, y2, th2 = self.kinematics.get_trailer2_axle(self.state)
            self.current_path_idx, self.cross_track_err = self.path.update_progression(
                x2, y2, self.current_path_idx
            )

            # Check Tractor Cab lateral excursion from starting road axis
            th0 = self.path.headings[0]
            fx = self.state.x0 + self.dims.L0 * math.cos(self.state.theta0)
            fy = self.state.y0 + self.dims.L0 * math.sin(self.state.theta0)
            cab_lat_err = abs(-(fx - self.path.x[0]) * math.sin(th0) + (fy - self.path.y[0]) * math.cos(th0))

            if (self.mode == "AUTONOMOUS" or self.ai_save_active) and (self.cross_track_err > 5.5 or (self.current_path_idx <= 5 and cab_lat_err > 5.5)):
                self.is_off_track = True
                self.current_speed = 0.0
                return

            # Trigger arrival if Trailer 2 has reached or crossed the finish line while reversing
            if (self.mode == "AUTONOMOUS" or self.ai_save_active) and self.current_speed < 0:
                if self.current_path_idx >= len(self.path.x) - 6:
                    end_x, end_y = self.path.x[-1], self.path.y[-1]
                    end_th = self.path.headings[-1]
                    t_x, t_y = math.cos(end_th), math.sin(end_th)
                    n_x, n_y = -math.sin(end_th), math.cos(end_th)

                    dx = x2 - end_x
                    dy = y2 - end_y
                    s_past = dx * t_x + dy * t_y
                    lat_err = abs(dx * n_x + dy * n_y)

                    # Arrived if within 2.2m approaching the line OR crossed past the line (up to 6.0m past),
                    # provided lateral position is within full track width (4.8m)
                    if -2.2 <= s_past <= 6.0 and lat_err <= 4.8:
                        self.has_arrived = True
                        self.current_speed = 0.0

        if not self.has_arrived and not self.is_off_track and not self.is_paused and (self.mode == "AUTONOMOUS" or self.ai_save_active):
            self.current_speed = -1.5
            if run_inference:
                self.step_autonomous_policy()

        self.state = self.kinematics.step_rk4(
            state=self.state,
            v0=self.current_speed,
            target_phi=self.target_steer,
            dt=self.dt
        )

    # ==========================================================================
    # CLEAN 2.5D TOP-DOWN RENDERING PIPELINE
    # ==========================================================================

    def draw_night_ground(self, surf: pygame.Surface):
        # 1. Dark Nocturnal Void outside Yard Boundary
        surf.fill(COLOR_NIGHT_BG)

        yw, yh = self.camera.yard_w, self.camera.yard_h
        half_w, half_h = yw / 2.0, yh / 2.0

        # 2. Main Concrete Logistics Pad inside Yard (160m x 100m)
        tl = self.camera.project(-half_w, half_h)
        tr = self.camera.project(half_w, half_h)
        br = self.camera.project(half_w, -half_h)
        bl = self.camera.project(-half_w, -half_h)
        yard_poly = [tl, tr, br, bl]
        pygame.draw.polygon(surf, COLOR_YARD_PAD, yard_poly)

        # 3. Heavy-Duty Concrete Slab Expansion Grid (16m x 16m panels)
        grid_xs = np.arange(-half_w + 16.0, half_w, 16.0)
        grid_ys = np.arange(-half_h + 16.0, half_h, 16.0)

        for gx in grid_xs:
            p1 = self.camera.project(gx, -half_h)
            p2 = self.camera.project(gx, half_h)
            pygame.draw.line(surf, COLOR_YARD_EXPANSION, p1, p2, 1)

        for gy in grid_ys:
            p1 = self.camera.project(-half_w, gy)
            p2 = self.camera.project(half_w, gy)
            pygame.draw.line(surf, COLOR_YARD_EXPANSION, p1, p2, 1)

        # Expansion joint cross caps at slab intersections
        for gx in grid_xs:
            for gy in grid_ys:
                pt = self.camera.project(gx, gy)
                pygame.draw.circle(surf, (36, 46, 62), pt, 2)

        # 4. Drainage Stormwater Trench Grates along inner perimeter sections
        for drain_y in [-half_h + 3.0, half_h - 3.0]:
            d_p1 = self.camera.project(-half_w + 6.0, drain_y)
            d_p2 = self.camera.project(half_w - 6.0, drain_y)
            pygame.draw.line(surf, COLOR_DRAIN_GRATE, d_p1, d_p2, 3)

        # 5. HIGH-CONTRAST SAFETY HAZARD PERIMETER CURB (Unmistakable Yard Boundary!)
        curb_w = 1.6
        curb_surf = pygame.Surface((self.width, self.height), pygame.SRCALPHA)

        stripe_step = 2.4
        for x_start in np.arange(-half_w, half_w, stripe_step):
            x_end = x_start + stripe_step
            p1 = self.camera.project(x_start, half_h)
            p2 = self.camera.project(min(half_w, x_end), half_h)
            p3 = self.camera.project(min(half_w, x_end - curb_w * 0.7), half_h - curb_w)
            p4 = self.camera.project(max(-half_w, x_start - curb_w * 0.7), half_h - curb_w)
            
            is_yellow = (int(x_start / stripe_step) % 2 == 0)
            col = COLOR_HAZARD_YELLOW if is_yellow else COLOR_HAZARD_BLACK
            pygame.draw.polygon(curb_surf, col, [p1, p2, p3, p4])

        for x_start in np.arange(-half_w, half_w, stripe_step):
            if -12.0 <= x_start <= 10.0:
                continue
            x_end = x_start + stripe_step
            p1 = self.camera.project(x_start, -half_h)
            p2 = self.camera.project(min(half_w, x_end), -half_h)
            p3 = self.camera.project(min(half_w, x_end + curb_w * 0.7), -half_h + curb_w)
            p4 = self.camera.project(max(-half_w, x_start + curb_w * 0.7), -half_h + curb_w)
            
            is_yellow = (int(x_start / stripe_step) % 2 == 0)
            col = COLOR_HAZARD_YELLOW if is_yellow else COLOR_HAZARD_BLACK
            pygame.draw.polygon(curb_surf, col, [p1, p2, p3, p4])

        for y_start in np.arange(-half_h, half_h, stripe_step):
            y_end = y_start + stripe_step
            p1 = self.camera.project(-half_w, y_start)
            p2 = self.camera.project(-half_w, min(half_h, y_end))
            p3 = self.camera.project(-half_w + curb_w, min(half_h, y_end - curb_w * 0.7))
            p4 = self.camera.project(-half_w + curb_w, max(-half_h, y_start - curb_w * 0.7))
            
            is_yellow = (int(y_start / stripe_step) % 2 == 0)
            col = COLOR_HAZARD_YELLOW if is_yellow else COLOR_HAZARD_BLACK
            pygame.draw.polygon(curb_surf, col, [p1, p2, p3, p4])

        for y_start in np.arange(-half_h, half_h, stripe_step):
            y_end = y_start + stripe_step
            p1 = self.camera.project(half_w, y_start)
            p2 = self.camera.project(half_w, min(half_h, y_end))
            p3 = self.camera.project(half_w - curb_w, min(half_h, y_end + curb_w * 0.7))
            p4 = self.camera.project(half_w - curb_w, max(-half_h, y_start + curb_w * 0.7))
            
            is_yellow = (int(y_start / stripe_step) % 2 == 0)
            col = COLOR_HAZARD_YELLOW if is_yellow else COLOR_HAZARD_BLACK
            pygame.draw.polygon(curb_surf, col, [p1, p2, p3, p4])

        inner_tl = self.camera.project(-half_w + curb_w, half_h - curb_w)
        inner_tr = self.camera.project(half_w - curb_w, half_h - curb_w)
        inner_br = self.camera.project(half_w - curb_w, -half_h + curb_w)
        inner_bl = self.camera.project(-half_w + curb_w, -half_h + curb_w)
        pygame.draw.polygon(curb_surf, (255, 220, 50, 240), [inner_tl, inner_tr, inner_br, inner_bl], 2)

        surf.blit(curb_surf, (0, 0))

        # 7. Heavy Outer Reinforced Security Retaining Wall & Fence Posts
        wall_pts = [tl, tr, br, bl]
        pygame.draw.polygon(surf, COLOR_PERIMETER_WALL, wall_pts, 3)

        fence_offset = 1.0
        f_tl = self.camera.project(-half_w - fence_offset, half_h + fence_offset)
        f_tr = self.camera.project(half_w + fence_offset, half_h + fence_offset)
        f_br = self.camera.project(half_w + fence_offset, -half_h - fence_offset)
        f_bl = self.camera.project(-half_w - fence_offset, -half_h - fence_offset)
        pygame.draw.polygon(surf, COLOR_PERIMETER_FENCE, [f_tl, f_tr, f_br, f_bl], 1)

        for px in np.arange(-half_w - fence_offset, half_w + fence_offset + 1.0, 8.0):
            pt_top = self.camera.project(px, half_h + fence_offset)
            pt_bot = self.camera.project(px, -half_h - fence_offset)
            pygame.draw.circle(surf, (160, 175, 195), pt_top, 2)
            pygame.draw.circle(surf, (160, 175, 195), pt_bot, 2)
        for py in np.arange(-half_h - fence_offset, half_h + fence_offset + 1.0, 8.0):
            pt_left = self.camera.project(-half_w - fence_offset, py)
            pt_right = self.camera.project(half_w + fence_offset, py)
            pygame.draw.circle(surf, (160, 175, 195), pt_left, 2)
            pygame.draw.circle(surf, (160, 175, 195), pt_right, 2)

        # 8. Four Corner Fortified Security Towers with Illuminated Beacons
        tower_size = 4.8
        for tx, ty in [(-half_w, half_h), (half_w, half_h), (half_w, -half_h), (-half_w, -half_h)]:
            base_p1 = self.camera.project(tx - tower_size / 2.0, ty + tower_size / 2.0)
            base_p2 = self.camera.project(tx + tower_size / 2.0, ty + tower_size / 2.0)
            base_p3 = self.camera.project(tx + tower_size / 2.0, ty - tower_size / 2.0)
            base_p4 = self.camera.project(tx - tower_size / 2.0, ty - tower_size / 2.0)
            pygame.draw.polygon(surf, (34, 44, 58), [base_p1, base_p2, base_p3, base_p4])
            pygame.draw.polygon(surf, (62, 78, 102), [base_p1, base_p2, base_p3, base_p4], 2)

            cab_size = 3.4
            cab_p1 = self.camera.project(tx - cab_size / 2.0, ty + cab_size / 2.0)
            cab_p2 = self.camera.project(tx + cab_size / 2.0, ty + cab_size / 2.0)
            cab_p3 = self.camera.project(tx + cab_size / 2.0, ty - cab_size / 2.0)
            cab_p4 = self.camera.project(tx - cab_size / 2.0, ty - cab_size / 2.0)
            pygame.draw.polygon(surf, (45, 54, 68), [cab_p1, cab_p2, cab_p3, cab_p4])

            win_size = 2.4
            win_p1 = self.camera.project(tx - win_size / 2.0, ty + win_size / 2.0)
            win_p2 = self.camera.project(tx + win_size / 2.0, ty + win_size / 2.0)
            win_p3 = self.camera.project(tx + win_size / 2.0, ty - win_size / 2.0)
            win_p4 = self.camera.project(tx - win_size / 2.0, ty - win_size / 2.0)
            pygame.draw.polygon(surf, (255, 215, 85), [win_p1, win_p2, win_p3, win_p4], 1)

            p_ctr = self.camera.project(tx, ty)
            pygame.draw.circle(surf, (255, 65, 45), p_ctr, 4)
            pygame.draw.circle(surf, (255, 220, 120), p_ctr, 2)

        # 9. Main Depot Entrance Gate at South Perimeter
        gh_w, gh_d = 3.6, 2.6
        gh_p1 = self.camera.project(-gh_w / 2.0, -half_h + gh_d)
        gh_p2 = self.camera.project( gh_w / 2.0, -half_h + gh_d)
        gh_p3 = self.camera.project( gh_w / 2.0, -half_h)
        gh_p4 = self.camera.project(-gh_w / 2.0, -half_h)
        pygame.draw.polygon(surf, (38, 48, 64), [gh_p1, gh_p2, gh_p3, gh_p4])
        pygame.draw.polygon(surf, (75, 95, 125), [gh_p1, gh_p2, gh_p3, gh_p4], 1)
        
        for arm_x0, arm_x1 in [(-11.0, -2.2), (2.2, 11.0)]:
            b_p0 = self.camera.project(arm_x0, -half_h + 1.2)
            b_p1 = self.camera.project(arm_x1, -half_h + 1.2)
            pygame.draw.line(surf, (235, 45, 45), b_p0, b_p1, 3)
            for bx in np.linspace(arm_x0, arm_x1, 6):
                wp = self.camera.project(bx, -half_h + 1.2)
                pygame.draw.circle(surf, (250, 250, 255), wp, 2)


    def draw_path_static(self, surf: pygame.Surface):
        if not self.path or len(self.path.x) < 2:
            return

        w_road = 3.8
        w_curb = 4.8
        N = len(self.path.x)

        # 1. Road Surface (Plowed Dark Asphalt)
        road_left = [self.camera.project(px - math.sin(pth) * w_road, py + math.cos(pth) * w_road)
                     for px, py, pth in zip(self.path.x, self.path.y, self.path.headings)]
        road_right = [self.camera.project(px + math.sin(pth) * w_road, py - math.cos(pth) * w_road)
                      for px, py, pth in zip(self.path.x, self.path.y, self.path.headings)]

        if len(road_left) >= 2:
            road_surf = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            for i in range(N - 1):
                quad = [road_left[i], road_left[i+1], road_right[i+1], road_right[i]]
                pygame.draw.polygon(road_surf, (*COLOR_ROAD_SURF, 245), quad)
            surf.blit(road_surf, (0, 0))

            # 1b. Road Texture: Transverse Expansion Joints & Rubbered-In Tire Tracks
            detail_surf = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            for i in range(0, N, max(1, int(14.0 / self.path.ds))):
                jx, jy, jth = self.path.x[i], self.path.y[i], self.path.headings[i]
                j_nx, j_ny = -math.sin(jth), math.cos(jth)
                jp1 = self.camera.project(jx - j_nx * (w_road - 0.2), jy - j_ny * (w_road - 0.2))
                jp2 = self.camera.project(jx + j_nx * (w_road - 0.2), jy + j_ny * (w_road - 0.2))
                pygame.draw.line(detail_surf, (14, 18, 25, 110), jp1, jp2, 2)

            tire_l = [self.camera.project(px - math.sin(pth) * 1.25, py + math.cos(pth) * 1.25)
                      for px, py, pth in zip(self.path.x, self.path.y, self.path.headings)]
            tire_r = [self.camera.project(px + math.sin(pth) * 1.25, py - math.cos(pth) * 1.25)
                      for px, py, pth in zip(self.path.x, self.path.y, self.path.headings)]
            if len(tire_l) >= 2:
                pygame.draw.lines(detail_surf, (12, 16, 22, 100), False, tire_l, 4)
                pygame.draw.lines(detail_surf, (12, 16, 22, 100), False, tire_r, 4)
            surf.blit(detail_surf, (0, 0))

        # 2. F1 Curbs (Red and White Alternating Blocks with Bevels)
        curb_surf = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        outer_left_pts = []
        outer_right_pts = []

        for i in range(N - 1):
            x0, y0, th0 = self.path.x[i], self.path.y[i], self.path.headings[i]
            x1, y1, th1 = self.path.x[i+1], self.path.y[i+1], self.path.headings[i+1]
            s_val = self.path.s[i]

            nx0, ny0 = -math.sin(th0), math.cos(th0)
            nx1, ny1 = -math.sin(th1), math.cos(th1)

            is_red = (int(s_val / 1.6) % 2 == 0)
            curb_col = COLOR_F1_RED if is_red else COLOR_F1_WHITE
            bevel_col = COLOR_F1_RED_DARK if is_red else COLOR_F1_WHITE_DARK

            l_in0 = self.camera.project(x0 + nx0 * w_road, y0 + ny0 * w_road)
            l_out0 = self.camera.project(x0 + nx0 * w_curb, y0 + ny0 * w_curb)
            l_out1 = self.camera.project(x1 + nx1 * w_curb, y1 + ny1 * w_curb)
            l_in1 = self.camera.project(x1 + nx1 * w_road, y1 + ny1 * w_road)

            pygame.draw.polygon(curb_surf, (*curb_col, 250), [l_in0, l_out0, l_out1, l_in1])
            pygame.draw.line(curb_surf, (*bevel_col, 255), l_out0, l_out1, 2)

            r_in0 = self.camera.project(x0 - nx0 * w_road, y0 - ny0 * w_road)
            r_out0 = self.camera.project(x0 - nx0 * w_curb, y0 - ny0 * w_curb)
            r_out1 = self.camera.project(x1 - nx1 * w_curb, y1 - ny1 * w_curb)
            r_in1 = self.camera.project(x1 - nx1 * w_road, y1 - ny1 * w_road)

            pygame.draw.polygon(curb_surf, (*curb_col, 250), [r_in0, r_out0, r_out1, r_in1])
            pygame.draw.line(curb_surf, (*bevel_col, 255), r_out0, r_out1, 2)

            if i == 0:
                outer_left_pts.append(l_out0)
                outer_right_pts.append(r_out0)
            outer_left_pts.append(l_out1)
            outer_right_pts.append(r_out1)

        surf.blit(curb_surf, (0, 0))

        # 3. Track Boundary Lines
        pygame.draw.lines(surf, COLOR_TRACK_LIMIT, False, road_left, 2)
        pygame.draw.lines(surf, COLOR_TRACK_LIMIT, False, road_right, 2)

        center_pts = [self.camera.project(px, py) for px, py in zip(self.path.x, self.path.y)]
        pygame.draw.lines(surf, COLOR_PATH_CENTER, False, center_pts, 2)

        # 4. Chequered Finish Band across entire track width (9.6m)
        end_x, end_y = self.path.x[-1], self.path.y[-1]
        end_th = self.path.headings[-1]
        nx = -math.sin(end_th)
        ny = math.cos(end_th)
        tx = math.cos(end_th)
        ty = math.sin(end_th)

        band_depth = 1.4
        num_checks_w = 12
        num_checks_d = 2
        dw = (2.0 * w_curb) / num_checks_w
        dd = band_depth / num_checks_d

        finish_surf = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        for row in range(num_checks_d):
            d0 = -band_depth + row * dd
            d1 = d0 + dd
            for col in range(num_checks_w):
                w0 = -w_curb + col * dw
                w1 = w0 + dw
                is_white = ((row + col) % 2 == 0)
                check_col = (245, 248, 255, 230) if is_white else (20, 24, 32, 230)

                p1 = self.camera.project(end_x + nx * w0 + tx * d0, end_y + ny * w0 + ty * d0)
                p2 = self.camera.project(end_x + nx * w1 + tx * d0, end_y + ny * w1 + ty * d0)
                p3 = self.camera.project(end_x + nx * w1 + tx * d1, end_y + ny * w1 + ty * d1)
                p4 = self.camera.project(end_x + nx * w0 + tx * d1, end_y + ny * w0 + ty * d1)
                pygame.draw.polygon(finish_surf, check_col, [p1, p2, p3, p4])

        surf.blit(finish_surf, (0, 0))

    def draw_dynamic_finish_line_glow(self):
        if not self.path or len(self.path.x) < 2:
            return
        end_x, end_y = self.path.x[-1], self.path.y[-1]
        end_th = self.path.headings[-1]
        nx = -math.sin(end_th)
        ny = math.cos(end_th)
        w_curb = 4.8

        left_end = self.camera.project(end_x + nx * w_curb, end_y + ny * w_curb)
        right_end = self.camera.project(end_x - nx * w_curb, end_y - ny * w_curb)
        glow_col = (60, 240, 145, 90) if self.has_arrived else (235, 45, 50, 70)
        pygame.draw.line(self.dynamic_alpha_surf, glow_col, left_end, right_end, 10)

    def draw_dynamic_finish_line_core(self):
        if not self.path or len(self.path.x) < 2:
            return
        end_x, end_y = self.path.x[-1], self.path.y[-1]
        end_th = self.path.headings[-1]
        nx = -math.sin(end_th)
        ny = math.cos(end_th)
        w_curb = 4.8

        left_end = self.camera.project(end_x + nx * w_curb, end_y + ny * w_curb)
        right_end = self.camera.project(end_x - nx * w_curb, end_y - ny * w_curb)
        center_end = self.camera.project(end_x, end_y)

        beam_col = COLOR_TARGET_GREEN if self.has_arrived else (235, 45, 50)
        pygame.draw.line(self.screen, beam_col, left_end, right_end, 5)

        for pylon_pos in [left_end, right_end]:
            pygame.draw.circle(self.screen, (15, 18, 24), pylon_pos, 7)
            pygame.draw.circle(self.screen, beam_col, pylon_pos, 5)
            pygame.draw.circle(self.screen, (255, 255, 255), pylon_pos, 2)

        pygame.draw.circle(self.screen, beam_col, center_end, 6)
        pygame.draw.circle(self.screen, (255, 255, 255), center_end, 3)

    def draw_lighting(self):
        # Volumetric headlights and taillight glows rendered directly to persistent dynamic_alpha_surf
        light_surf = self.dynamic_alpha_surf

        th0 = self.state.theta0
        cos_t, sin_t = math.cos(th0), math.sin(th0)
        fx = self.state.x0 + self.dims.L0 * cos_t
        fy = self.state.y0 + self.dims.L0 * sin_t

        beam_len = 13.5
        hw = self.dims.cab_width * 0.44

        b_l = (fx + 0.85 * cos_t - hw * sin_t, fy + 0.85 * sin_t + hw * cos_t)
        b_r = (fx + 0.85 * cos_t + hw * sin_t, fy + 0.85 * sin_t - hw * cos_t)
        spread_w = 4.0
        f_l = (fx + (0.85 + beam_len) * cos_t - (hw + spread_w) * sin_t, fy + (0.85 + beam_len) * sin_t + (hw + spread_w) * cos_t)
        f_r = (fx + (0.85 + beam_len) * cos_t + (hw + spread_w) * sin_t, fy + (0.85 + beam_len) * sin_t - (hw + spread_w) * cos_t)

        pygame.draw.polygon(light_surf, (210, 235, 255, 12), [self.camera.project(*b_l), self.camera.project(*f_l), self.camera.project(*f_r), self.camera.project(*b_r)])
        core_len = beam_len * 0.60
        cf_l = (fx + (0.85 + core_len) * cos_t - (hw + spread_w*0.5) * sin_t, fy + (0.85 + core_len) * sin_t + (hw + spread_w*0.5) * cos_t)
        cf_r = (fx + (0.85 + core_len) * cos_t + (hw + spread_w*0.5) * sin_t, fy + (0.85 + core_len) * sin_t - (hw + spread_w*0.5) * cos_t)
        pygame.draw.polygon(light_surf, (230, 245, 255, 22), [self.camera.project(*b_l), self.camera.project(*cf_l), self.camera.project(*cf_r), self.camera.project(*b_r)])

        pygame.draw.circle(light_surf, (255, 255, 240, 200), self.camera.project(*b_l), 2)
        pygame.draw.circle(light_surf, (255, 255, 240, 200), self.camera.project(*b_r), 2)

        x2, y2, th2 = self.kinematics.get_trailer2_axle(self.state)
        for side in [-self.dims.trailer2_width * 0.44, self.dims.trailer2_width * 0.44]:
            tl_x = x2 - 1.48 * math.cos(th2) - side * math.sin(th2)
            tl_y = y2 - 1.48 * math.sin(th2) + side * math.cos(th2)
            tl_p = self.camera.project(tl_x, tl_y)
            pygame.draw.circle(light_surf, (235, 35, 45, 18), tl_p, 16)
            pygame.draw.circle(light_surf, (255, 45, 55, 50), tl_p, 7)
            pygame.draw.circle(light_surf, (255, 110, 120, 190), tl_p, 2)

    def draw_predictive_trajectories(self):
        if self.mode != "MANUAL" or self.is_off_track or self.has_arrived:
            return

        future_state = VehicleState(
            self.state.x0, self.state.y0, self.state.theta0,
            self.state.beta1, self.state.beta2, self.state.phi
        )
        ghost_dt = 0.20
        num_ghost_steps = 15
        pred_v = self.current_speed if abs(self.current_speed) > 0.1 else -1.5

        ghost_surf = self.dynamic_alpha_surf

        for step in range(1, num_ghost_steps + 1):
            future_state = self.kinematics.step_rk4(
                future_state, pred_v, self.target_steer, ghost_dt
            )
            if step in [3, 6, 9, 12, 15]:
                alpha = int(30 + (step / num_ghost_steps) * 85)
                polys = self.kinematics.get_body_polygons(future_state)
                for part, color in [
                    ('cab', (245, 175, 45)),
                    ('trailer1', (65, 160, 245)),
                    ('trailer2', (45, 225, 255))
                ]:
                    pts = [self.camera.project(p[0], p[1]) for p in polys[part]]
                    pygame.draw.polygon(ghost_surf, (*color, alpha), pts, 1)

    def draw_attention_rays(self):
        if self.mode != "AUTONOMOUS" and not self.ai_save_active:
            return

        if not self.show_rays or not self.path or self.attention_weights is None:
            return

        x0, y0, th0 = self.state.x0, self.state.y0, self.state.theta0
        cos_t, sin_t = math.cos(th0), math.sin(th0)
        cab_emitter = self.camera.project(x0 + self.dims.L0 * 0.4 * cos_t, y0 + self.dims.L0 * 0.4 * sin_t)

        x2, y2, _ = self.kinematics.get_trailer2_axle(self.state)
        nearest_idx = self.current_path_idx

        weights = self.attention_weights
        max_w = float(np.max(weights)) if len(weights) > 0 and np.max(weights) > 0 else 1.0

        ray_surf = self.dynamic_alpha_surf

        for k in range(min(64, len(weights))):
            idx = min(len(self.path.x) - 1, nearest_idx + k)
            st_x, st_y = self.path.x[idx], self.path.y[idx]
            st_s = self.camera.project(st_x, st_y)

            norm_w = float(weights[k] / max_w)
            if norm_w > 0.15:
                alpha_glow = int(norm_w * 55)
                pygame.draw.line(ray_surf, (45, 180, 225, alpha_glow), cab_emitter, st_s, 2)
                alpha_core = int(norm_w * 135)
                pygame.draw.line(ray_surf, (185, 235, 255, alpha_core), cab_emitter, st_s, 1)

                disc_r = max(1, int(norm_w * 4))
                pygame.draw.circle(ray_surf, (65, 195, 235, int(norm_w * 160)), st_s, disc_r)

    def draw_vehicle_shadows(self):
        x0, y0, th0 = self.state.x0, self.state.y0, self.state.theta0
        dims = self.dims
        x1, y1, th1 = self.kinematics.get_trailer1_axle(self.state)
        x2, y2, th2 = self.kinematics.get_trailer2_axle(self.state)

        cos_t0, sin_t0 = math.cos(th0), math.sin(th0)
        cos_t1, sin_t1 = math.cos(th1), math.sin(th1)
        cos_t2, sin_t2 = math.cos(th2), math.sin(th2)

        def get_poly(cx: float, cy: float, heading: float, length: float, width: float):
            hl, hw = length / 2.0, width / 2.0
            ch, sh = math.cos(heading), math.sin(heading)
            return [
                self.camera.project(cx + hl*ch - hw*sh, cy + hl*sh + hw*ch),
                self.camera.project(cx - hl*ch - hw*sh, cy - hl*sh + hw*ch),
                self.camera.project(cx - hl*ch + hw*sh, cy - hl*sh - hw*ch),
                self.camera.project(cx + hl*ch + hw*sh, cy + hl*sh - hw*ch)
            ]

        s_off = (4, 6)
        for cx, cy, th, l, w in [
            (x0 + 1.8 * cos_t0, y0 + 1.8 * sin_t0, th0, 5.3, dims.cab_width + 0.1),
            (x1 + 3.0 * cos_t1, y1 + 3.0 * sin_t1, th1, 6.8, dims.trailer1_width),
            (x2 + 2.5 * cos_t2, y2 + 2.5 * sin_t2, th2, 7.6, dims.trailer2_width)
        ]:
            poly = get_poly(cx, cy, th, l, w)
            s_poly = [(p[0] + s_off[0], p[1] + s_off[1]) for p in poly]
            pygame.draw.polygon(self.dynamic_alpha_surf, (10, 14, 20, 160), s_poly)

    def draw_truck_and_trailers(self):
        """
        Renders the multi-articulated vehicle matching the user's authentic top-down reference:
        - Sculpted Freightliner aerodynamic yellow cab with tapered hood, curved fenders, grille, and headlights
        - Curved tinted panoramic windshield with wipers
        - Sleeper cab roof with aerodynamic highlight contours and rear collar fairings hugging the trailer
        - Prominent aerodynamic side mirrors extending from A-pillars
        - Clean architectural silver-white trailers with perimeter frames, corner brackets, and rear cargo doors
        - Coiled pneumatic red/yellow air umbilicals
        """
        x0, y0, th0 = self.state.x0, self.state.y0, self.state.theta0
        dims = self.dims
        x1, y1, th1 = self.kinematics.get_trailer1_axle(self.state)
        x2, y2, th2 = self.kinematics.get_trailer2_axle(self.state)

        cos_t0, sin_t0 = math.cos(th0), math.sin(th0)
        cos_t1, sin_t1 = math.cos(th1), math.sin(th1)
        cos_t2, sin_t2 = math.cos(th2), math.sin(th2)

        # Helper for oriented 2D bounding boxes
        def get_poly(cx: float, cy: float, heading: float, length: float, width: float):
            hl, hw = length / 2.0, width / 2.0
            ch, sh = math.cos(heading), math.sin(heading)
            return [
                self.camera.project(cx + hl*ch - hw*sh, cy + hl*sh + hw*ch),
                self.camera.project(cx - hl*ch - hw*sh, cy - hl*sh + hw*ch),
                self.camera.project(cx - hl*ch + hw*sh, cy - hl*sh - hw*ch),
                self.camera.project(cx + hl*ch + hw*sh, cy + hl*sh - hw*ch)
            ]

        # 2. Detailed 2D Wheels with Rims & Tires
        def draw_detailed_wheel(center_x, center_y, heading):
            tire_pts = get_poly(center_x, center_y, heading, 0.95, 0.35)
            pygame.draw.polygon(self.screen, (20, 22, 28), tire_pts)
            pygame.draw.polygon(self.screen, (10, 12, 16), tire_pts, 1)
            rim_pts = get_poly(center_x, center_y, heading, 0.52, 0.16)
            pygame.draw.polygon(self.screen, (175, 182, 192), rim_pts)
            pygame.draw.polygon(self.screen, (80, 85, 95), rim_pts, 1)

        # Tractor Drive Tandem
        for rx_off in [0.0, -1.05]:
            rx = x0 + rx_off * cos_t0
            ry = y0 + rx_off * sin_t0
            for side in [-dims.cab_width * 0.47, dims.cab_width * 0.47]:
                draw_detailed_wheel(rx - side*sin_t0, ry + side*cos_t0, th0)

        # Trailer 1 Wheels
        for off in [-0.58, +0.58]:
            ax_x = x1 + off * cos_t1
            ax_y = y1 + off * sin_t1
            for side in [-dims.trailer1_width * 0.47, dims.trailer1_width * 0.47]:
                draw_detailed_wheel(ax_x - side*sin_t1, ax_y + side*cos_t1, th1)

        # Trailer 2 Wheels
        for off in [-0.58, +0.58]:
            ax_x = x2 + off * cos_t2
            ax_y = y2 + off * sin_t2
            for side in [-dims.trailer2_width * 0.47, dims.trailer2_width * 0.47]:
                draw_detailed_wheel(ax_x - side*sin_t2, ax_y + side*cos_t2, th2)

        # 3. Tractor Rear Chassis Catwalk & Fifth-Wheel Turntable
        catwalk_cx = x0 - 0.35 * cos_t0
        catwalk_cy = y0 - 0.35 * sin_t0
        catwalk_poly = get_poly(catwalk_cx, catwalk_cy, th0, 2.10, dims.cab_width * 0.68)
        pygame.draw.polygon(self.screen, (34, 40, 50), catwalk_poly)
        pygame.draw.polygon(self.screen, (60, 68, 80), catwalk_poly, 1)

        # Catwalk diamond-plate tread lines
        for off in [-0.8, -0.4, 0.0, 0.4, 0.8]:
            p1 = self.camera.project(catwalk_cx + off*cos_t0 - (dims.cab_width*0.30)*sin_t0,
                                     catwalk_cy + off*sin_t0 + (dims.cab_width*0.30)*cos_t0)
            p2 = self.camera.project(catwalk_cx + off*cos_t0 + (dims.cab_width*0.30)*sin_t0,
                                     catwalk_cy + off*sin_t0 - (dims.cab_width*0.30)*cos_t0)
            pygame.draw.line(self.screen, (46, 54, 66), p1, p2, 1)

        # Tractor Fifth-Wheel Turntable Base Plate directly over drive axle (x0, y0)
        fw0_poly = get_poly(x0, y0, th0, 0.90, 0.82)
        pygame.draw.polygon(self.screen, (20, 24, 30), fw0_poly)
        pygame.draw.polygon(self.screen, (85, 92, 104), fw0_poly, 1)

        # Horseshoe plate with rearward throat cutout
        hp0 = self.camera.project(x0, y0)
        pygame.draw.circle(self.screen, (38, 44, 54), hp0, 11)
        pygame.draw.circle(self.screen, (190, 198, 210), hp0, 7)
        th_e = self.camera.project(x0 - 0.32*cos_t0, y0 - 0.32*sin_t0)
        pygame.draw.line(self.screen, (15, 18, 24), hp0, th_e, 4)

        # ---------------------------------------------------------------------
        # REFINED B-DOUBLE CHASSIS EXTENSION & INTER-TRAILER ARTICULATION
        # ---------------------------------------------------------------------
        t1_box_l = 5.20
        t1_front_overhang = 0.75
        t1_cx = x0 - (t1_box_l/2 - t1_front_overhang) * cos_t1
        t1_cy = y0 - (t1_box_l/2 - t1_front_overhang) * sin_t1

        # A. Trailer 1 Under-Nose Kingpin Skid Plate (resting on tractor fifth wheel)
        t1_skid_cx = x0 + 0.15 * cos_t1
        t1_skid_cy = y0 + 0.15 * sin_t1
        t1_skid_poly = get_poly(t1_skid_cx, t1_skid_cy, th1, 1.05, dims.trailer1_width * 0.72)
        pygame.draw.polygon(self.screen, (28, 34, 44), t1_skid_poly)
        pygame.draw.polygon(self.screen, (85, 94, 106), t1_skid_poly, 1)

        # B. Trailer 1 Extended Rear Chassis Frame & Treadplate Platform (extends behind box over axle)
        deck_cx = x1 + 0.30 * cos_t1
        deck_cy = y1 + 0.30 * sin_t1
        deck_l = 2.40
        deck_w = dims.trailer1_width * 0.78
        deck_poly = get_poly(deck_cx, deck_cy, th1, deck_l, deck_w)
        pygame.draw.polygon(self.screen, (32, 38, 48), deck_poly)
        pygame.draw.polygon(self.screen, (60, 70, 85), deck_poly, 2)

        # Steel cross-members / tread pattern lines across deck
        for d_off in [-0.8, -0.4, 0.0, 0.4, 0.8]:
            p1 = self.camera.project(deck_cx + d_off*cos_t1 - (deck_w/2 - 0.1)*sin_t1,
                                     deck_cy + d_off*sin_t1 + (deck_w/2 - 0.1)*cos_t1)
            p2 = self.camera.project(deck_cx + d_off*cos_t1 + (deck_w/2 - 0.1)*sin_t1,
                                     deck_cy + d_off*sin_t1 - (deck_w/2 - 0.1)*cos_t1)
            pygame.draw.line(self.screen, (45, 54, 66), p1, p2, 1)

        # Tandem wheel mudguard fenders on Trailer 1 extension deck
        for side in [-dims.trailer1_width * 0.46, dims.trailer1_width * 0.46]:
            fender_poly = get_poly(x1 - side*sin_t1, y1 + side*cos_t1, th1, 2.10, 0.28)
            pygame.draw.polygon(self.screen, (24, 28, 36), fender_poly)
            pygame.draw.polygon(self.screen, (75, 82, 94), fender_poly, 1)

        # C. Trailer 1 Fifth-Wheel Mounting Subframe & Turntable Base Plate
        fw1_poly = get_poly(x1, y1, th1, 0.95, 0.85)
        pygame.draw.polygon(self.screen, (18, 22, 28), fw1_poly)
        pygame.draw.polygon(self.screen, (90, 98, 110), fw1_poly, 1)

        # Fifth-wheel horseshoe plate with rear-facing throat cutout
        hp1 = self.camera.project(x1, y1)
        pygame.draw.circle(self.screen, (38, 44, 54), hp1, 11)
        pygame.draw.circle(self.screen, (190, 198, 210), hp1, 7)
        throat1_e = self.camera.project(x1 - 0.35*cos_t1, y1 - 0.35*sin_t1)
        pygame.draw.line(self.screen, (15, 18, 24), hp1, throat1_e, 4)

        # D. Trailer 1 Box with Chamfered Front Corners (matches reference!)
        hw1 = dims.trailer1_width / 2.0
        hl1 = t1_box_l / 2.0
        chamfer1 = 0.35

        t1_corners_local = [
            (hl1 - chamfer1,  hw1),
            (hl1,             hw1 - chamfer1),
            (hl1,            -hw1 + chamfer1),
            (hl1 - chamfer1, -hw1),
            (-hl1,           -hw1),
            (-hl1,            hw1)
        ]
        t1_pts = [self.camera.project(t1_cx + lx*cos_t1 - ly*sin_t1, t1_cy + lx*sin_t1 + ly*cos_t1) for lx, ly in t1_corners_local]
        pygame.draw.polygon(self.screen, (226, 229, 233), t1_pts)
        pygame.draw.polygon(self.screen, (148, 156, 166), t1_pts, 2)

        t1_inner_corners = [
            (hl1 - chamfer1 - 0.08,  hw1 - 0.09),
            (hl1 - 0.09,             hw1 - chamfer1 - 0.08),
            (hl1 - 0.09,            -hw1 + chamfer1 + 0.08),
            (hl1 - chamfer1 - 0.08, -hw1 + 0.09),
            (-hl1 + 0.09,           -hw1 + 0.09),
            (-hl1 + 0.09,            hw1 - 0.09)
        ]
        t1_inner_pts = [self.camera.project(t1_cx + lx*cos_t1 - ly*sin_t1, t1_cy + lx*sin_t1 + ly*cos_t1) for lx, ly in t1_inner_corners]
        pygame.draw.polygon(self.screen, (235, 238, 242), t1_inner_pts)
        pygame.draw.polygon(self.screen, (180, 186, 195), t1_inner_pts, 1)

        # Amber clearance lights on Trailer 1 front
        for side in [-hw1 + 0.15, hw1 - 0.15]:
            cl_p = self.camera.project(t1_cx + hl1*cos_t1 - side*sin_t1, t1_cy + hl1*sin_t1 + side*cos_t1)
            pygame.draw.circle(self.screen, (255, 180, 40), cl_p, 2)

        # Trailer 1 Rear Wall: Cargo Door frame line
        t1_rear_l = self.camera.project(t1_cx - hl1*cos_t1 - hw1*sin_t1,
                                        t1_cy - hl1*sin_t1 + hw1*cos_t1)
        t1_rear_r = self.camera.project(t1_cx - hl1*cos_t1 + hw1*sin_t1,
                                        t1_cy - hl1*sin_t1 - hw1*cos_t1)
        pygame.draw.line(self.screen, (130, 138, 148), t1_rear_l, t1_rear_r, 3)

        # E. Trailer 2: Rear Semi-Trailer with Under-Nose Bolster & Aerodynamic Chamfered Corners
        t2_l = 7.70
        t2_front_overhang = 0.85
        t2_cx = x2 + (dims.L2 + t2_front_overhang - t2_l/2) * cos_t2
        t2_cy = y2 + (dims.L2 + t2_front_overhang - t2_l/2) * sin_t2

        # Under-nose Kingpin Skid Plate (Dark circular bolster resting on fifth wheel)
        skid_cx = x1 + 0.15 * cos_t2
        skid_cy = y1 + 0.15 * sin_t2
        skid_poly = get_poly(skid_cx, skid_cy, th2, 1.10, dims.trailer2_width * 0.72)
        pygame.draw.polygon(self.screen, (28, 34, 44), skid_poly)
        pygame.draw.polygon(self.screen, (85, 94, 106), skid_poly, 1)

        # Trailer 2 Body with Chamfered Front Aerodynamic Corners (Zero awkward polygon clipping)
        hw2 = dims.trailer2_width / 2.0
        hl2 = t2_l / 2.0
        ch2, sh2 = cos_t2, sin_t2
        chamfer2 = 0.35

        t2_corners_local = [
            (hl2 - chamfer2,  hw2),
            (hl2,             hw2 - chamfer2),
            (hl2,            -hw2 + chamfer2),
            (hl2 - chamfer2, -hw2),
            (-hl2,           -hw2),
            (-hl2,            hw2)
        ]
        t2_pts = [self.camera.project(t2_cx + lx*ch2 - ly*sh2, t2_cy + lx*sh2 + ly*ch2) for lx, ly in t2_corners_local]
        pygame.draw.polygon(self.screen, (226, 229, 233), t2_pts)
        pygame.draw.polygon(self.screen, (148, 156, 166), t2_pts, 2)

        t2_inner_corners = [
            (hl2 - chamfer2 - 0.08,  hw2 - 0.09),
            (hl2 - 0.09,             hw2 - chamfer2 - 0.08),
            (hl2 - 0.09,            -hw2 + chamfer2 + 0.08),
            (hl2 - chamfer2 - 0.08, -hw2 + 0.09),
            (-hl2 + 0.09,           -hw2 + 0.09),
            (-hl2 + 0.09,            hw2 - 0.09)
        ]
        t2_inner_pts = [self.camera.project(t2_cx + lx*ch2 - ly*sh2, t2_cy + lx*sh2 + ly*ch2) for lx, ly in t2_inner_corners]
        pygame.draw.polygon(self.screen, (235, 238, 242), t2_inner_pts)
        pygame.draw.polygon(self.screen, (180, 186, 195), t2_inner_pts, 1)

        # Front Bulkhead Amber Clearance Lights on Trailer 2
        for side in [-hw2 + 0.15, hw2 - 0.15]:
            cl_p = self.camera.project(t2_cx + hl2*ch2 - side*sh2, t2_cy + hl2*sh2 + side*ch2)
            pygame.draw.circle(self.screen, (255, 180, 40), cl_p, 2)

        # Rear Cargo Doors: Split line and vertical lock rods
        t2_rear_s = self.camera.project(t2_cx - hl2*cos_t2, t2_cy - hl2*sin_t2)
        t2_rear_e = self.camera.project(t2_cx - (hl2 - 0.35)*cos_t2, t2_cy - (hl2 - 0.35)*sin_t2)
        pygame.draw.line(self.screen, (130, 136, 145), t2_rear_s, t2_rear_e, 2)

        # Rear Bumper Bar & Taillights
        t2_bmp_poly = get_poly(t2_cx - (hl2 + 0.08)*cos_t2, t2_cy - (hl2 + 0.08)*sin_t2, th2, 0.16, dims.trailer2_width - 0.08)
        pygame.draw.polygon(self.screen, (40, 44, 52), t2_bmp_poly)
        for side in [-dims.trailer2_width * 0.44, dims.trailer2_width * 0.44]:
            tl_p = self.camera.project(t2_cx - (hl2 + 0.08)*cos_t2 - side*sin_t2,
                                       t2_cy - (hl2 + 0.08)*sin_t2 + side*cos_t2)
            pygame.draw.circle(self.screen, (255, 45, 55), tl_p, 3)

        # ---------------------------------------------------------------------
        # G. TRACTOR CAB: EXACT REPLICA OF USER'S FREIGHTLINER REFERENCE
        # Color: Warm Golden Amber (248, 178, 48) with sculpted aerodynamic contours
        # ---------------------------------------------------------------------
        cab_cx = x0 + 1.80 * cos_t0
        cab_cy = y0 + 1.80 * sin_t0

        # 1. Fuel Tanks & Chassis Skirts flanking chassis
        for side in [-dims.cab_width * 0.48, dims.cab_width * 0.48]:
            ft_cx = x0 + 1.35 * cos_t0 - side * sin_t0
            ft_cy = y0 + 1.35 * sin_t0 + side * cos_t0
            ft_poly = get_poly(ft_cx, ft_cy, th0, 1.65, 0.38)
            pygame.draw.polygon(self.screen, (220, 155, 36), ft_poly)
            pygame.draw.polygon(self.screen, (175, 120, 24), ft_poly, 1)

        # 2. Rear Cab Side Fairing Extenders / Collar (Wings that reach back towards trailer)
        for side in [-dims.cab_width * 0.44, dims.cab_width * 0.44]:
            wing_cx = x0 + 0.65 * cos_t0 - side * sin_t0
            wing_cy = y0 + 0.65 * sin_t0 + side * cos_t0
            wing_poly = get_poly(wing_cx, wing_cy, th0, 0.45, 0.22)
            pygame.draw.polygon(self.screen, (238, 168, 42), wing_poly)
            pygame.draw.polygon(self.screen, (185, 128, 28), wing_poly, 1)

        # 3. Main Sleeper Cabin Roof (Sculpted aerodynamic yellow body)
        cab_poly = get_poly(cab_cx, cab_cy, th0, 1.95, dims.cab_width)
        pygame.draw.polygon(self.screen, (248, 178, 48), cab_poly)
        pygame.draw.polygon(self.screen, (198, 138, 32), cab_poly, 2)

        # Aerodynamic Roof Contours (The raised center ridge & side flow ridges from reference)
        r_center_poly = get_poly(cab_cx - 0.10 * cos_t0, cab_cy - 0.10 * sin_t0, th0, 1.35, dims.cab_width * 0.36)
        pygame.draw.polygon(self.screen, (255, 195, 80), r_center_poly)
        pygame.draw.polygon(self.screen, (225, 155, 38), r_center_poly, 1)

        # Side roof flow lines
        for r_side in [-dims.cab_width * 0.32, dims.cab_width * 0.32]:
            p_s = self.camera.project(cab_cx + 0.80 * cos_t0 - r_side * sin_t0, cab_cy + 0.80 * sin_t0 + r_side * cos_t0)
            p_e = self.camera.project(cab_cx - 0.80 * cos_t0 - r_side * sin_t0, cab_cy - 0.80 * sin_t0 + r_side * cos_t0)
            pygame.draw.line(self.screen, (215, 148, 35), p_s, p_e, 2)

        # 4. Panoramic Curved Front Windshield & Wipers
        ws_cx = x0 + 2.72 * cos_t0
        ws_cy = y0 + 2.72 * sin_t0
        ws_hw = (dims.cab_width - 0.20) / 2.0
        ws_poly = [
            self.camera.project(ws_cx + 0.20*cos_t0 - (ws_hw - 0.15)*sin_t0, ws_cy + 0.20*sin_t0 + (ws_hw - 0.15)*cos_t0),
            self.camera.project(ws_cx - 0.16*cos_t0 - ws_hw*sin_t0, ws_cy - 0.16*sin_t0 + ws_hw*cos_t0),
            self.camera.project(ws_cx - 0.16*cos_t0 + ws_hw*sin_t0, ws_cy - 0.16*sin_t0 - ws_hw*cos_t0),
            self.camera.project(ws_cx + 0.20*cos_t0 + (ws_hw - 0.15)*sin_t0, ws_cy + 0.20*sin_t0 - (ws_hw - 0.15)*cos_t0)
        ]
        pygame.draw.polygon(self.screen, (26, 34, 46), ws_poly)
        pygame.draw.polygon(self.screen, (14, 18, 26), ws_poly, 1)

        # Dual Windshield Wipers parked at bottom edge
        for w_side in [-0.42, 0.42]:
            wp_s = self.camera.project(ws_cx - 0.12*cos_t0 - w_side*sin_t0, ws_cy - 0.12*sin_t0 + w_side*cos_t0)
            wp_e = self.camera.project(ws_cx + 0.12*cos_t0 - (w_side + 0.25)*sin_t0, ws_cy + 0.12*sin_t0 + (w_side + 0.25)*cos_t0)
            pygame.draw.line(self.screen, (10, 12, 16), wp_s, wp_e, 2)

        # 5. Aerodynamic Tapered Front Hood
        hood_cx = x0 + 3.65 * cos_t0
        hood_cy = y0 + 3.65 * sin_t0
        hood_l = 1.70
        hood_w_rear = dims.cab_width * 0.90
        hood_w_front = dims.cab_width * 0.72

        h_rear_l = (hood_cx - (hood_l/2)*cos_t0 - (hood_w_rear/2)*sin_t0, hood_cy - (hood_l/2)*sin_t0 + (hood_w_rear/2)*cos_t0)
        h_rear_r = (hood_cx - (hood_l/2)*cos_t0 + (hood_w_rear/2)*sin_t0, hood_cy - (hood_l/2)*sin_t0 - (hood_w_rear/2)*cos_t0)
        h_front_l = (hood_cx + (hood_l/2)*cos_t0 - (hood_w_front/2)*sin_t0, hood_cy + (hood_l/2)*sin_t0 + (hood_w_front/2)*cos_t0)
        h_front_r = (hood_cx + (hood_l/2)*cos_t0 + (hood_w_front/2)*sin_t0, hood_cy + (hood_l/2)*sin_t0 - (hood_w_front/2)*cos_t0)

        hood_poly = [self.camera.project(*h_rear_l), self.camera.project(*h_rear_r),
                     self.camera.project(*h_front_r), self.camera.project(*h_front_l)]
        pygame.draw.polygon(self.screen, (248, 178, 48), hood_poly)
        pygame.draw.polygon(self.screen, (198, 138, 32), hood_poly, 2)

        # Hood Center Crease Line
        hc_s = self.camera.project(hood_cx - 0.75 * cos_t0, hood_cy - 0.75 * sin_t0)
        hc_e = self.camera.project(hood_cx + 0.80 * cos_t0, hood_cy + 0.80 * sin_t0)
        pygame.draw.line(self.screen, (255, 202, 95), hc_s, hc_e, 2)

        # Hood Curved Fender Ridges (dual flowing crease lines leading to headlights)
        for f_side in [-0.68, 0.68]:
            f_s = self.camera.project(hood_cx - 0.75 * cos_t0 - (f_side * 1.15) * sin_t0, hood_cy - 0.75 * sin_t0 + (f_side * 1.15) * cos_t0)
            f_m = self.camera.project(hood_cx + 0.10 * cos_t0 - (f_side * 1.05) * sin_t0, hood_cy + 0.10 * sin_t0 + (f_side * 1.05) * cos_t0)
            f_e = self.camera.project(hood_cx + 0.78 * cos_t0 - (f_side * 0.85) * sin_t0, hood_cy + 0.78 * sin_t0 + (f_side * 0.85) * cos_t0)
            pygame.draw.lines(self.screen, (215, 148, 35), False, [f_s, f_m, f_e], 2)

        # 6. Aerodynamic Headlight Assemblies at Front Fender Corners (Clear lenses with amber markers)
        for side in [-hood_w_front * 0.46, hood_w_front * 0.46]:
            hl_cx = hood_cx + 0.80 * cos_t0 - side * sin_t0
            hl_cy = hood_cy + 0.80 * sin_t0 + side * cos_t0
            hl_poly = get_poly(hl_cx, hl_cy, th0, 0.28, 0.26)
            pygame.draw.polygon(self.screen, (230, 242, 255), hl_poly)
            pygame.draw.polygon(self.screen, (70, 75, 85), hl_poly, 1)
            amb_p = self.camera.project(hl_cx - 0.05 * cos_t0 - (side * 0.08) * sin_t0,
                                        hl_cy - 0.05 * sin_t0 + (side * 0.08) * cos_t0)
            pygame.draw.circle(self.screen, (255, 175, 30), amb_p, 2)

        # 7. Chrome Front Grille Inset at Nose
        grille_cx = hood_cx + 0.88 * cos_t0
        grille_cy = hood_cy + 0.88 * sin_t0
        grille_poly = get_poly(grille_cx, grille_cy, th0, 0.15, hood_w_front * 0.65)
        pygame.draw.polygon(self.screen, (50, 55, 65), grille_poly)
        pygame.draw.polygon(self.screen, (225, 230, 240), grille_poly, 1)

        # 8. Front Curved Aerodynamic Bumper
        bumper_cx = hood_cx + 0.94 * cos_t0
        bumper_cy = hood_cy + 0.94 * sin_t0
        bumper_poly = get_poly(bumper_cx, bumper_cy, th0, 0.16, hood_w_front * 0.96)
        pygame.draw.polygon(self.screen, (235, 168, 42), bumper_poly)
        pygame.draw.polygon(self.screen, (185, 128, 28), bumper_poly, 1)

        # 9. Aerodynamic Side Mirrors (Extending from A-pillars, yellow brackets, mirror glass)
        for side in [-dims.cab_width * 0.54, dims.cab_width * 0.54]:
            arm_start = self.camera.project(x0 + 2.70 * cos_t0 - (side * 0.82) * sin_t0,
                                            y0 + 2.70 * sin_t0 + (side * 0.82) * cos_t0)
            arm_end   = self.camera.project(x0 + 2.64 * cos_t0 - (side * 1.18) * sin_t0,
                                            y0 + 2.64 * sin_t0 + (side * 1.18) * cos_t0)
            pygame.draw.line(self.screen, (225, 155, 36), arm_start, arm_end, 2)

            m_poly = get_poly(x0 + 2.64 * cos_t0 - (side * 1.18) * sin_t0,
                              y0 + 2.64 * sin_t0 + (side * 1.18) * cos_t0,
                              th0, 0.42, 0.16)
            pygame.draw.polygon(self.screen, (248, 178, 48), m_poly)
            pygame.draw.polygon(self.screen, (185, 128, 28), m_poly, 1)

        # 10. Front Steer Wheels
        steer_th = th0 + self.state.phi
        fx = x0 + dims.L0 * cos_t0
        fy = y0 + dims.L0 * sin_t0

        for side in [-dims.cab_width * 0.52, dims.cab_width * 0.52]:
            wx = fx - side * sin_t0
            wy = fy + side * cos_t0
            draw_detailed_wheel(wx, wy, steer_th)


    # ==========================================================================
    # THEMATIC FROSTED GLASS HUD WITH REFINED DISTRIBUTION PLOT
    # ==========================================================================

    def draw_hud(self):
        s = self.ui_scale

        # 1. Top Minimal Status Pill Bar (Clean, with Interactive Speed Warp & Sample Prior Buttons)
        bar_w, bar_h = int(600 * s), int(38 * s)
        bar_x = (self.width - bar_w) // 2
        bar_y = int(16 * s)
        bar_surf = pygame.Surface((bar_w, bar_h), pygame.SRCALPHA)
        pygame.draw.rect(bar_surf, COLOR_HUD_BG, (0, 0, bar_w, bar_h), border_radius=int(19 * s))
        pygame.draw.rect(bar_surf, COLOR_HUD_BORDER, (0, 0, bar_w, bar_h), 1, border_radius=int(19 * s))

        gear_str = "REV" if self.current_speed < -0.1 else ("FWD" if self.current_speed > 0.1 else "PARK")
        bar_surf.blit(self.font_mono_bold.render(f"GEAR: {gear_str}", True, COLOR_ACCENT_AMBER), (int(16 * s), int(10 * s)))
        bar_surf.blit(self.font_mono.render(f"v = {self.current_speed:+.1f} m/s", True, COLOR_TEXT_WHITE), (int(104 * s), int(10 * s)))
        bar_surf.blit(self.font_mono.render(f"{self.inference_time_ms:.1f}ms", True, COLOR_TEXT_DIM), (int(212 * s), int(10 * s)))

        # 1a. Interactive Clickable Speed Button
        btn_w_spd, btn_h = int(122 * s), int(30 * s)
        btn_spd_local = pygame.Rect(bar_w - int(282 * s), int(4 * s), btn_w_spd, btn_h)
        self.speed_btn_rect = pygame.Rect(bar_x + btn_spd_local.x, bar_y + btn_spd_local.y, btn_spd_local.w, btn_spd_local.h)
        mouse_pos = pygame.mouse.get_pos()
        is_spd_hover = self.speed_btn_rect.collidepoint(mouse_pos)

        spd_bg = (38, 58, 82, 240) if is_spd_hover else ((24, 44, 64, 210) if self.time_multiplier > 1 else (22, 28, 38, 180))
        spd_border = (0, 230, 255) if is_spd_hover else (COLOR_ACCENT_CYAN if self.time_multiplier > 1 else (55, 70, 90))
        pygame.draw.rect(bar_surf, spd_bg, btn_spd_local, border_radius=int(15 * s))
        pygame.draw.rect(bar_surf, spd_border, btn_spd_local, 1, border_radius=int(15 * s))

        spd_col = (255, 255, 255) if is_spd_hover else (COLOR_ACCENT_CYAN if self.time_multiplier > 1 else (170, 182, 198))
        spd_txt = self.font_mono_bold.render(f"SPEED: {self.time_multiplier}x [T]", True, spd_col)
        bar_surf.blit(spd_txt, (btn_spd_local.centerx - spd_txt.get_width() // 2, int(7 * s)))

        # 1b. Interactive Clickable Sample Random Path Button (from Prior)
        btn_w_smp = int(144 * s)
        btn_smp_local = pygame.Rect(bar_w - int(150 * s), int(4 * s), btn_w_smp, btn_h)
        self.sample_btn_rect = pygame.Rect(bar_x + btn_smp_local.x, bar_y + btn_smp_local.y, btn_smp_local.w, btn_smp_local.h)
        is_smp_hover = self.sample_btn_rect.collidepoint(mouse_pos)

        smp_bg = (42, 64, 88, 240) if is_smp_hover else (24, 38, 54, 200)
        smp_border = (255, 215, 65) if is_smp_hover else (65, 95, 130)
        pygame.draw.rect(bar_surf, smp_bg, btn_smp_local, border_radius=int(15 * s))
        pygame.draw.rect(bar_surf, smp_border, btn_smp_local, 1, border_radius=int(15 * s))

        smp_col = (255, 255, 255) if is_smp_hover else COLOR_ACCENT_AMBER
        smp_txt = self.font_mono_bold.render("SAMPLE PATH [P]", True, smp_col)
        bar_surf.blit(smp_txt, (btn_smp_local.centerx - smp_txt.get_width() // 2, int(7 * s)))

        self.screen.blit(bar_surf, (bar_x, bar_y))

        # 2. Sleek Simplified Stability Margin Card (Lower Right Corner)
        sm_w, sm_h = int(240 * s), int(64 * s)
        sm_surf = pygame.Surface((sm_w, sm_h), pygame.SRCALPHA)
        pygame.draw.rect(sm_surf, COLOR_HUD_BG, (0, 0, sm_w, sm_h), border_radius=int(10 * s))
        pygame.draw.rect(sm_surf, COLOR_HUD_BORDER, (0, 0, sm_w, sm_h), 1, border_radius=int(10 * s))

        pad_x = int(14 * s)
        meter_w = sm_w - 2 * pad_x

        rho = self.kinematics.jackknife_proximity(self.state)
        is_jk = self.kinematics.is_jackknifed(self.state)
        stab_str = "JACKKNIFED" if is_jk else f"{rho * 100:.0f}%"
        stab_col = COLOR_TARGET_GREEN if rho > 0.45 else (COLOR_ACCENT_AMBER if rho > 0.20 else (225, 55, 60))

        sm_surf.blit(self.font_title.render("STABILITY MARGIN", True, COLOR_TEXT_DIM), (pad_x, int(10 * s)))
        stab_txt = self.font_mono_bold.render(stab_str, True, stab_col)
        sm_surf.blit(stab_txt, (sm_w - pad_x - stab_txt.get_width(), int(10 * s)))

        meter_y, meter_h = int(34 * s), max(5, int(9 * s))
        pygame.draw.rect(sm_surf, (24, 30, 40), (pad_x, meter_y, meter_w, meter_h), border_radius=int(4 * s))
        fill_w = int(meter_w * rho)
        if fill_w > 0:
            pygame.draw.rect(sm_surf, stab_col, (pad_x, meter_y, fill_w, meter_h), border_radius=int(4 * s))

        self.screen.blit(sm_surf, (self.width - sm_w - int(20 * s), self.height - sm_h - int(24 * s)))

        # 3. Refined Bottom-Left Steering Distribution Plot Card (Clean, strictly bounded)
        p_w, p_h = int(360 * s), int(132 * s)
        p_surf = pygame.Surface((p_w, p_h), pygame.SRCALPHA)
        pygame.draw.rect(p_surf, COLOR_HUD_BG, (0, 0, p_w, p_h), border_radius=int(12 * s))
        pygame.draw.rect(p_surf, COLOR_HUD_BORDER, (0, 0, p_w, p_h), 1, border_radius=int(12 * s))

        chart_x = pad_x
        chart_y = int(42 * s)
        chart_w = p_w - 2 * pad_x
        chart_h = int(54 * s)

        if self.mode == "AUTONOMOUS" or self.ai_save_active:
            angles_deg = np.linspace(-35, 35, 15)
            probs = self.action_probs
            num_bins = len(probs)
            bar_width = int(chart_w / num_bins) - 3

            pygame.draw.line(p_surf, (35, 45, 60), (chart_x, chart_y + chart_h), (chart_x + chart_w, chart_y + chart_h), 1)

            peak_bin = int(np.argmax(probs))
            expected_deg = float(np.sum(probs * angles_deg))
            max_p = max(0.20, float(np.max(probs)))

            p_surf.blit(self.font_title.render("STEERING DISTRIBUTION", True, COLOR_TEXT_WHITE), (pad_x, int(12 * s)))
            exp_txt = self.font_mono_bold.render(f"E[φ]={expected_deg:+.1f}°", True, COLOR_STATUS_GREEN)
            p_surf.blit(exp_txt, (p_w - pad_x - exp_txt.get_width(), int(12 * s)))

            for b in range(num_bins):
                bx = chart_x + b * (bar_width + 3)
                b_prob = float(probs[b])
                bh = max(2, min(chart_h - 4, int((b_prob / max_p) * (chart_h - 8))))
                by = chart_y + chart_h - bh

                if b == peak_bin:
                    pygame.draw.rect(p_surf, COLOR_ACCENT_CYAN, (bx, by, bar_width, bh), border_radius=int(2 * s))
                    pygame.draw.rect(p_surf, (185, 235, 255), (bx, by, bar_width, bh), 1, border_radius=int(2 * s))
                else:
                    b_col = (48, 76, 108) if b_prob > 0.05 else (28, 42, 60)
                    pygame.draw.rect(p_surf, b_col, (bx, by, bar_width, bh), border_radius=int(2 * s))

            needle_frac = (expected_deg - (-35.0)) / 70.0
            needle_x = chart_x + int(needle_frac * chart_w)
            needle_x = max(chart_x, min(chart_x + chart_w, needle_x))
            pygame.draw.line(p_surf, COLOR_STATUS_GREEN, (needle_x, chart_y + 4), (needle_x, chart_y + chart_h + 2), 2)
            pygame.draw.circle(p_surf, COLOR_STATUS_GREEN, (needle_x, chart_y + 4), max(2, int(3 * s)))
        else:
            p_surf.blit(self.font_title.render("MANUAL STEERING", True, COLOR_TEXT_WHITE), (pad_x, int(12 * s)))

            pygame.draw.rect(p_surf, (22, 28, 38), (chart_x, chart_y + int(14 * s), chart_w, int(20 * s)), border_radius=int(6 * s))
            pygame.draw.line(p_surf, (55, 70, 90), (chart_x + chart_w//2, chart_y + int(10 * s)), (chart_x + chart_w//2, chart_y + int(38 * s)), 2)

            steer_deg = math.degrees(self.state.phi)
            s_frac = (steer_deg - (-35.0)) / 70.0
            sx = chart_x + int(s_frac * chart_w)
            sx = max(chart_x + 4, min(chart_x + chart_w - 4, sx))
            pygame.draw.circle(p_surf, COLOR_ACCENT_AMBER, (sx, chart_y + int(24 * s)), max(4, int(7 * s)))
            pygame.draw.circle(p_surf, (255, 255, 255), (sx, chart_y + int(24 * s)), max(2, int(2 * s)))

            sw_cx, sw_cy, sw_r = p_w - int(120 * s), int(20 * s), int(11 * s)
            pygame.draw.circle(p_surf, (45, 55, 72), (sw_cx, sw_cy), sw_r, 2)
            pygame.draw.circle(p_surf, (70, 85, 110), (sw_cx, sw_cy), max(2, int(3 * s)))
            wheel_rot = -self.state.phi * 2.2
            for spoke_ang in [wheel_rot, wheel_rot + 2.1, wheel_rot - 2.1]:
                sp_x = sw_cx + (sw_r - 2) * math.sin(spoke_ang)
                sp_y = sw_cy - (sw_r - 2) * math.cos(spoke_ang)
                pygame.draw.line(p_surf, (160, 180, 205), (sw_cx, sw_cy), (int(sp_x), int(sp_y)), 2)
            top_x = sw_cx + (sw_r) * math.sin(wheel_rot)
            top_y = sw_cy - (sw_r) * math.cos(wheel_rot)
            pygame.draw.circle(p_surf, COLOR_ACCENT_AMBER, (int(top_x), int(top_y)), max(2, int(2 * s)))

            m_txt = self.font_mono_bold.render(f"φ={steer_deg:+.1f}°", True, COLOR_ACCENT_AMBER)
            p_surf.blit(m_txt, (p_w - pad_x - m_txt.get_width(), int(12 * s)))

        # Clean Axis Labels
        p_surf.blit(self.font_sm.render("-35°", True, COLOR_TEXT_DIM), (chart_x, chart_y + chart_h + int(5 * s)))
        lbl_0 = self.font_sm.render("0°", True, COLOR_TEXT_DIM)
        p_surf.blit(lbl_0, (chart_x + chart_w // 2 - lbl_0.get_width() // 2, chart_y + chart_h + int(5 * s)))
        lbl_p35 = self.font_sm.render("+35°", True, COLOR_TEXT_DIM)
        p_surf.blit(lbl_p35, (chart_x + chart_w - lbl_p35.get_width(), chart_y + chart_h + int(5 * s)))

        self.screen.blit(p_surf, (int(24 * s), self.height - p_h - int(24 * s)))

        # 4. Right Controls Card (Compact, perfectly fitting within boundary)
        if self.show_help:
            h_w, h_h = int(250 * s), int(238 * s)
            h_surf = pygame.Surface((h_w, h_h), pygame.SRCALPHA)
            pygame.draw.rect(h_surf, COLOR_HUD_BG, (0, 0, h_w, h_h), border_radius=int(10 * s))
            pygame.draw.rect(h_surf, COLOR_HUD_BORDER, (0, 0, h_w, h_h), 1, border_radius=int(10 * s))

            h_surf.blit(self.font_title.render("CONTROLS", True, COLOR_TEXT_WHITE), (int(14 * s), int(12 * s)))

            controls = [
                ("[W / S]", "Gas / Reverse"),
                ("[A / D]", "Manual Steer"),
                ("[M]", "Auto / Manual"),
                ("[T]", "Speed 1x/8x/16x"),
                ("[P]", "Sample Prior"),
                ("[Space]", "Pause / Resume"),
                ("[R]", "Reset to Start"),
                ("[1 - 5]", "Preset Tracks"),
                ("[L-Drag]", "Draw on Yard"),
            ]

            line_h = int(21 * s)
            desc_x = int(92 * s)
            for i, (k_tag, desc) in enumerate(controls):
                h_surf.blit(self.font_mono_bold.render(k_tag, True, COLOR_ACCENT_AMBER), (int(14 * s), int(34 * s) + i * line_h))
                h_surf.blit(self.font_mono.render(desc, True, COLOR_TEXT_DIM), (desc_x, int(34 * s) + i * line_h))

            self.screen.blit(h_surf, (self.width - h_w - int(20 * s), int(20 * s)))

        # 5. Drawing & Prominent Center Messages
        if self.is_drawing:
            msg = self.font_mono_bold.render("DRAWING PATH... (Physical R >= 10.0m)", True, COLOR_ACCENT_CYAN)
            self.screen.blit(msg, (self.width // 2 - msg.get_width() // 2, int(45 * s)))
        elif self.has_arrived:
            banner_w, banner_h = int(490 * s), int(100 * s)
            banner_surf = pygame.Surface((banner_w, banner_h), pygame.SRCALPHA)
            pygame.draw.rect(banner_surf, (12, 34, 22, 245), (0, 0, banner_w, banner_h), border_radius=int(14 * s))
            pygame.draw.rect(banner_surf, (45, 225, 125), (0, 0, banner_w, banner_h), 2, border_radius=int(14 * s))
            msg = self.font_big_alert.render("DOCK SECURED", True, (45, 225, 125))
            sub = self.font_big_sub.render("PRESS [M] TO DRIVE FORWARD · [R] RESTART", True, (215, 245, 225))
            banner_surf.blit(msg, (banner_w // 2 - msg.get_width() // 2, int(16 * s)))
            banner_surf.blit(sub, (banner_w // 2 - sub.get_width() // 2, int(58 * s)))
            self.screen.blit(banner_surf, (self.width // 2 - banner_w // 2, self.height // 2 - banner_h // 2))

        # 6. Prominent Centered Alerts
        if self.kinematics.is_jackknifed(self.state):
            jk_w, jk_h = int(490 * s), int(115 * s)
            jk_surf = pygame.Surface((jk_w, jk_h), pygame.SRCALPHA)
            pygame.draw.rect(jk_surf, (28, 10, 14, 245), (0, 0, jk_w, jk_h), border_radius=int(14 * s))
            pygame.draw.rect(jk_surf, (255, 60, 65), (0, 0, jk_w, jk_h), 2, border_radius=int(14 * s))

            title = self.font_big_alert.render("JACKKNIFED", True, (255, 65, 70))
            sub = self.font_big_sub.render("HITCH LOCKED · PRESS [R] TO RESET", True, (235, 220, 225))
            jk_surf.blit(title, (jk_w // 2 - title.get_width() // 2, int(20 * s)))
            jk_surf.blit(sub, (jk_w // 2 - sub.get_width() // 2, int(68 * s)))
            self.screen.blit(jk_surf, (self.width // 2 - jk_w // 2, self.height // 2 - jk_h // 2))
        elif self.is_off_track:
            ot_w, ot_h = int(510 * s), int(115 * s)
            ot_surf = pygame.Surface((ot_w, ot_h), pygame.SRCALPHA)
            pygame.draw.rect(ot_surf, (36, 20, 12, 245), (0, 0, ot_w, ot_h), border_radius=int(14 * s))
            pygame.draw.rect(ot_surf, (255, 145, 45), (0, 0, ot_w, ot_h), 2, border_radius=int(14 * s))

            title = self.font_big_alert.render("OFF COURSE", True, (255, 145, 45))
            sub = self.font_big_sub.render(f"LATERAL DEVIATION > 6.0 m ({self.cross_track_err:.1f} m) · PRESS [R]", True, (255, 235, 220))
            ot_surf.blit(title, (ot_w // 2 - title.get_width() // 2, int(20 * s)))
            ot_surf.blit(sub, (ot_w // 2 - sub.get_width() // 2, int(68 * s)))
            self.screen.blit(ot_surf, (self.width // 2 - ot_w // 2, self.height // 2 - ot_h // 2))
        elif self.is_paused and not self.has_arrived:
            pause_w, pause_h = int(450 * s), int(116 * s)
            pause_surf = pygame.Surface((pause_w, pause_h), pygame.SRCALPHA)
            pygame.draw.rect(pause_surf, (15, 22, 34, 245), (0, 0, pause_w, pause_h), border_radius=int(14 * s))
            pygame.draw.rect(pause_surf, (70, 160, 245, 230), (0, 0, pause_w, pause_h), 2, border_radius=int(14 * s))
            pygame.draw.rect(pause_surf, (120, 195, 255, 60), (2, 2, pause_w - 4, pause_h - 4), 1, border_radius=int(12 * s))

            title = self.font_big_alert.render("PAUSED", True, (80, 185, 255))
            sub1 = self.font_big_sub.render("PRESS [SPACE] TO RESUME", True, (220, 235, 252))
            sub2 = self.font_mono.render("[R] Restart   [M] Mode   [P] Sample Path", True, (140, 165, 195))

            pause_surf.blit(title, (pause_w // 2 - title.get_width() // 2, int(18 * s)))
            pause_surf.blit(sub1, (pause_w // 2 - sub1.get_width() // 2, int(64 * s)))
            pause_surf.blit(sub2, (pause_w // 2 - sub2.get_width() // 2, int(90 * s)))
            self.screen.blit(pause_surf, (self.width // 2 - pause_w // 2, self.height // 2 - pause_h // 2))

    def render(self):
        # 1. Instant Blit of pre-baked static environment (clean yard boundary, road, F1 curbs, finish line, vignette)
        if self.background_surf is not None:
            self.screen.blit(self.background_surf, (0, 0))
        else:
            self.rebuild_static_background()
            if self.background_surf is not None:
                self.screen.blit(self.background_surf, (0, 0))

        if self.is_drawing and len(self.drawing_points) > 1:
            pts = [self.camera.project(p[0], p[1]) for p in self.drawing_points]
            pygame.draw.lines(self.screen, COLOR_ACCENT_CYAN, False, pts, 2)

        # 2. Clear and populate single unified dynamic alpha layer (Zero 4K allocation)
        self.dynamic_alpha_surf.fill((0, 0, 0, 0))
        self.draw_dynamic_finish_line_glow()
        self.draw_vehicle_shadows()
        self.draw_lighting()
        self.draw_predictive_trajectories()
        self.draw_attention_rays()

        # 3. Single instant alpha compositing pass
        self.screen.blit(self.dynamic_alpha_surf, (0, 0))

        # 4. Direct draw dynamic foreground elements onto screen
        self.draw_dynamic_finish_line_core()
        self.draw_truck_and_trailers()
        self.draw_hud()
        pygame.display.flip()

    def run(self):
        print("Launching Clean 2.5D Top-Down Simulation Game...")
        while self.is_running:
            self.handle_input()
            if not self.is_paused:
                substeps = self.time_multiplier
                # Decouple inference: neural policy runs at realistic control rate (~12-25Hz),
                # while RK4 handles sub-stepping smoothly
                infer_interval = 4 if substeps >= 8 else 1
                for step_i in range(substeps):
                    should_infer = (step_i % infer_interval == 0)
                    self.update_physics(run_inference=should_infer)
                    if self.has_arrived or self.kinematics.is_jackknifed(self.state) or self.is_off_track:
                        break

            # Per-frame camera updates
            self.camera.update()
            self.frame_count += 1

            self.render()
            self.clock.tick(60)

        pygame.quit()
        sys.exit(0)

if __name__ == "__main__":
    game = JackknifeGame25D("models/model.pt")
    game.run()
