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
from .hud import AMBER, GameHUD
from .depot import DepotTerrain
from .lighting import TruckLighting
from .vehicle import VehicleRenderer
from .weather import Snowfall

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

COLOR_T1_ROOF         = (83, 131, 172)     # Lead trailer: fleet blue
COLOR_T1_BEVEL        = (38, 66, 95)
COLOR_T2_ROOF         = (87, 155, 140)     # Rear trailer: contrasting sea green
COLOR_T2_BEVEL        = (38, 83, 76)
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
    Guarantees consistent physical scale and spatial intuition across all routes
    and user drawings without camera jump, shift, or zoom distortions.
    """
    def __init__(self, width: int, height: int, yard_w: float = 160.0, yard_h: float = 100.0):
        self.width = width
        self.height = height
        self.viewport = pygame.Rect(0, 0, width, height)
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
        scale_x = max(1.0, self.viewport.width - margin_x) / self.yard_w
        scale_y = max(1.0, self.viewport.height - margin_y) / self.yard_h
        self.scale = min(scale_x, scale_y)

    def set_viewport(self, rect):
        self.viewport = pygame.Rect(rect)
        self.width, self.height = self.viewport.size
        self.update_scale()

    def fit_to_bounds(self, min_x: Optional[float] = None, max_x: Optional[float] = None, min_y: Optional[float] = None, max_y: Optional[float] = None, instant: bool = False):
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
        sx = int(self.viewport.centerx + (wx - self.cx) * self.scale)
        sy = int(self.viewport.centery - (wy - self.cy) * self.scale)
        return sx, sy

    def unproject_ground(self, sx: int, sy: int) -> Tuple[float, float]:
        wx = self.cx + (sx - self.viewport.centerx) / self.scale
        wy = self.cy + (self.viewport.centery - sy) / self.scale
        return wx, wy


class JackknifeGame25D:
    def __init__(self, model_path: Optional[str] = "models/model.pt"):
        pygame.init()
        pygame.font.init()
        self.width, self.height = 1400, 900
        self.screen = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE | pygame.DOUBLEBUF)
        pygame.display.set_caption("B-Double Truck Reversing Simulator")
        self.clock = pygame.time.Clock()

        # Dynamic Resolution-Responsive Typography (Scales crisp and readable from 1080p to 4K)
        self.ui_scale = 1.0
        self._init_fonts()

        self.dims = VehicleDimensions()
        self.kinematics = TractorDoubleTrailerKinematics(self.dims)
        self.camera = TopDownCamera(self.width, self.height, yard_w=160.0, yard_h=100.0)
        self.terrain = DepotTerrain(self.camera.yard_w, self.camera.yard_h)
        self.lighting = TruckLighting()
        self.snowfall = Snowfall()

        # PyTorch Model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cpu":
            # Batch-one inference is faster without CPU thread-pool overhead.
            torch.set_num_threads(1)
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
        self.simulation_accumulator = 0.0
        self.steps_until_inference = 0
        self.current_speed = 0.0
        self.target_steer = 0.0
        self.current_path_idx = 0  # Monotonic curve arc-length tracker (handles 8-shapes and loops)
        self.time_multiplier = 1   # Fast-forward simulation speed warp: 1x through 16x
        self.speed_options = [1, 2, 4, 8, 16]

        # Drawing mode
        self.drawing_points: List[Tuple[float, float]] = []
        self.is_drawing = False
        self.drawing_mode = False

        # Pre-allocated High-Performance Surfaces (Zero memory allocation in 4K render loop)
        self.background_surf: Optional[pygame.Surface] = None
        self.dynamic_alpha_surf = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        self.vignette_surf = None
        self._build_vignette()

        # Path & Procedural Yard
        self.path: Optional[Path] = None

        # Telemetry & Atmosphere
        self.cross_track_err = 0.0
        self.inference_time_ms = 0.0
        self._inference_stats_time = 0.0
        self.attention_weights = np.ones(64) / 64.0
        self.action_probs = np.ones(15) / 15.0
        self.show_help = False
        self.show_rays = False
        self.frame_count = 0
        self.hud = GameHUD(self)
        self.camera.set_viewport(self.hud.yard_rect)
        self._build_vignette()
        self.load_default_path()

    def _init_fonts(self):
        """
        Dynamically scales typography according to current display resolution.
        Ensures HUD text, meters, and diagnostics remain crisp and easily readable on 4K monitors.
        """
        self.ui_scale = max(1.0, min(self.width / 1400.0, self.height / 900.0))
        s = self.ui_scale
        font_dir = os.path.join(os.path.dirname(__file__), "assets", "fonts")
        sans_bold = os.path.join(font_dir, "DejaVuSans-Bold.ttf")
        mono = os.path.join(font_dir, "DejaVuSansMono.ttf")
        mono_bold = os.path.join(font_dir, "DejaVuSansMono-Bold.ttf")
        arcade = os.path.join(font_dir, "PressStart2P-Regular.ttf")
        self.font_arcade = pygame.font.Font(arcade, max(10, int(10 * s)))
        self.font_arcade_small = pygame.font.Font(arcade, max(8, int(8 * s)))
        self.font_arcade_title = pygame.font.Font(arcade, max(14, int(18 * s)))
        self.font_arcade_alert = pygame.font.Font(arcade, max(16, int(24 * s)))
        self.font_big_alert = pygame.font.Font(sans_bold, max(18, int(34 * s)))
        self.font_big_sub   = pygame.font.Font(mono_bold, max(11, int(15 * s)))
        self.font_alert     = pygame.font.Font(sans_bold, max(14, int(22 * s)))
        self.font_title     = pygame.font.Font(sans_bold, max(11, int(15 * s)))
        self.font_mono      = pygame.font.Font(mono, max(10, int(13 * s)))
        self.font_mono_bold = pygame.font.Font(mono_bold, max(10, int(14 * s)))
        self.font_badge     = pygame.font.Font(sans_bold, max(9, int(12 * s)))
        self.font_sm        = pygame.font.Font(mono, max(8, int(11 * s)))

    def _build_vignette(self):
        self.vignette_surf = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        diag = math.hypot(self.camera.viewport.width, self.camera.viewport.height)
        center = self.camera.viewport.center
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

    def load_default_path(self):
        """Four-turn alley-dock course (the former preset 5)."""
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
                self.path = Path(centered_x, centered_y, ds=0.8, constrain_curvature=False)
                print(f"Sampled random prior path: '{ptype}' (Length: {self.path.total_length:.1f}m, max_k={np.max(np.abs(self.path.curvatures)):.3f})")
                self.generate_procedural_yard()
                self.reset_vehicle_to_path()
                return

        # Fallback to guaranteed-safe centered S-curve if random proposals exceeded margins
        fallback = generate_synthetic_path(length_m=72.0, ds=0.8, max_curvature=0.08, path_type="multi_scale_s")
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
        self.simulation_accumulator = 0.0
        self.steps_until_inference = 0
        self.steer_hold_time = 0.0
        self.has_arrived = False
        self.is_off_track = False
        self.cross_track_err = 0.0
        self.inference_time_ms = 0.0
        self._inference_stats_time = 0.0
        self.attention_weights = np.ones(64) / 64.0
        self.action_probs = np.ones(15) / 15.0
        self.auto_frame_camera(instant=True)
        self.terrain.configure(self.path, self.kinematics, self.state)
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
        Bake the depot, route and loading bay once per path or viewport change.
        """
        if self.width <= 0 or self.height <= 0:
            return
        # Explicit depth keeps this opaque: macOS display formats can otherwise
        # contribute an unwanted alpha channel that breaks overlay compositing.
        self.background_surf = pygame.Surface((self.width, self.height), depth=32)
        self.draw_night_ground(self.background_surf)
        if not self.drawing_mode:
            self.draw_path_static(self.background_surf)
        if self.vignette_surf is not None:
            self.background_surf.blit(self.vignette_surf, (0, 0))

    def generate_procedural_yard(self):
        self.terrain.invalidate()

    def update_controls(self, keys):
        """Apply held controls once per physics step, including during fast-forward."""
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

    def update_hud_layout(self):
        self.hud.layout()
        self.camera.set_viewport(self.hud.yard_rect)
        self._build_vignette()
        self.rebuild_static_background()

    def cycle_speed(self):
        index = self.speed_options.index(self.time_multiplier)
        self.time_multiplier = self.speed_options[(index + 1) % len(self.speed_options)]

    def toggle_control_mode(self):
        self.mode = "AUTONOMOUS" if self.mode == "MANUAL" else "MANUAL"
        self.ai_save_active = False
        self.steps_until_inference = 0
        self.steer_hold_time = 0.0
        if self.mode == "MANUAL":
            self.has_arrived = False
            self.is_off_track = False
        else:
            self.time_multiplier = 16

    def start_drawing_mode(self):
        if self.drawing_mode:
            return
        self.drawing_mode = True
        self.is_drawing = False
        self.drawing_points = []
        self.simulation_accumulator = 0.0
        self.update_hud_layout()

    def finish_drawing(self, cancel=False):
        new_path = None
        if not cancel and len(self.drawing_points) >= 4:
            new_path = smooth_and_constrain_user_path(self.drawing_points, ds=0.8, lead_in_m=12.0)
        self.drawing_mode = False
        self.is_drawing = False
        self.drawing_points = []
        if new_path is not None:
            self.path = new_path
        self.update_hud_layout()
        if new_path is not None:
            self.generate_procedural_yard()
            self.reset_vehicle_to_path()

    def drawing_point(self, pos):
        wx, wy = self.camera.unproject_ground(*pos)
        half_w = self.camera.yard_w / 2.0 - 7.0
        half_h = self.camera.yard_h / 2.0 - 7.0
        return max(-half_w, min(half_w, wx)), max(-half_h, min(half_h, wy))

    def handle_input(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.is_running = False
            elif event.type == pygame.MOUSEWHEEL:
                if self.hud.drawer_rect.collidepoint(pygame.mouse.get_pos()):
                    self.hud.scroll(event.y)
            elif event.type == pygame.VIDEORESIZE:
                self.width, self.height = max(320, event.w), max(320, event.h)
                self.screen = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE | pygame.DOUBLEBUF)
                self._init_fonts()
                self.dynamic_alpha_surf = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
                self.update_hud_layout()
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if self.hud.click(event.pos):
                    continue
                if not self.camera.viewport.collidepoint(event.pos):
                    continue
                wx, wy = self.camera.unproject_ground(*event.pos)
                if abs(wx) > self.camera.yard_w/2 or abs(wy) > self.camera.yard_h/2:
                    continue
                self.start_drawing_mode()
                self.is_drawing = True
                self.drawing_points = [self.drawing_point(event.pos)]
            elif event.type == pygame.MOUSEMOTION and self.is_drawing:
                point = self.drawing_point(event.pos)
                if math.hypot(point[0]-self.drawing_points[-1][0], point[1]-self.drawing_points[-1][1]) > 0.5:
                    self.drawing_points.append(point)
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1 and self.is_drawing:
                self.finish_drawing()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_v:
                    self.show_rays = not self.show_rays
                    continue
                if self.drawing_mode:
                    if event.key == pygame.K_ESCAPE:
                        self.finish_drawing(cancel=True)
                    continue
                if event.key == pygame.K_e:
                    self.start_drawing_mode()
                elif event.key == pygame.K_m:
                    self.toggle_control_mode()
                elif event.key == pygame.K_r:
                    self.reset_vehicle_to_path()
                elif event.key == pygame.K_SPACE:
                    self.is_paused = not self.is_paused
                elif event.key == pygame.K_t:
                    self.cycle_speed()
                elif event.key == pygame.K_p:
                    self.sample_random_prior_path()
                elif event.key == pygame.K_h:
                    self.show_help = not self.show_help
                    self.update_hud_layout()

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

        t_end = time.perf_counter()
        sample_ms = (t_end - t_start) * 1000.0
        # Half-second wall-time smoothing behaves consistently at every speed.
        previous = self._inference_stats_time
        weight = 1.0 - math.exp(-(t_end - previous) / .5) if previous else 1.0
        self.inference_time_ms += weight * (sample_ms - self.inference_time_ms)
        self._inference_stats_time = t_end

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

            if self.cross_track_err > 5.5 or (self.current_path_idx <= 5 and cab_lat_err > 5.5):
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
        self.terrain.draw_ground(surf, self.camera, self.font_mono_bold, editing=self.drawing_mode)

    def draw_path_static(self, surf: pygame.Surface):
        self.terrain.draw_route(surf, self.camera, self.path, self.font_mono_bold)

    def draw_dynamic_finish_line_glow(self):
        self.terrain.draw_dock_lights(self.dynamic_alpha_surf, self.camera, self.has_arrived, glow=True)

    def draw_dynamic_finish_line_core(self):
        self.terrain.draw_dock_lights(self.screen, self.camera, self.has_arrived)

    def draw_lighting(self):
        self.lighting.draw_ground(
            self.dynamic_alpha_surf, self.camera, self.state, self.dims,
            self.kinematics.get_trailer2_axle(self.state), self.current_speed < -0.1,
        )

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
                for part in ('cab', 'trailer1', 'trailer2'):
                    pts = [self.camera.project(p[0], p[1]) for p in polys[part]]
                    pygame.draw.polygon(ghost_surf, (255, 255, 255, alpha), pts, 1)

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


    def draw_vehicle_shadows(self):
        VehicleRenderer().draw_shadows(
            self.dynamic_alpha_surf, self.camera, self.state, self.dims, self.kinematics,
        )

    def draw_truck_and_trailers(self):
        VehicleRenderer().draw(self.screen, self.camera, self.state, self.dims, self.kinematics)

    # ==========================================================================
    # THEMATIC FROSTED GLASS HUD WITH REFINED DISTRIBUTION PLOT
    # ==========================================================================

    def draw_hud(self):
        self.hud.draw()

    def render(self):
        self.screen.fill(COLOR_NIGHT_BG)
        self.screen.set_clip(self.camera.viewport)
        # 1. Instant Blit of pre-baked static environment (clean yard boundary, road, F1 curbs, finish line, vignette)
        if self.background_surf is not None:
            self.screen.blit(self.background_surf, (0, 0))
        else:
            self.rebuild_static_background()
            if self.background_surf is not None:
                self.screen.blit(self.background_surf, (0, 0))

        if self.is_drawing and len(self.drawing_points) > 1:
            pts = [self.camera.project(p[0], p[1]) for p in self.drawing_points]
            pygame.draw.lines(self.screen, AMBER, False, pts, 2)

        # 2. Clear and populate single unified dynamic alpha layer (Zero 4K allocation)
        if not self.drawing_mode:
            self.dynamic_alpha_surf.fill((0, 0, 0, 0))
            self.draw_dynamic_finish_line_glow()
            self.draw_vehicle_shadows()
            self.draw_lighting()
            self.draw_predictive_trajectories()
            self.draw_attention_rays()
            self.screen.blit(self.dynamic_alpha_surf, (0, 0))
            self.draw_dynamic_finish_line_core()
            self.draw_truck_and_trailers()
            self.snowfall.draw(self.screen, self.camera.viewport)
        self.screen.set_clip(None)
        self.draw_hud()
        pygame.display.flip()

    def advance_simulation(self, elapsed_seconds: float, keys=None):
        """Advance fixed steps with the same policy cadence at every speed.

        Ignore wall-time beyond 250 ms after a stall (e.g. dragging the window)
        so resuming cannot trigger an unbounded physics catch-up loop.
        """
        if self.is_paused or getattr(self, "drawing_mode", False):
            self.simulation_accumulator = 0.0
            return
        if keys is None:
            keys = pygame.key.get_pressed()
        self.simulation_accumulator += min(max(elapsed_seconds, 0.0), 0.25) * self.time_multiplier
        while self.simulation_accumulator + 1e-12 >= self.dt:
            self.update_controls(keys)
            # 12.5 Hz in simulation time at every speed; physics stays at 50 Hz.
            # Carry the phase across rendered frames to preserve trajectories.
            self.update_physics(run_inference=self.steps_until_inference == 0)
            self.steps_until_inference = (self.steps_until_inference - 1) % 4
            self.simulation_accumulator = max(0.0, self.simulation_accumulator - self.dt)
            if self.has_arrived or self.kinematics.is_jackknifed(self.state) or self.is_off_track:
                self.simulation_accumulator = 0.0
                break

    def run(self):
        print("Launching Clean 2.5D Top-Down Simulation Game...")
        previous_time = time.perf_counter()
        while self.is_running:
            self.clock.tick(60)
            now = time.perf_counter()
            elapsed_seconds = now - previous_time
            previous_time = now
            self.handle_input()
            if not self.is_running:
                break
            self.advance_simulation(elapsed_seconds)

            # Per-frame camera updates
            self.camera.update()
            self.frame_count += 1

            self.render()

        pygame.quit()
        sys.exit(0)

if __name__ == "__main__":
    game = JackknifeGame25D("models/model.pt")
    game.run()
