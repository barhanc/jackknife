import os
import sys
import math
import time
from typing import Dict, Any, List, Tuple
import numpy as np
from PIL import Image

os.environ["SDL_VIDEODRIVER"] = "dummy"
import pygame

from src.game import JackknifeGame25D
from src.path_generator import generate_synthetic_path, Path
from src.lqr_controller import ReverseLQRController

ARTIFACT_DIR = "/home/barto/.gemini/antigravity-cli/brain/7edd1242-1bd0-4ec5-9d3d-c75c7b69ab53"

COLOR_CYAN = (0, 229, 255)
COLOR_ORANGE = (255, 145, 45)
COLOR_GREEN = (45, 225, 125)
COLOR_RED = (255, 65, 70)
COLOR_WHITE = (240, 245, 255)
COLOR_MUTED = (160, 175, 195)
COLOR_PANEL = (18, 24, 34)

class PureLQRDrivingAgent:
    """
    Standard Pure LQR Path-Following Controller for Multi-Trailer Reversing.
    Uses instantaneous path curvature & lateral tracking error to compute nominal
    hitch angles, then applies Riccati feedback gain K to stabilize the hitch joints.
    Notice: It lacks Transformer lookahead attention / non-linear multi-step anticipation.
    """
    def __init__(self, dims):
        self.dims = dims
        self.lqr = ReverseLQRController(dims)

    def compute_steer(self, game: JackknifeGame25D) -> float:
        if not game.path:
            return 0.0

        x2, y2, th2 = game.kinematics.get_trailer2_axle(game.state)
        idx, cte = game.path.update_progression(x2, y2, game.current_path_idx)
        game.current_path_idx = idx
        game.cross_track_err = cte

        # Target heading for Trailer 2 (facing backwards along path direction)
        target_th2 = game.path.headings[idx] - math.pi
        heading_err = math.atan2(math.sin(th2 - target_th2), math.cos(th2 - target_th2))

        # Lateral error with sign
        dx = x2 - game.path.x[idx]
        dy = y2 - game.path.y[idx]
        th_p = game.path.headings[idx]
        signed_lat_err = -(dx * math.sin(th_p) - dy * math.cos(th_p))

        # Desired curvature for trailer 2
        path_curv = float(game.path.curvatures[idx])
        # Stanley-like lateral feedback gain
        k_lat = 0.35
        k_head = 1.2
        kappa_cmd = path_curv - k_lat * signed_lat_err - k_head * heading_err

        # Kinematic steady-state nominal hitch angles
        sin_b2 = -self.dims.L2 * kappa_cmd
        sin_b2 = float(np.clip(sin_b2, -0.75, 0.75))
        b2_nom = math.asin(sin_b2)

        cos_b2 = math.cos(b2_nom)
        sin_b1 = -self.dims.L1 * (kappa_cmd / max(0.2, cos_b2))
        sin_b1 = float(np.clip(sin_b1, -0.75, 0.75))
        b1_nom = math.asin(sin_b1)

        cos_b1 = math.cos(b1_nom)
        tan_phi = -self.dims.L0 * (kappa_cmd / max(0.2, cos_b1 * cos_b2))
        phi_nom = math.atan(tan_phi)
        phi_nom = float(np.clip(phi_nom, -self.dims.max_steer_rad, self.dims.max_steer_rad))

        # Apply LQR Riccati feedback regulation around nominal point
        phi_star = self.lqr.corrective_steering(
            nominal_phi=phi_nom,
            nominal_beta1=b1_nom,
            nominal_beta2=b2_nom,
            actual_beta1=game.state.beta1,
            actual_beta2=game.state.beta2,
            v0=-1.5
        )
        return phi_star

def run_pure_lqr_benchmark(output_prefix: str = "pure_lqr_"):
    print("Launching Pure LQR Controller benchmark on presets...")
    game = JackknifeGame25D(model_path=None) # Disable neural network
    agent = PureLQRDrivingAgent(game.dims)

    tracks = [
        ("Preset 1 (Alley Dock)", 1, None, 4000),
        ("Preset 5 (Gauntlet)", 5, None, 7000),
        ("Fast Chicane (Preset 3)", 3, None, 5000),
        ("Slalom (Serpentine)", 0, "yard_serpentine_slalom", 5000),
    ]

    for name, preset_id, path_type, max_steps in tracks:
        if preset_id > 0:
            game.load_preset_path(preset_id)
        else:
            raw = generate_synthetic_path(length_m=110.0, ds=0.8, path_type=str(path_type))
            cx = (np.min(raw.x) + np.max(raw.x)) / 2.0
            cy = (np.min(raw.y) + np.max(raw.y)) / 2.0
            game.preset_idx = 0
            game.path = Path(np.array(raw.x) - cx, np.array(raw.y) - cy, ds=0.8)

        game.reset_vehicle_to_path()
        game.mode = "AUTONOMOUS"
        game.current_speed = -1.5

        t2_trail: List[Tuple[float, float]] = []
        cte_list: List[float] = []
        beta1_list: List[float] = []
        beta2_list: List[float] = []
        gif_frames: List[Image.Image] = []

        t_start = time.perf_counter()
        status = "IN_PROGRESS"
        step = 0

        for step in range(max_steps):
            x2, y2, _ = game.kinematics.get_trailer2_axle(game.state)
            t2_trail.append((x2, y2))
            cte_list.append(game.cross_track_err)
            beta1_list.append(math.degrees(abs(game.state.beta1)))
            beta2_list.append(math.degrees(abs(game.state.beta2)))

            # Pure LQR steering command
            game.target_steer = agent.compute_steer(game)
            game.update_physics(run_inference=False)

            if step % 25 == 0:
                game.render()
                if len(t2_trail) > 1:
                    trail_surf = pygame.Surface((game.width, game.height), pygame.SRCALPHA)
                    screen_pts = [game.camera.project(px, py) for (px, py) in t2_trail]
                    pygame.draw.lines(trail_surf, (*COLOR_ORANGE, 80), False, screen_pts, 6)
                    pygame.draw.lines(trail_surf, (*COLOR_ORANGE, 230), False, screen_pts, 3)
                    game.screen.blit(trail_surf, (0, 0))

                raw_bytes = pygame.image.tostring(game.screen, "RGB")
                frame_img = Image.frombytes("RGB", (game.width, game.height), raw_bytes)
                gif_frames.append(frame_img)

            if game.kinematics.is_jackknifed(game.state):
                status = "JACKKNIFED"
                break
            if game.is_off_track:
                status = "OFF_TRACK"
                break
            if game.has_arrived:
                status = "ARRIVED"
                break

        if status == "IN_PROGRESS":
            is_completed = (game.path is not None) and (game.current_path_idx >= len(game.path.x) - 10)
            status = "COMPLETED" if is_completed else "TIMEOUT"

        elapsed_time = time.perf_counter() - t_start

        # Render final frame
        game.render()
        trail_surf = pygame.Surface((game.width, game.height), pygame.SRCALPHA)
        if len(t2_trail) > 1:
            screen_pts = [game.camera.project(px, py) for (px, py) in t2_trail]
            pygame.draw.lines(trail_surf, (*COLOR_ORANGE, 90), False, screen_pts, 8)
            pygame.draw.lines(trail_surf, (*COLOR_ORANGE, 240), False, screen_pts, 3)
        game.screen.blit(trail_surf, (0, 0))

        # Badge
        badge_w, badge_h = 400, 165
        badge_rect = pygame.Rect(game.width - badge_w - 20, 20, badge_w, badge_h)
        pygame.draw.rect(game.screen, (*COLOR_PANEL, 240), badge_rect, border_radius=8)
        pygame.draw.rect(game.screen, (100, 75, 60), badge_rect, width=1, border_radius=8)

        status_color = COLOR_GREEN if status in ["ARRIVED", "COMPLETED"] else (COLOR_RED if status == "JACKKNIFED" else COLOR_ORANGE)
        title_surf = game.font_title.render(f"PURE LQR: {name.upper()}", True, COLOR_WHITE)
        status_surf = game.font_title.render(f"STATUS: {status}", True, status_color)

        mean_cte = float(np.mean(cte_list)) if cte_list else 0.0
        max_b1 = float(np.max(beta1_list)) if beta1_list else 0.0
        max_b2 = float(np.max(beta2_list)) if beta2_list else 0.0
        dist_m = step * game.dt * 1.5

        stat1 = game.font_mono.render(f"Steps: {step} ({dist_m:.1f}m in {elapsed_time:.1f}s)", True, COLOR_MUTED)
        stat2 = game.font_mono.render(f"CTE Mean: {mean_cte:.3f}m | End: {game.cross_track_err:.3f}m", True, COLOR_MUTED)
        stat3 = game.font_mono.render(f"Max Hitch: B1={max_b1:.1f}° | B2={max_b2:.1f}°", True, COLOR_MUTED)

        game.screen.blit(title_surf, (badge_rect.x + 14, badge_rect.y + 12))
        game.screen.blit(status_surf, (badge_rect.x + 14, badge_rect.y + 36))
        game.screen.blit(stat1, (badge_rect.x + 14, badge_rect.y + 68))
        game.screen.blit(stat2, (badge_rect.x + 14, badge_rect.y + 94))
        game.screen.blit(stat3, (badge_rect.x + 14, badge_rect.y + 120))
        pygame.display.flip()

        base_clean = name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
        if "preset_1" in base_clean:
            fname = "pure_lqr_preset1"
        elif "preset_5" in base_clean:
            fname = "pure_lqr_preset5"
        elif "fast_chicane" in base_clean or "preset_3" in base_clean:
            fname = "pure_lqr_fast_chicane"
        else:
            fname = "pure_lqr_slalom"

        png_path = os.path.join(ARTIFACT_DIR, f"{fname}.png")
        pygame.image.save(game.screen, png_path)

        gif_path = os.path.join(ARTIFACT_DIR, f"{fname}.gif")
        quant_frames = [
            f.resize((520, 334), Image.Resampling.BILINEAR).quantize(colors=64, method=Image.Resampling.NEAREST)
            for f in gif_frames[::2]
        ]
        if quant_frames:
            quant_frames.extend([quant_frames[-1]] * 8)
            quant_frames[0].save(gif_path, save_all=True, append_images=quant_frames[1:], duration=70, loop=0, optimize=True)

        print(f"[PURE LQR - {name}] Result: {status} | Steps: {step} | Dist: {dist_m:.1f}m | Mean CTE: {mean_cte:.3f}m | Max B2: {max_b2:.1f}° -> Saved {gif_path}")

if __name__ == "__main__":
    run_pure_lqr_benchmark()
