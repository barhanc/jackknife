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

ARTIFACT_DIR = "/home/barto/.gemini/antigravity-cli/brain/7edd1242-1bd0-4ec5-9d3d-c75c7b69ab53"

# UI palette
COLOR_CYAN = (0, 229, 255)
COLOR_ORANGE = (255, 145, 45)
COLOR_GREEN = (45, 225, 125)
COLOR_RED = (255, 65, 70)
COLOR_WHITE = (240, 245, 255)
COLOR_MUTED = (160, 175, 195)
COLOR_PANEL = (18, 24, 34)

def simulate_and_record(
    game: JackknifeGame25D,
    name: str,
    output_base: str,
    max_steps: int = 7000,
    record_gif: bool = True,
    gif_fps_interval: int = 25
) -> Dict[str, Any]:
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

        game.step_autonomous_policy()
        game.update_physics()

        # Capture frame for GIF
        if record_gif and (step % gif_fps_interval == 0):
            game.render()
            if len(t2_trail) > 1:
                trail_surf = pygame.Surface((game.width, game.height), pygame.SRCALPHA)
                screen_pts = [game.camera.project(px, py) for (px, py) in t2_trail]
                pygame.draw.lines(trail_surf, (*COLOR_CYAN, 80), False, screen_pts, 6)
                pygame.draw.lines(trail_surf, (*COLOR_CYAN, 230), False, screen_pts, 3)
                game.screen.blit(trail_surf, (0, 0))

            raw_bytes = pygame.image.tostring(game.screen, "RGB")
            frame_img = Image.frombytes("RGB", (game.width, game.height), raw_bytes)
            # Resize to clean 700x450 for fast loading & smooth playback
            small_frame = frame_img.resize((700, 450), Image.Resampling.BILINEAR)
            gif_frames.append(small_frame)

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

    # Render final high-res frame
    game.render()

    # Draw the complete driven breadcrumb trail of Trailer 2
    trail_surf = pygame.Surface((game.width, game.height), pygame.SRCALPHA)
    if len(t2_trail) > 1:
        screen_pts = [game.camera.project(x, y) for (x, y) in t2_trail]
        pygame.draw.lines(trail_surf, (*COLOR_CYAN, 90), False, screen_pts, 8)
        pygame.draw.lines(trail_surf, (*COLOR_CYAN, 240), False, screen_pts, 3)

    game.screen.blit(trail_surf, (0, 0))

    # Render custom telemetry overlay badge on top-right
    badge_w, badge_h = 380, 165
    badge_rect = pygame.Rect(game.width - badge_w - 20, 20, badge_w, badge_h)
    pygame.draw.rect(game.screen, (*COLOR_PANEL, 240), badge_rect, border_radius=8)
    pygame.draw.rect(game.screen, (70, 85, 115), badge_rect, width=1, border_radius=8)

    status_color = COLOR_GREEN if status in ["ARRIVED", "COMPLETED"] else (COLOR_RED if status == "JACKKNIFED" else COLOR_ORANGE)

    title_surf = game.font_title.render(f"TEST: {name.upper()}", True, COLOR_WHITE)
    status_surf = game.font_title.render(f"STATUS: {status}", True, status_color)

    mean_cte = float(np.mean(cte_list)) if cte_list else 0.0
    max_b1 = float(np.max(beta1_list)) if beta1_list else 0.0
    max_b2 = float(np.max(beta2_list)) if beta2_list else 0.0
    dist_m = step * game.dt * 1.5

    stat1 = game.font_mono.render(f"Steps: {step} ({dist_m:.1f}m reversed in {elapsed_time:.1f}s)", True, COLOR_MUTED)
    stat2 = game.font_mono.render(f"CTE Mean: {mean_cte:.3f}m | Final: {game.cross_track_err:.3f}m", True, COLOR_MUTED)
    stat3 = game.font_mono.render(f"Max Hitch: B1={max_b1:.1f}° | B2={max_b2:.1f}°", True, COLOR_MUTED)

    game.screen.blit(title_surf, (badge_rect.x + 14, badge_rect.y + 12))
    game.screen.blit(status_surf, (badge_rect.x + 14, badge_rect.y + 36))
    game.screen.blit(stat1, (badge_rect.x + 14, badge_rect.y + 68))
    game.screen.blit(stat2, (badge_rect.x + 14, badge_rect.y + 94))
    game.screen.blit(stat3, (badge_rect.x + 14, badge_rect.y + 120))

    pygame.display.flip()

    png_path = os.path.join(ARTIFACT_DIR, f"{output_base}.png")
    pygame.image.save(game.screen, png_path)

    gif_path = ""
    if record_gif and gif_frames:
        gif_path = os.path.join(ARTIFACT_DIR, f"{output_base}.gif")
        # Quantize and optimize frames for lightweight remote viewing (< 1.5MB)
        quant_frames = [
            f.resize((520, 334), Image.Resampling.BILINEAR).quantize(colors=64, method=Image.Resampling.NEAREST)
            for f in gif_frames[::2]
        ]
        if quant_frames:
            # Hold final frame for 1.5s
            quant_frames.extend([quant_frames[-1]] * 8)
            quant_frames[0].save(
                gif_path,
                save_all=True,
                append_images=quant_frames[1:],
                duration=70,
                loop=0,
                optimize=True
            )

    print(f"[{name}] Result: {status} | Steps: {step} | Dist: {dist_m:.1f}m | Mean CTE: {mean_cte:.3f}m | Max B2: {max_b2:.1f}° -> Saved PNG & GIF")

    return {
        "name": name,
        "status": status,
        "steps": step,
        "distance_m": dist_m,
        "mean_cte": mean_cte,
        "final_cte": game.cross_track_err,
        "max_beta1_deg": max_b1,
        "max_beta2_deg": max_b2,
        "elapsed_s": elapsed_time,
        "image_path": png_path,
        "gif_path": gif_path
    }

def record_all_presets(model_path: str = "models/pfn_best.pt", prefix: str = "", record_gif: bool = True) -> List[Dict[str, Any]]:
    print(f"Initializing headless Jackknife simulation with model {model_path}...")
    game = JackknifeGame25D(model_path=model_path)

    results = []

    # 1. Preset 1: 90-Degree Standard Warehouse Alley Dock
    game.load_preset_path(1)
    results.append(simulate_and_record(game, "Preset 1 (Alley Dock)", f"{prefix}run_preset1", max_steps=4000, record_gif=record_gif))

    # 2. Preset 5: Extreme Gauntlet (Four Hard 90° Turns & Chicanes)
    game.load_preset_path(5)
    results.append(simulate_and_record(game, "Preset 5 (Gauntlet)", f"{prefix}run_preset5", max_steps=7000, record_gif=record_gif))

    # 3. Fast Chicane: Preset 3 (Diagonal Z-Chicane Reversing Lane)
    game.load_preset_path(3)
    results.append(simulate_and_record(game, "Fast Chicane (Preset 3 Z-Chicane)", f"{prefix}run_fast_chicane", max_steps=5000, record_gif=record_gif))

    # 4. Slalom: Procedural Yard Serpentine Slalom
    raw_slalom = generate_synthetic_path(length_m=110.0, ds=0.8, path_type="yard_serpentine_slalom")
    game.preset_idx = 0
    cx = (np.min(raw_slalom.x) + np.max(raw_slalom.x)) / 2.0
    cy = (np.min(raw_slalom.y) + np.max(raw_slalom.y)) / 2.0
    game.path = Path(np.array(raw_slalom.x) - cx, np.array(raw_slalom.y) - cy, ds=0.8)
    results.append(simulate_and_record(game, "Slalom (Serpentine)", f"{prefix}run_slalom", max_steps=5000, record_gif=record_gif))

    return results

if __name__ == "__main__":
    model_p = sys.argv[1] if len(sys.argv) > 1 else "models/pfn_best.pt"
    pref = sys.argv[2] if len(sys.argv) > 2 else ""
    record_all_presets(model_p, pref, record_gif=True)
