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
from src.path_generator import Path
from record_pure_lqr import PureLQRDrivingAgent

ARTIFACT_DIR = "/home/barto/.gemini/antigravity-cli/brain/7edd1242-1bd0-4ec5-9d3d-c75c7b69ab53"

COLOR_CYAN = (0, 229, 255)
COLOR_ORANGE = (255, 145, 45)
COLOR_GREEN = (45, 225, 125)
COLOR_RED = (255, 65, 70)
COLOR_WHITE = (240, 245, 255)
COLOR_MUTED = (160, 175, 195)
COLOR_PANEL = (18, 24, 34)

def build_extreme_opposite_90_path(bridge_length_m: float = 4.0, radius_m: float = 8.0) -> Path:
    """
    Constructs an extreme S-dock with two full 90-degree turns in opposite directions
    separated by an ultra-short straight road of length `bridge_length_m`.
    1. 30m Straight Approach at y = -12m
    2. 90° Turn LEFT (R = radius_m)
    3. Ultra-short connecting road along +Y of length bridge_length_m (e.g. 4.0m)
    4. 90° Turn RIGHT (R = radius_m) in opposite direction back to parallel heading
    5. 30m Exit Straight
    """
    # Approach
    s_app = np.linspace(0, 30.0, 40)
    x_app = -45.0 + s_app
    y_app = np.full_like(s_app, -12.0)

    # Turn 1: 90 deg Left (R = radius_m)
    center1_x = -15.0
    center1_y = -12.0 + radius_m
    th1 = np.linspace(-math.pi/2, 0.0, 24)
    x_t1 = center1_x + radius_m * np.cos(th1)
    y_t1 = center1_y + radius_m * np.sin(th1)

    # Short connecting bridge going North along +Y
    t1_end_x = center1_x + radius_m
    t1_end_y = center1_y
    n_mid = max(3, int(bridge_length_m / 0.8))
    s_mid = np.linspace(0.8, bridge_length_m, n_mid)
    x_mid = np.full_like(s_mid, t1_end_x)
    y_mid = t1_end_y + s_mid

    # Turn 2: 90 deg Right in opposite direction (R = radius_m)
    # Starts at (t1_end_x, t1_end_y + bridge_length_m), heading North (+Y)
    # Center of turn 2 is to the right (+X) by radius_m:
    center2_x = t1_end_x + radius_m
    center2_y = t1_end_y + bridge_length_m
    th2 = np.linspace(math.pi, math.pi/2, 24)
    x_t2 = center2_x + radius_m * np.cos(th2)
    y_t2 = center2_y + radius_m * np.sin(th2)

    # Exit straight
    t2_end_x = center2_x
    t2_end_y = center2_y + radius_m
    s_exit = np.linspace(0.8, 40.0, 50)
    x_exit = t2_end_x + s_exit
    y_exit = np.full_like(s_exit, t2_end_y)

    raw_x = np.concatenate([x_app, x_t1[1:], x_mid, x_t2[1:], x_exit])
    raw_y = np.concatenate([y_app, y_t1[1:], y_mid, y_t2[1:], y_exit])

    return Path(raw_x, raw_y, ds=0.8, constrain_curvature=False)

def simulate_extreme_run(
    game: JackknifeGame25D,
    path: Path,
    agent_type: str,
    test_title: str,
    output_base: str,
    max_steps: int = 5500
) -> Dict[str, Any]:
    cx = (np.min(path.x) + np.max(path.x)) / 2.0
    cy = (np.min(path.y) + np.max(path.y)) / 2.0
    game.path = Path(np.array(path.x) - cx, np.array(path.y) - cy, ds=0.8, constrain_curvature=False)
    game.reset_vehicle_to_path()
    game.mode = "AUTONOMOUS"
    game.current_speed = -1.5

    lqr_agent = PureLQRDrivingAgent(game.dims) if agent_type == "LQR" else None

    t2_trail: List[Tuple[float, float]] = []
    cte_list: List[float] = []
    beta1_list: List[float] = []
    beta2_list: List[float] = []
    gif_frames: List[Image.Image] = []

    t_start = time.perf_counter()
    status = "IN_PROGRESS"
    step = 0

    ribbon_color = COLOR_CYAN if agent_type == "PFN" else COLOR_ORANGE

    for step in range(max_steps):
        x2, y2, _ = game.kinematics.get_trailer2_axle(game.state)
        t2_trail.append((x2, y2))
        cte_list.append(game.cross_track_err)
        beta1_list.append(math.degrees(abs(game.state.beta1)))
        beta2_list.append(math.degrees(abs(game.state.beta2)))

        if agent_type == "PFN":
            game.step_autonomous_policy()
            game.update_physics()
        else:
            game.target_steer = lqr_agent.compute_steer(game) # type: ignore
            game.update_physics(run_inference=False)

        if step % 25 == 0:
            game.render()
            if len(t2_trail) > 1:
                trail_surf = pygame.Surface((game.width, game.height), pygame.SRCALPHA)
                screen_pts = [game.camera.project(px, py) for (px, py) in t2_trail]
                pygame.draw.lines(trail_surf, (*ribbon_color, 80), False, screen_pts, 6)
                pygame.draw.lines(trail_surf, (*ribbon_color, 230), False, screen_pts, 3)
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
        pygame.draw.lines(trail_surf, (*ribbon_color, 90), False, screen_pts, 8)
        pygame.draw.lines(trail_surf, (*ribbon_color, 240), False, screen_pts, 3)
    game.screen.blit(trail_surf, (0, 0))

    # Badge
    badge_w, badge_h = 420, 165
    badge_rect = pygame.Rect(game.width - badge_w - 20, 20, badge_w, badge_h)
    pygame.draw.rect(game.screen, (*COLOR_PANEL, 240), badge_rect, border_radius=8)
    pygame.draw.rect(game.screen, (75, 85, 110), badge_rect, width=1, border_radius=8)

    status_color = COLOR_GREEN if status in ["ARRIVED", "COMPLETED"] else (COLOR_RED if status == "JACKKNIFED" else COLOR_ORANGE)
    title_surf = game.font_title.render(f"{test_title.upper()} [{agent_type}]", True, COLOR_WHITE)
    status_surf = game.font_title.render(f"STATUS: {status}", True, status_color)

    mean_cte = float(np.mean(cte_list)) if cte_list else 0.0
    max_b1 = float(np.max(beta1_list)) if beta1_list else 0.0
    max_b2 = float(np.max(beta2_list)) if beta2_list else 0.0
    dist_m = step * game.dt * 1.5

    stat1 = game.font_mono.render(f"Steps: {step} ({dist_m:.1f}m in {elapsed_time:.1f}s)", True, COLOR_MUTED)
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

    gif_path = os.path.join(ARTIFACT_DIR, f"{output_base}.gif")
    quant_frames = [
        f.resize((520, 334), Image.Resampling.BILINEAR).quantize(colors=64, method=Image.Resampling.NEAREST)
        for f in gif_frames[::2]
    ]
    if quant_frames:
        quant_frames.extend([quant_frames[-1]] * 8)
        quant_frames[0].save(gif_path, save_all=True, append_images=quant_frames[1:], duration=70, loop=0, optimize=True)

    print(f"[{test_title} - {agent_type}] Result: {status} | Steps: {step} | Dist: {dist_m:.1f}m | Mean CTE: {mean_cte:.3f}m | Max B2: {max_b2:.1f}° -> Saved {gif_path}")

    return {
        "title": test_title,
        "agent": agent_type,
        "status": status,
        "steps": step,
        "dist_m": dist_m,
        "mean_cte": mean_cte,
        "final_cte": game.cross_track_err,
        "max_b1": max_b1,
        "max_b2": max_b2,
        "elapsed_s": elapsed_time,
        "png_path": png_path,
        "gif_path": gif_path
    }

def main():
    print("Testing Two 90-Degree Turns in Opposite Directions with Short Bridges...")
    game = JackknifeGame25D(model_path="models/pfn_best.pt")

    # 1. 4-meter bridge (Ultra-short bridge between opposite 90° turns)
    path_4m = build_extreme_opposite_90_path(bridge_length_m=4.0, radius_m=8.5)
    simulate_extreme_run(game, path_4m, "PFN", "Two Opposite 90° (4m Bridge)", "opp_90_4m_pfn")
    simulate_extreme_run(game, path_4m, "LQR", "Two Opposite 90° (4m Bridge)", "opp_90_4m_lqr")

    # 2. 2-meter bridge (Practically back-to-back opposite 90° turns with near-zero road!)
    path_2m = build_extreme_opposite_90_path(bridge_length_m=2.0, radius_m=8.5)
    simulate_extreme_run(game, path_2m, "PFN", "Two Opposite 90° (2m Bridge)", "opp_90_2m_pfn")
    simulate_extreme_run(game, path_2m, "LQR", "Two Opposite 90° (2m Bridge)", "opp_90_2m_lqr")

if __name__ == "__main__":
    main()
