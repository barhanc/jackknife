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
from src.path_generator import smooth_and_constrain_user_path, Path
from record_pure_lqr import PureLQRDrivingAgent

ARTIFACT_DIR = "/home/barto/.gemini/antigravity-cli/brain/7edd1242-1bd0-4ec5-9d3d-c75c7b69ab53"

COLOR_CYAN = (0, 229, 255)
COLOR_ORANGE = (255, 145, 45)
COLOR_GREEN = (45, 225, 125)
COLOR_RED = (255, 65, 70)
COLOR_WHITE = (240, 245, 255)
COLOR_MUTED = (160, 175, 195)
COLOR_PANEL = (18, 24, 34)

def build_step_function_chicane() -> Path:
    """
    Step function chicane with very short horizontal section:
    Lane 1 (y = -7.0m) -> steep step transition -> very short horizontal bridge (6m) -> steep step transition -> Lane 2 (y = +7.0m)
    """
    pts = [
        (-48.0, -7.0),
        (-20.0, -7.0),
        (-12.0, -7.0),
        (-6.0,  -6.0),
        (-2.0,   0.0), # Steep transition
        ( 1.0,   4.0),
        ( 4.0,   7.0),
        ( 9.0,   7.0), # Very short horizontal section (5 meters!)
        (14.0,   7.0),
        (22.0,   7.0),
        (48.0,   7.0)
    ]
    return smooth_and_constrain_user_path(pts, ds=0.8, min_radius=10.0)

def build_double_step_chicane() -> Path:
    """
    Double step function chicane:
    Step up to +7.5m -> 6m horizontal bridge -> immediate step down to -7.5m -> 6m bridge -> exit
    """
    pts = [
        (-46.0,  0.0),
        (-22.0,  0.0),
        (-14.0,  0.0),
        (-9.0,   6.5),  # Step 1 up
        (-3.0,   7.5),  # 6m short horizontal bridge
        ( 3.0,   7.5),
        ( 8.0,  -6.5),  # Immediate step 2 down
        (13.0,  -7.5),  # 6m short horizontal bridge
        (18.0,  -7.5),
        (23.0,   0.0),  # Step back to exit
        (46.0,   0.0)
    ]
    return smooth_and_constrain_user_path(pts, ds=0.8, min_radius=9.5)

def run_step_test(
    game: JackknifeGame25D,
    path: Path,
    agent_type: str, # "PFN" or "LQR"
    title_name: str,
    output_base: str,
    max_steps: int = 5000
) -> Dict[str, Any]:
    cx = (np.min(path.x) + np.max(path.x)) / 2.0
    cy = (np.min(path.y) + np.max(path.y)) / 2.0
    game.path = Path(np.array(path.x) - cx, np.array(path.y) - cy, ds=0.8)
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
    title_surf = game.font_title.render(f"TEST: {title_name.upper()} [{agent_type}]", True, COLOR_WHITE)
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

    print(f"[{title_name} - {agent_type}] Result: {status} | Steps: {step} | Dist: {dist_m:.1f}m | Mean CTE: {mean_cte:.3f}m | Max B2: {max_b2:.1f}° -> Saved {gif_path}")

    return {
        "title": title_name,
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
    print("Testing Step-Function Chicanes with Very Short Horizontal Sections...")
    game = JackknifeGame25D(model_path="models/pfn_best.pt")

    step_lane_path = build_step_function_chicane()
    double_step_path = build_double_step_chicane()

    # 1. Step Function Lane Shift (ControlPFN vs Pure LQR)
    run_step_test(game, step_lane_path, "PFN", "Step Lane Shift", "step_shift_pfn")
    run_step_test(game, step_lane_path, "LQR", "Step Lane Shift", "step_shift_lqr")

    # 2. Brutal Double Step Chicane (ControlPFN vs Pure LQR)
    run_step_test(game, double_step_path, "PFN", "Short-Bridge Chicane", "short_chicane_pfn")
    run_step_test(game, double_step_path, "LQR", "Short-Bridge Chicane", "short_chicane_lqr")

if __name__ == "__main__":
    main()
